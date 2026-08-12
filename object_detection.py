import torch
import math
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageOps
from transformers.utils import logging as hf_logging
from transformers import (
    CLIPProcessor,
    CLIPModel,
    OwlViTProcessor,
    OwlViTForObjectDetection,
)

hf_logging.set_verbosity_error()

# 預設設定區

DEFAULT_QUERY_IMAGE_PATH = Path("Content/Brad-Pitt.jpg")
DEFAULT_CANDIDATE_DIR = Path("Style")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 物件偵測設定
OBJECT_LABELS = [
    # 人
    "person",
    "face",

    # 動物
    "dog",
    "cat",
    "bird",
    "horse",

    # 交通工具
    "car",
    "bus",
    "motorcycle",
    "bicycle",

    # 戶外物件
    "tree",
    "plant",
    "building",
    "house",

    # 室內物件
    "table",
    "chair",
    "sofa",
    "bed",
    "desk",
    "door",
    
]

# 只有這些穩定物件會被當成硬性必要條件。
IMPORTANT_REQUIRED_LABELS = {
    # 人與動物
    "person",
    "dog",
    "cat",
    "bird",
    "horse",

    # 交通工具
    "car",
    "bus",
    "motorcycle",
    "bicycle",

    # 穩定戶外物件
    "tree",
    "building",
    "house",

    # 穩定室內物件
    "table",
    "chair",
    "sofa",
    "bed",
    "desk",
    "door",

}

# Level 2：語意類別相容設定
SEMANTIC_GROUPS = {
    # 人像 / 人臉類
    "portrait": {"face","person"},

    # 動物類
    "animal": {"dog","cat","bird","horse"},

    # 交通工具類
    "vehicle": {"car","bus","motorcycle","bicycle"},

    # 建築類
    "architecture": {"building","house"},

    # 自然物件類
    "nature": {"tree","plant"},

    # 室內家具/結構類
    "indoor_furniture": {"table","chair","sofa","bed","desk","door"},

}

# 語意群組優先順序
SEMANTIC_GROUP_PRIORITY = [
    "portrait",
    "animal",
    "vehicle",
    "architecture",
    "indoor_furniture",
    "indoor_structure",
    "nature",
]

def get_label_best_score(detected_objects, label):
    """
    取得某個 label 在 detected_objects 中的最高偵測分數。
    如果該 label 不存在，回傳 0。
    """
    if label not in detected_objects:
        return 0.0

    if len(detected_objects[label]) == 0:
        return 0.0

    return max(
        obj["score"]
        for obj in detected_objects[label]
    )

def get_query_semantic_group(query_detected_objects):
    """
    根據 query 圖片偵測到的物件，判斷 query 的主要語意類別。

    回傳：
        group_name，例如 "portrait", "vehicle", "architecture"
        如果無法判斷，回傳 None
    """
    group_scores = {}

    for group_name, labels in SEMANTIC_GROUPS.items():
        score_sum = 0.0

        for label in labels:
            score_sum += get_label_best_score(
                query_detected_objects,
                label,
            )

        if score_sum > 0:
            group_scores[group_name] = score_sum

    if len(group_scores) == 0:
        return None

    # 先依照語意優先順序判斷。
    # 若該 group 有分數，就直接回傳。
    for group_name in SEMANTIC_GROUP_PRIORITY:
        if group_name in group_scores:
            return group_name

    # 保險：如果有 group 不在 priority 裡，就回傳分數最高的 group。
    return max(
        group_scores,
        key=group_scores.get,
    )

# 同一個 label 最多保留幾個 bounding boxes。
MAX_BOXES_PER_LABEL = 5
# 物件偵測最低信心門檻。
OBJECT_THRESHOLD = 0.10

# 場景分數設定
SCENE_TOP_K = 3

TOP1_SCENE_WEIGHT = 0.60
TOPK_SCENE_WEIGHT = 0.40

SCENE_LABELS = [
    # 城市與道路
    "city street",
    "street scene",
    "road scene",
    "parking lot",
    "residential area",
    "highway",
    "urban area",

    # 自然與戶外背景
    "forest road",
    "beach",
    "mountain",
    "farm",
    "outdoor landscape",
    "clear sky",
    "cloudy sky",
    "grassy field",

    # 室內
    "indoor room",
    "living room",
    "bedroom",
    "office",
    "restaurant",

]

# 總分權重
WEIGHTS = {
    "position": 0.35,
    "clip": 0.45,
    "scene": 0.20,
}

# 位置分數的嚴格程度
POSITION_SIGMA = 0.35

# 模型初始化
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model_name = "openai/clip-vit-base-patch32"
owlvit_model_name = "google/owlvit-base-patch32"

clip_model = None
clip_processor = None
owl_processor = None
owl_model = None

# 基本工具函式
def load_retrieval_models():
    """
    延後載入 CLIP 和 OWL-ViT 模型。
    """
    global clip_model
    global clip_processor
    global owl_processor
    global owl_model

    if clip_model is not None and owl_model is not None:
        return

    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    owl_processor = OwlViTProcessor.from_pretrained(owlvit_model_name)
    owl_model = OwlViTForObjectDetection.from_pretrained(owlvit_model_name).to(device)

    clip_model.eval()
    owl_model.eval()

def normalize_weights(weights):
    """
    權重總和不一定要剛好是 1。
    這個函式會自動正規化。
    """
    total = sum(weights.values())

    if total <= 0:
        raise ValueError("WEIGHTS 的總和必須大於 0。")

    return {
        key: value / total
        for key, value in weights.items()
    }

def normalize_scene_weights():
    """
    正規化場景分數內部權重。
    """
    total = TOP1_SCENE_WEIGHT + TOPK_SCENE_WEIGHT

    if total <= 0:
        raise ValueError("TOP1_SCENE_WEIGHT + TOPK_SCENE_WEIGHT 必須大於 0。")

    return TOP1_SCENE_WEIGHT / total, TOPK_SCENE_WEIGHT / total

def get_image_paths(folder: Path):
    """
    取得資料夾中的所有圖片路徑。
    """
    if not folder.exists():
        raise FileNotFoundError(f"找不到資料夾：{folder}")

    image_paths = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    return sorted(image_paths)

def validate_query_image(query_path: Path):
    """
    檢查 query 圖片是否存在、是否為支援格式。
    """
    if not query_path.exists():
        raise FileNotFoundError(f"找不到指定的 query 圖片：{query_path}")

    if not query_path.is_file():
        raise ValueError(f"指定的 query path 不是檔案：{query_path}")

    if query_path.suffix.lower() not in IMAGE_EXTS:
        raise ValueError(f"指定的 query 檔案不是支援的圖片格式：{query_path}")

def load_image(path: Path):
    """
    載入圖片，並修正手機照片常見的 EXIF 旋轉問題。
    """
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")

# CLIP：圖片 embedding
@torch.no_grad()
def maybe_apply_visual_projection(features):
    """
    只在 features 維度符合 visual_projection 輸入維度時才投影。

    CLIP visual_projection 通常是：
        768 -> 512

    如果 features 已經是 512 維，就代表它已經是 image embedding，
    不可以再丟進 visual_projection。
    """
    if not hasattr(clip_model, "visual_projection"):
        return features

    projection = clip_model.visual_projection

    if not hasattr(projection, "in_features"):
        return features

    expected_input_dim = projection.in_features
    current_dim = features.shape[-1]

    if current_dim == expected_input_dim:
        return projection(features)

    return features

def extract_clip_image_features(outputs):
    """
    兼容不同 transformers 版本的 CLIP image feature 輸出。
    """
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        pooled = outputs.pooler_output
        return maybe_apply_visual_projection(pooled)

    raise TypeError(
        f"無法從 CLIP image outputs 取出 Tensor features，實際型別是：{type(outputs)}"
    )

def maybe_apply_text_projection(features):
    """
    只在 features 維度符合 text_projection 輸入維度時才投影。

    如果 features 已經是 CLIP text embedding 維度，例如 512，
    就不要再丟進 text_projection。
    """
    if not hasattr(clip_model, "text_projection"):
        return features

    projection = clip_model.text_projection

    if not hasattr(projection, "in_features"):
        return features

    expected_input_dim = projection.in_features
    current_dim = features.shape[-1]

    if current_dim == expected_input_dim:
        return projection(features)

    return features

def extract_clip_text_features(outputs):
    """
    兼容不同 transformers 版本的 CLIP text feature 輸出。
    """
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        pooled = outputs.pooler_output
        return maybe_apply_text_projection(pooled)

    raise TypeError(
        f"無法從 CLIP text outputs 取出 Tensor features，實際型別是：{type(outputs)}"
    )

@torch.no_grad()
def encode_images_with_clip(image_paths, batch_size=16):
    """
    使用 CLIP 將圖片轉成 embedding。
    回傳 shape:
        [num_images, embedding_dim]
    """
    all_embeddings = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images with CLIP", disable=True):
        batch_paths = image_paths[i:i + batch_size]
        images = [load_image(p) for p in batch_paths]

        inputs = clip_processor(
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(device)

        outputs = clip_model.get_image_features(**inputs)
        features = extract_clip_image_features(outputs)

        # 正規化後，dot product 等同 cosine similarity
        features = features / features.norm(dim=-1, keepdim=True)

        all_embeddings.append(features.cpu())

    return torch.cat(all_embeddings, dim=0)

def minmax_normalize_clip_scores(raw_scores):
    """
    對通過物體硬性條件的 candidate 圖片做 min-max normalization。

    raw_scores:
        List[float]

    回傳：
        List[float]，範圍為 0~1

    如果所有 raw score 都一樣，代表 CLIP 無法區分這批圖片，
    此時全部給 1.0，避免除以 0。
    """
    if len(raw_scores) == 0:
        return []

    min_score = min(raw_scores)
    max_score = max(raw_scores)

    if abs(max_score - min_score) < 1e-8:
        return [1.0 for _ in raw_scores]

    return [
        (score - min_score) / (max_score - min_score)
        for score in raw_scores
    ]

# OWL-ViT：物件偵測
def clean_label_name(label_name: str):
    """
    清理新版 grounded object detection 可能回傳的文字 label。

    例如：
        "a photo of a person" -> "person"
    """
    label_name = str(label_name).strip()

    prefixes = [
        "a photo of a ",
        "a photo of an ",
        "a photo of ",
        "photo of a ",
        "photo of an ",
        "photo of ",
    ]

    lower_label = label_name.lower()

    for prefix in prefixes:
        if lower_label.startswith(prefix):
            return label_name[len(prefix):].strip()

    return label_name

def resolve_label_name(label_item):
    """
    兼容不同 transformers 版本的 label 格式。
    """
    if isinstance(label_item, str):
        return clean_label_name(label_item)

    raise TypeError(f"不支援的 label 格式：{type(label_item)}，內容：{label_item}")

def post_process_owlvit_outputs(outputs, target_sizes):
    """
    兼容不同 transformers 版本的 OWL-ViT 後處理 API。
    """
    text_labels = [[label for label in OBJECT_LABELS]]

    if hasattr(owl_processor, "post_process_grounded_object_detection"):
        try:
            return owl_processor.post_process_grounded_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=OBJECT_THRESHOLD,
                text_labels=text_labels,
            )[0]
        except TypeError:
            # 某些版本沒有 text_labels 參數
            return owl_processor.post_process_grounded_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=OBJECT_THRESHOLD,
            )[0]

    raise AttributeError(
        "目前安裝的 transformers 版本不支援"
    )

@torch.no_grad()
def detect_objects(image_path: Path):
    """
    使用 OWL-ViT 偵測 OBJECT_LABELS 中指定的物體。

    回傳格式：

    {
        "person": [
            {
                "score": 0.91,
                "box": [x1, y1, x2, y2],
                "center": [cx, cy]
            }
        ],
        "car": [
            {
                "score": 0.88,
                "box": [x1, y1, x2, y2],
                "center": [cx, cy]
            }
        ]
    }
    """
    image = load_image(image_path)

    text_prompts = [[f"a photo of a {label}" for label in OBJECT_LABELS]]

    inputs = owl_processor(
        text=text_prompts,
        images=image,
        return_tensors="pt",
    ).to(device)

    outputs = owl_model(**inputs)

    target_sizes = torch.tensor(
        [[image.height, image.width]],
        device=device,
    )

    results = post_process_owlvit_outputs(
        outputs=outputs,
        target_sizes=target_sizes,
    )

    detected = {}

    width = image.width
    height = image.height

    scores = results.get("scores", [])
    boxes = results.get("boxes", [])

    # 回傳text_labels
    if "text_labels" in results:
        labels = results["text_labels"]
    else:
        raise KeyError(
            "OWL-ViT 後處理結果中找不到 labels 或 text_labels。"
        )

    for score, label_item, box in zip(scores, labels, boxes):
        label_name = resolve_label_name(label_item)
        score_value = float(score.detach().cpu().item())

        if score_value < OBJECT_THRESHOLD:
            continue

        # 如果新版 API 回傳的 label 不在 OBJECT_LABELS 裡，跳過，避免後續權重邏輯混亂
        if label_name not in OBJECT_LABELS:
            continue

        x1, y1, x2, y2 = box.detach().cpu().tolist()

        # 正規化 bounding box 到 0~1
        x1_norm = max(0.0, min(1.0, x1 / width))
        y1_norm = max(0.0, min(1.0, y1 / height))
        x2_norm = max(0.0, min(1.0, x2 / width))
        y2_norm = max(0.0, min(1.0, y2 / height))

        cx = (x1_norm + x2_norm) / 2.0
        cy = (y1_norm + y2_norm) / 2.0

        object_info = {
            "score": score_value,
            "box": [x1_norm, y1_norm, x2_norm, y2_norm],
            "center": [cx, cy],
        }

        if label_name not in detected:
            detected[label_name] = []

        detected[label_name].append(object_info)

    # 每個 label 依照 score 由高到低排序，只保留前 MAX_BOXES_PER_LABEL 個
    for label_name in list(detected.keys()):
        detected[label_name] = sorted(
            detected[label_name],
            key=lambda obj: obj["score"],
            reverse=True,
        )[:MAX_BOXES_PER_LABEL]

    return detected

# Position Score：物體位置分數
def center_distance(obj_a, obj_b):
    """
    計算兩個 object_info 的中心點距離。
    """
    ax, ay = obj_a["center"]
    bx, by = obj_b["center"]

    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

def distance_to_position_score(distance):
    """
    將中心點距離轉成 0~1 的位置分數。

    距離越小，分數越接近 1。
    距離越大，分數越接近 0。
    """
    return math.exp(
        -((distance ** 2) / (2 * (POSITION_SIGMA ** 2)))
    )

def compute_label_position_score(query_objects, candidate_objects):
    """
    計算同一個 label 的位置相似度。

    做法：
    - 每一個 query object 找一個最近的 candidate object
    - candidate object 被配對後就不重複使用
    - 如果 query object 沒有 candidate 可以配對，分數為 0
    - 最後取平均
    """
    if len(query_objects) == 0:
        return 0.0

    if len(candidate_objects) == 0:
        return 0.0

    used_candidate_indices = set()
    matched_scores = []

    # query objects 依照信心分數由高到低處理
    sorted_query_objects = sorted(
        query_objects,
        key=lambda obj: obj["score"],
        reverse=True,
    )

    for query_obj in sorted_query_objects:
        best_idx = None
        best_distance = None

        for candidate_idx, candidate_obj in enumerate(candidate_objects):
            if candidate_idx in used_candidate_indices:
                continue

            distance = center_distance(query_obj, candidate_obj)

            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_idx = candidate_idx

        if best_idx is None:
            matched_scores.append(0.0)
            continue

        used_candidate_indices.add(best_idx)
        matched_scores.append(distance_to_position_score(best_distance))

    if len(matched_scores) == 0:
        return 0.0

    return sum(matched_scores) / len(matched_scores)

def compute_position_score(query_detected_objects, candidate_detected_objects, required_objects):
    """
    比較同一種物體在兩張圖片中的位置是否接近。
    """
    scores = []

    for label in required_objects:
        if label not in query_detected_objects:
            continue

        if label not in candidate_detected_objects:
            scores.append(0.0)
            continue

        query_objects = query_detected_objects[label]
        candidate_objects = candidate_detected_objects[label]

        label_score = compute_label_position_score(
            query_objects=query_objects,
            candidate_objects=candidate_objects,
        )

        scores.append(label_score)

    if len(scores) == 0:
        return 0.0

    return sum(scores) / len(scores)

# Scene Score：場景類型分數
@torch.no_grad()
def encode_scene_texts():
    """
    將 SCENE_LABELS 轉成 CLIP text embeddings。
    """
    scene_prompts = [
        f"a photo of a {scene_label}"
        for scene_label in SCENE_LABELS
    ]

    inputs = clip_processor(
        text=scene_prompts,
        return_tensors="pt",
        padding=True,
    ).to(device)

    outputs = clip_model.get_text_features(**inputs)
    text_features = extract_clip_text_features(outputs)

    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features.cpu()

def compute_scene_distribution(image_embedding, scene_text_embeddings):
    """
    用 CLIP 比較圖片和各個場景文字 prompt 的相似度。
    回傳每個場景的機率分布。
    """
    logit_scale = float(clip_model.logit_scale.exp().detach().cpu().item())

    logits = logit_scale * (image_embedding @ scene_text_embeddings.T)
    probs = torch.softmax(logits, dim=-1)

    return probs.squeeze(0)

def get_top_scene_label(scene_distribution):
    """
    取得最高分的場景類型。
    """
    idx = int(torch.argmax(scene_distribution).item())
    return SCENE_LABELS[idx], float(scene_distribution[idx].item())

def get_top_k_scene_labels(scene_distribution, k=SCENE_TOP_K):
    """
    取得 Top-K 場景類型。

    回傳格式：
    [
        ("city street", 0.42),
        ("street scene", 0.28),
        ("urban area", 0.15)
    ]
    """
    top_k = min(k, len(SCENE_LABELS))

    values, indices = torch.topk(scene_distribution, k=top_k)

    top_labels = []

    for value, idx in zip(values, indices):
        label = SCENE_LABELS[int(idx.item())]
        score = float(value.item())
        top_labels.append((label, score))

    return top_labels

def compute_scene_score(query_scene_distribution, candidate_scene_distribution):
    """
    場景分數：

    1. top1_match_score：
       query 第一名場景和 candidate 第一名場景是否相同。
       相同給 1，不同給 0。

    2. topk_overlap_score：
       query Top-K 場景和 candidate Top-K 場景的重疊比例。
       例如 K=3，重疊 2 個，分數就是 2/3。

    最後：
       scene_score = 0.6 * top1_match_score + 0.4 * topk_overlap_score
    """
    top1_weight, topk_weight = normalize_scene_weights()

    query_top1_label, _ = get_top_scene_label(query_scene_distribution)
    candidate_top1_label, _ = get_top_scene_label(candidate_scene_distribution)

    top1_match_score = 1.0 if query_top1_label == candidate_top1_label else 0.0

    query_top_k = get_top_k_scene_labels(
        query_scene_distribution,
        k=SCENE_TOP_K,
    )

    candidate_top_k = get_top_k_scene_labels(
        candidate_scene_distribution,
        k=SCENE_TOP_K,
    )

    query_top_k_labels = {label for label, _ in query_top_k}
    candidate_top_k_labels = {label for label, _ in candidate_top_k}

    intersection = query_top_k_labels & candidate_top_k_labels

    if SCENE_TOP_K <= 0:
        topk_overlap_score = 0.0
    else:
        topk_overlap_score = len(intersection) / min(
            SCENE_TOP_K,
            len(SCENE_LABELS),
        )

    scene_score = (
        top1_weight * top1_match_score
        + topk_weight * topk_overlap_score
    )

    return scene_score

# 主程式
def find_best_pair(
    query_image_path=DEFAULT_QUERY_IMAGE_PATH,
    candidate_dir=DEFAULT_CANDIDATE_DIR,
):
    load_retrieval_models()

    query_image_path = Path(query_image_path)
    candidate_dir = Path(candidate_dir)

    weights = normalize_weights(WEIGHTS)

    validate_query_image(query_image_path)

    candidate_images = get_image_paths(candidate_dir)

    if len(candidate_images) == 0:
        raise ValueError(f"資料夾中沒有可比對的圖片：{candidate_dir}")

    # Step 1: 偵測 query 物體
    query_detected_objects = detect_objects(query_image_path)

    # 位置分數仍然使用 query 偵測到的重要物件計算
    position_objects = set(query_detected_objects.keys()) & IMPORTANT_REQUIRED_LABELS

    # Step 2: 偵測 candidate 物體
    candidate_object_results = {}

    for image_path in tqdm(candidate_images, desc="Object detection", disable=True):
        detected = detect_objects(image_path)
        candidate_object_results[image_path] = detected

    # Step 3: CLIP 圖片 embedding
    query_embedding = encode_images_with_clip([query_image_path])
    candidate_embeddings = encode_images_with_clip(candidate_images)

    clip_raw_scores = query_embedding @ candidate_embeddings.T
    clip_raw_scores = clip_raw_scores.squeeze(0)

    # Step 4: Scene embeddings
    scene_text_embeddings = encode_scene_texts()

    query_scene_distribution = compute_scene_distribution(
        query_embedding,
        scene_text_embeddings,
    )

    # Step 5: 計算每張 candidate 的非 CLIP-normalized 分數
    candidate_rows = []

    for idx, image_path in enumerate(candidate_images):
        candidate_detected_objects = candidate_object_results[image_path]

        # 1. position_score
        position_score = compute_position_score(
            query_detected_objects,
            candidate_detected_objects,
            position_objects,
        )

        # 2. clip raw score
        clip_raw_score = float(clip_raw_scores[idx].item())

        # 3. scene_score
        candidate_embedding = candidate_embeddings[idx:idx + 1]

        candidate_scene_distribution = compute_scene_distribution(
            candidate_embedding,
            scene_text_embeddings,
        )

        scene_score = compute_scene_score(
            query_scene_distribution,
            candidate_scene_distribution,
        )

        candidate_rows.append({
            "image_path": image_path,
            "position_score": position_score,
            "clip_raw_score": clip_raw_score,
            "scene_score": scene_score,
        })

    # Step 6: 對 candidate 做 CLIP min-max normalization
    if len(candidate_rows) == 0:
        return query_image_path, None

    clip_raw_score_list = [
        row["clip_raw_score"]
        for row in candidate_rows
    ]

    clip_minmax_scores = minmax_normalize_clip_scores(clip_raw_score_list)

    results = []

    for row, clip_minmax_score in zip(candidate_rows, clip_minmax_scores):
        final_score = (
            weights["position"] * row["position_score"]
            + weights["clip"] * clip_minmax_score
            + weights["scene"] * row["scene_score"]
        )

        row["clip_score"] = clip_minmax_score
        row["final_score"] = final_score

        results.append(row)

    # Step 7: 排序，回傳最佳圖片
    ranked_results = sorted(
        results,
        key=lambda x: x["final_score"],
        reverse=True,
    )

    best_result = ranked_results[0]
    best_style_path = best_result["image_path"]

    return query_image_path, best_style_path

def main():
    content_path, style_path = find_best_pair(
        DEFAULT_QUERY_IMAGE_PATH,
        DEFAULT_CANDIDATE_DIR,
    )

    print(f"輸入圖片：{content_path}")
    print(f"挑選圖片：{style_path}")

if __name__ == "__main__":
    main()