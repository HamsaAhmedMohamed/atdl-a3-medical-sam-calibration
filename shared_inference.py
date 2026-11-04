import hashlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from experiment_protocol import DATASET_PROTOCOLS, PROTOCOL, CaseAssignment, prompt_clicks


PROJECT_DIR = Path(__file__).resolve().parent
UPSTREAM_DIR = PROJECT_DIR / "references" / "Medical-SAM-Adapter"


@dataclass(frozen=True)
class RefugeCase:
    case_id: str
    image_path: Path
    rater_paths: tuple[Path, ...]


@dataclass
class CaseData:
    case_id: str
    image: np.ndarray
    ground_truth: np.ndarray
    clicks: np.ndarray


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_test_cases(data_dir: str | Path) -> list[RefugeCase]:
    split_dirs = [
        path
        for path in Path(data_dir).rglob("*")
        if path.is_dir()
        and path.name.casefold() == "test-400"
        and "__MACOSX" not in path.parts
    ]
    if len(split_dirs) != 1:
        raise RuntimeError(
            f"Expected one Test-400 directory under {data_dir}, found {len(split_dirs)}."
        )
    cases = []
    for image_path in sorted(split_dirs[0].rglob("*.jpg")):
        case_id = image_path.stem
        case_dir = image_path.parent
        rater_paths = tuple(
            case_dir / f"{case_id}_seg_disc_{rater}.png" for rater in range(1, 8)
        )
        if all(path.is_file() for path in rater_paths):
            cases.append(RefugeCase(case_id, image_path, rater_paths))
    if not cases:
        raise RuntimeError(
            "No labeled Test-400 images with all seven optic-disc rater masks were found."
        )
    if len({case.case_id for case in cases}) != len(cases):
        raise RuntimeError("Case IDs must be unique within Test-400.")
    return cases


def load_case(case: RefugeCase) -> CaseData:
    with Image.open(case.image_path) as image_file:
        image = np.asarray(image_file.convert("RGB"), dtype=np.uint8)
    rater_masks = []
    for mask_path in case.rater_paths:
        with Image.open(mask_path) as mask_file:
            rater_masks.append(np.asarray(mask_file.convert("L"), dtype=np.uint8) > 0)
    if any(mask.shape != rater_masks[0].shape for mask in rater_masks):
        raise ValueError(f"Rater mask dimensions disagree for {case.case_id}.")
    ground_truth = np.stack(rater_masks).sum(axis=0) >= 4
    if image.shape[:2] != ground_truth.shape:
        raise ValueError(f"Image and rater-mask dimensions disagree for {case.case_id}.")
    clicks = prompt_clicks(ground_truth, count=max(PROTOCOL.click_counts))
    return CaseData(case.case_id, image, ground_truth, clicks)


def select_mask_and_score(mask_logits, predicted_ious, token_index: int = 0):
    if mask_logits.ndim != 4 or predicted_ious.ndim != 2:
        raise ValueError(
            "Expected masks [batch, tokens, height, width] and scores "
            "[batch, tokens]."
        )
    if mask_logits.shape[:2] != predicted_ious.shape:
        raise ValueError("Each mask token must have its matching predicted-IoU score.")
    if mask_logits.shape[0] != 1 or not 0 <= token_index < mask_logits.shape[1]:
        raise ValueError("Expected one image and a valid mask-token index.")
    return mask_logits[0, token_index], float(predicted_ious[0, token_index].item())


def remap_base_checkpoint_key(key: str) -> str:
    key = key.replace(".attn.rel_pos_h", ".attn.rel_h")
    key = key.replace(".attn.rel_pos_w", ".attn.rel_w")
    if key.startswith("mask_decoder.transformer.layers."):
        key = key.replace(".mlp.lin1.", ".mlp.layers.0.0.")
        key = key.replace(".mlp.lin2.", ".mlp.fc.")
    return key


def load_model(model_name: str, device, base_checkpoint: Path, adapter_checkpoint: Path):
    if str(UPSTREAM_DIR) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_DIR))
    from models.sam import sam_model_registry

    is_adapter = model_name == "med_sa_adapter"
    args = SimpleNamespace(
        image_size=PROTOCOL.image_size,
        multimask_output=1,
        mod="sam_adpt" if is_adapter else "sam",
        mid_dim=None,
        thd=False,
        chunk=None,
    )
    model = sam_model_registry["vit_b"](
        args, checkpoint=str(base_checkpoint)
    ).to(device)
    checkpoint = torch.load(
        base_checkpoint, map_location="cpu", weights_only=True
    )
    model_state = model.state_dict()
    base_state = {}
    for key, value in checkpoint.items():
        mapped_key = remap_base_checkpoint_key(key)
        if mapped_key not in model_state or model_state[mapped_key].shape != value.shape:
            raise RuntimeError(f"SAM checkpoint parameter does not match the model: {key}")
        if mapped_key in base_state:
            raise RuntimeError(f"Multiple SAM checkpoint parameters map to {mapped_key}.")
        base_state[mapped_key] = value
    if len(base_state) != len(checkpoint):
        raise RuntimeError("Not all SAM checkpoint parameters were mapped.")
    model.load_state_dict(base_state, strict=False)

    adapter_keys_loaded = 0
    if is_adapter:
        checkpoint = torch.load(
            adapter_checkpoint, map_location="cpu", weights_only=True
        )
        adapter_state = checkpoint["state_dict"]
        if adapter_state and all(key.startswith("module.") for key in adapter_state):
            adapter_state = {
                key.removeprefix("module."): value
                for key, value in adapter_state.items()
            }
        model_state = model.state_dict()
        compatible_state = {
            key: value
            for key, value in adapter_state.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        adapter_keys_loaded = sum(
            "Adapter" in key or "adapter" in key for key in compatible_state
        )
        if not adapter_keys_loaded:
            raise RuntimeError("No Med-SA adapter weights matched the constructed ViT-B model.")
        model.load_state_dict(compatible_state, strict=False)
    model.eval().requires_grad_(False)
    return model, len(base_state), adapter_keys_loaded


def _resized_input(image: np.ndarray, device):
    height, width = image.shape[:2]
    scale = PROTOCOL.image_size / max(height, width)
    resized_height = int(height * scale + 0.5)
    resized_width = int(width * scale + 0.5)
    resized = Image.fromarray(image).resize(
        (resized_width, resized_height), Image.Resampling.BILINEAR
    )
    image_tensor = torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1).float()
    return image_tensor.unsqueeze(0).to(device), (resized_height, resized_width)


def predict_case(model, case: CaseData, device) -> dict[int, dict[str, object]]:
    image_tensor, resized_size = _resized_input(case.image, device)
    original_size = case.image.shape[:2]
    image_embedding = model.image_encoder(model.preprocess(image_tensor))
    predictions = {}
    for click_count in PROTOCOL.click_counts:
        coordinates = torch.as_tensor(
            case.clicks[:click_count], dtype=torch.float32, device=device
        )
        coordinates[:, 0] *= resized_size[1] / original_size[1]
        coordinates[:, 1] *= resized_size[0] / original_size[0]
        coordinates = coordinates.unsqueeze(0)
        labels = torch.ones((1, click_count), dtype=torch.int, device=device)
        sparse, dense = model.prompt_encoder(
            points=(coordinates, labels), boxes=None, masks=None
        )
        mask_logits, predicted_ious = model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        mask_logits = model.postprocess_masks(
            mask_logits, input_size=resized_size, original_size=original_size
        )
        selected_logits, raw_score = select_mask_and_score(mask_logits, predicted_ious)
        predictions[click_count] = {
            "mask": (selected_logits > model.mask_threshold).detach().cpu().numpy(),
            "native_score_raw": raw_score,
        }
    return predictions


def run_model(
    model_name: str,
    cases: list[CaseData],
    device,
    base_checkpoint: Path,
    adapter_checkpoint: Path,
):
    torch.manual_seed(PROTOCOL.model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(PROTOCOL.model_seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    model, base_keys_loaded, adapter_keys_loaded = load_model(
        model_name, device, base_checkpoint, adapter_checkpoint
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    predictions = {}
    with torch.inference_mode():
        for case in cases:
            predictions[case.case_id] = predict_case(model, case, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    measurement = {
        "inference_seconds": elapsed,
        "images": len(cases),
        "seconds_per_image": elapsed / len(cases),
        "base_state_keys_loaded": base_keys_loaded,
        "adapter_state_keys_loaded": adapter_keys_loaded,
    }
    if device.type == "cuda":
        measurement["peak_cuda_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(device)
        )
        measurement["peak_cuda_reserved_bytes"] = int(
            torch.cuda.max_memory_reserved(device)
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions, measurement


def save_predictions(
    output_dir: Path,
    cases: list[CaseData],
    predictions: dict[str, object],
    assignments: dict[str, CaseAssignment],
) -> None:
    case_ids = {case.case_id for case in cases}
    if case_ids != set(assignments):
        raise ValueError("Every prediction case must have exactly one role assignment.")

    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        assignment = assignments[case.case_id]
        arrays = {
            "ground_truth_disc": case.ground_truth.astype(np.uint8),
            "clicks_xy": case.clicks.astype(np.int32),
            "dataset_id": np.asarray(assignment.dataset_id),
            "source_pool": np.asarray(assignment.source_pool),
            "analysis_role": np.asarray(assignment.analysis_role),
            "adapter_exposure_status": np.asarray(
                DATASET_PROTOCOLS[assignment.dataset_id].adapter_exposure_status
            ),
        }
        for model_name in PROTOCOL.models:
            for click_count in PROTOCOL.click_counts:
                prediction = predictions[model_name][case.case_id][click_count]
                arrays[f"{model_name}_{click_count}_click_mask"] = prediction[
                    "mask"
                ].astype(np.uint8)
                arrays[f"{model_name}_{click_count}_native_score_raw"] = np.asarray(
                    prediction["native_score_raw"], dtype=np.float32
                )
        np.savez_compressed(prediction_dir / f"{case.case_id}.npz", **arrays)
