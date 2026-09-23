from __future__ import annotations
PERCEPTION_BUILD_ID = "2026-09-22_PUSH_GRASP_ENGINE_V1"

import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import timm
from einops import rearrange

from dotenv import load_dotenv
from google import genai
from google.genai import types


ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"
OUTPUT_DIR = ROOT / "outputs"
CLOUD_DIR = OUTPUT_DIR / "primitive_clouds"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CLOUD_DIR.mkdir(parents=True, exist_ok=True)

MRFORMER_PATH = WEIGHTS_DIR / "final_mrformer_fast.pth"

IMAGE_SIZE = 224
NUM_CLASSES = 8
PRIMITIVE_NAMES = {
    1: "cuboid",
    2: "sphere",
    3: "hemisphere",
    4: "cylinder",
    5: "ring",
    6: "stick",
    7: "cone",
}

# Same normalization used by the earlier MR-Former notebook.
MR_DEPTH_MIN_M = 0.8
MR_DEPTH_MAX_M = 4.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# MR-FORMER -- SAME ARCHITECTURE AS THE TRAINED WEIGHTS
# =============================================================================

class MRCrossAttention(nn.Module):
    def __init__(self, dim: int = 384, heads: int = 6):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x = self.norm1(x + a)
        f = self.ffn(x)
        return self.norm2(x + f)


class ConvRefine(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MRFormerFast(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            in_chans=4,
            num_classes=0,
        )
        self.refine = MRCrossAttention(dim=384)
        self.up1 = nn.ConvTranspose2d(384, 256, 2, 2)
        self.ref1 = ConvRefine(256, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.ref2 = ConvRefine(128, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.ref3 = ConvRefine(64, 64)
        self.up4 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.ref4 = ConvRefine(32, 32)
        self.final = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        f = self.backbone.forward_features(x)
        if f.ndim != 3:
            raise RuntimeError(f"Unexpected ViT feature shape: {tuple(f.shape)}")
        if f.shape[1] == 197:
            f = f[:, 1:, :]
        f = self.refine(f)
        b, n, c = f.shape
        side = int(round(math.sqrt(n)))
        if side * side != n:
            raise RuntimeError(f"Unexpected ViT token count: {n}")
        x = rearrange(f, "b (h w) c -> b c h w", h=side, w=side)
        x = self.ref1(self.up1(x))
        x = self.ref2(self.up2(x))
        x = self.ref3(self.up3(x))
        x = self.ref4(self.up4(x))
        x = self.final(x)
        return F.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)


def _safe_load_weights(path: Path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def load_mrformer() -> MRFormerFast:
    if not MRFORMER_PATH.exists():
        raise FileNotFoundError(
            f"MR-Former weights not found:\n{MRFORMER_PATH}\n"
            "Copy/rename the trained file to weights/final_mrformer_fast.pth"
        )
    model = MRFormerFast(NUM_CLASSES)
    state = _safe_load_weights(MRFORMER_PATH)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    print(f"MR-Former loaded on {DEVICE}")
    return model


# =============================================================================
# PYBULLET SEGMENTATION / CAMERA GEOMETRY
# =============================================================================

def decode_pybullet_segmentation(segmentation: np.ndarray):
    seg = np.asarray(segmentation, dtype=np.int64)
    valid = seg >= 0
    object_ids = np.full(seg.shape, -1, dtype=np.int32)
    link_ids = np.full(seg.shape, -2, dtype=np.int32)
    packed = seg[valid]
    object_ids[valid] = (packed & ((1 << 24) - 1)).astype(np.int32)
    link_ids[valid] = ((packed >> 24) - 1).astype(np.int32)
    return object_ids, link_ids


def _normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


def camera_basis(eye, target, up):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = _normalize(up)
    forward = _normalize(target - eye)
    right = _normalize(np.cross(forward, up))
    true_up = _normalize(np.cross(right, forward))
    down = -true_up  # image-v direction
    return eye, right, down, forward


def camera_transforms(camera_eye, camera_target, camera_up):
    eye, right, down, forward = camera_basis(camera_eye, camera_target, camera_up)
    T_world_from_camera = np.eye(4, dtype=np.float64)
    T_world_from_camera[:3, 0] = right
    T_world_from_camera[:3, 1] = down
    T_world_from_camera[:3, 2] = forward
    T_world_from_camera[:3, 3] = eye
    T_camera_from_world = np.linalg.inv(T_world_from_camera)
    return T_world_from_camera.astype(np.float32), T_camera_from_world.astype(np.float32)


def world_point_to_camera(point_world, T_camera_from_world):
    p4 = np.ones(4, dtype=np.float64)
    p4[:3] = np.asarray(point_world, dtype=np.float64)[:3]
    out = np.asarray(T_camera_from_world, dtype=np.float64) @ p4
    return out[:3].astype(np.float32)


def pybullet_intrinsics(width: int, height: int, fov_y_deg: float) -> Dict[str, float]:
    fy = height / (2.0 * math.tan(math.radians(fov_y_deg) / 2.0))
    fx = fy
    return {"fx": fx, "fy": fy, "cx": width / 2.0, "cy": height / 2.0}


def _pixel_depth_to_world(u: float, v: float, z: float, intr, camera_eye, camera_target, camera_up):
    x = (float(u) - intr["cx"]) * float(z) / intr["fx"]
    y = (float(v) - intr["cy"]) * float(z) / intr["fy"]
    eye, right, down, forward = camera_basis(camera_eye, camera_target, camera_up)
    return (
        eye
        + x * right
        + y * down
        + float(z) * forward
    ).astype(np.float32)


# =============================================================================
# BODY-ISOLATED MR-FORMER INPUT
# =============================================================================

def depth_to_mrformer_channel(depth_m: np.ndarray) -> np.ndarray:
    d = np.asarray(depth_m, dtype=np.float32)
    d = np.where(np.isfinite(d), d, MR_DEPTH_MAX_M)
    d = np.clip(d, MR_DEPTH_MIN_M, MR_DEPTH_MAX_M)
    d = (d - MR_DEPTH_MIN_M) / (MR_DEPTH_MAX_M - MR_DEPTH_MIN_M)
    return np.round(d * 255.0).astype(np.uint8)


def _square_roi_from_mask(mask: np.ndarray, pad_fraction: float = 0.18):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    bw = x_max - x_min + 1
    bh = y_max - y_min + 1
    side = int(math.ceil(max(bw, bh) * (1.0 + 2.0 * pad_fraction)))
    side = max(side, 20)
    side = min(side, w, h)
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    x0 = min(max(x0, 0), w - side)
    y0 = min(max(y0, 0), h - side)
    return x0, y0, x0 + side, y0 + side


def _prepare_single_object_for_mrformer(rgb, depth_m, body_mask_full):
    """Prepare a body-isolated MR-Former input WITHOUT object-crop magnification.

    The previous version cropped every PyBullet body tightly and stretched that crop
    to 224x224.  That changes the apparent scale/aspect ratio seen by MR-Former and
    was the main reason that small top caps were repeatedly classified as
    ``hemisphere``.  This version keeps the original full-camera composition, masks
    everything except the selected PyBullet body, and then performs the same
    full-frame resize that the trained model expects.
    """
    body_mask_full = np.asarray(body_mask_full, dtype=bool)
    if not np.any(body_mask_full):
        return None

    h, w = body_mask_full.shape
    rgb_iso = np.zeros_like(rgb)
    rgb_iso[body_mask_full] = rgb[body_mask_full]

    depth_iso = np.full_like(depth_m, MR_DEPTH_MAX_M, dtype=np.float32)
    depth_iso[body_mask_full] = depth_m[body_mask_full]

    rgb224 = cv2.resize(rgb_iso, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    depth8 = depth_to_mrformer_channel(depth_iso)
    depth224_8 = cv2.resize(depth8, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    mask224 = cv2.resize(
        body_mask_full.astype(np.uint8),
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    rgb224[~mask224] = 0
    depth224_8[~mask224] = 255
    rgbd = np.concatenate([rgb224, depth224_8[..., None]], axis=2)
    return {
        "rgbd": rgbd,
        "rgb224": rgb224,
        "foreground224": mask224,
        # The prediction maps back to the COMPLETE camera image, not a cropped ROI.
        "roi_full": (0, 0, w, h),
    }

def _mask224_to_full(mask224: np.ndarray, roi_full, full_shape):
    h, w = full_shape
    x0, y0, x1, y1 = roi_full
    resized = cv2.resize(mask224.astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST).astype(bool)
    full = np.zeros((h, w), dtype=bool)
    full[y0:y1, x0:x1] = resized
    return full


def _clean_components_for_class(binary: np.ndarray, min_pixels: int):
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
    comps = []
    for c in range(1, n):
        area = int(stats[c, cv2.CC_STAT_AREA])
        if area >= min_pixels:
            comps.append((area, labels == c))
    comps.sort(key=lambda x: x[0], reverse=True)
    return comps


def _extract_instances_from_logits(logits: torch.Tensor, foreground224: np.ndarray, min_pixels: int = 24):
    """Confidence/area-aware MR-Former instance extraction.

    PyBullet supplies ONLY the per-body foreground support.  Primitive class labels
    still come from MR-Former.  Two safeguards are used here:

    1) reject tiny/low-confidence class islands;
    2) suppress the systematic *small hemisphere cap* false positive when a much
       stronger non-hemisphere primitive explains the same isolated rigid object.

    A genuine isolated hemisphere is still retained when it is the dominant
    MR-Former prediction.
    """
    fg = np.asarray(foreground224, dtype=bool)
    fg_area = int(fg.sum())
    if fg_area <= 0:
        return []

    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    pred = np.argmax(probs, axis=0).astype(np.uint8)
    pred[~fg] = 0

    dynamic_min = max(int(min_pixels), int(round(0.025 * fg_area)))
    candidates = []
    for cls in range(1, NUM_CLASSES):
        comps = _clean_components_for_class(pred == cls, dynamic_min)
        for area, mask in comps[:2]:
            area = int(area)
            frac = float(area) / float(max(fg_area, 1))
            mean_conf = float(probs[cls][mask].mean()) if np.any(mask) else 0.0
            peak_conf = float(probs[cls][mask].max()) if np.any(mask) else 0.0
            # Small argmax islands are normally simulation-domain artifacts.
            if frac < 0.035:
                continue
            if mean_conf < 0.34 and peak_conf < 0.60:
                continue
            candidates.append({
                "class_id": int(cls),
                "primitive_type": PRIMITIVE_NAMES[cls],
                "mask": mask,
                "pixel_count": area,
                "foreground_fraction": frac,
                "mean_confidence": mean_conf,
                "peak_confidence": peak_conf,
                "quality": frac * (0.65 + 0.35 * mean_conf),
            })

    if candidates:
        # Strongest explanation first.
        candidates.sort(key=lambda d: d["quality"], reverse=True)

        non_hemi = [c for c in candidates if c["primitive_type"] != "hemisphere"]
        if non_hemi:
            strongest_non_hemi = max(c["quality"] for c in non_hemi)
            filtered = []
            for c in candidates:
                if c["primitive_type"] == "hemisphere":
                    # Typical false positive = a small rounded/top-cap region on
                    # bottles, cylinders, rings and compound objects.
                    if (
                        c["foreground_fraction"] < 0.45
                        and c["quality"] < 1.15 * strongest_non_hemi
                    ):
                        continue
                filtered.append(c)
            candidates = filtered or non_hemi

        # Avoid returning several tiny pieces of one body as separate primitives.
        # Keep at most three strong primitive instances, unless only one exists.
        candidates.sort(key=lambda d: d["quality"], reverse=True)
        return candidates[:3]

    # Conservative fallback: use the dominant non-background class probability
    # over the body support.  No PyBullet primitive label is used.
    means = [float(probs[cls][fg].mean()) for cls in range(1, NUM_CLASSES)]
    dominant = 1 + int(np.argmax(means))
    return [{
        "class_id": dominant,
        "primitive_type": PRIMITIVE_NAMES[dominant],
        "mask": fg.copy(),
        "pixel_count": fg_area,
        "foreground_fraction": 1.0,
        "mean_confidence": float(means[dominant - 1]),
        "peak_confidence": float(probs[dominant][fg].max()),
        "quality": 1.0,
    }]


# =============================================================================
# POINT CLOUD + PRIMITIVE GEOMETRY
# =============================================================================

def _full_mask_to_world_cloud(mask_full, rgb_full, depth_full_m, intr_full, camera_eye, camera_target, camera_up):
    valid = (
        np.asarray(mask_full, dtype=bool)
        & np.isfinite(depth_full_m)
        & (depth_full_m > 0.0)
        & (depth_full_m < MR_DEPTH_MAX_M - 1e-5)
    )
    v, u = np.where(valid)
    if len(u) == 0:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)
    z = depth_full_m[v, u]
    x = (u - intr_full["cx"]) * z / intr_full["fx"]
    y = (v - intr_full["cy"]) * z / intr_full["fy"]
    points_cam = np.stack([x, y, z], axis=1)
    eye, right, down, forward = camera_basis(camera_eye, camera_target, camera_up)
    points_world = (
        eye[None, :]
        + points_cam[:, 0:1] * right[None, :]
        + points_cam[:, 1:2] * down[None, :]
        + points_cam[:, 2:3] * forward[None, :]
    )
    return points_world.astype(np.float32), rgb_full[v, u].astype(np.uint8)


def estimate_primitive_geometry(points_world: np.ndarray, primitive_type: str):
    if len(points_world) < 12:
        raise ValueError("Too few 3-D points for primitive geometry estimation.")
    pts = np.asarray(points_world, dtype=np.float64)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]
    projected = centered @ evecs
    extents = projected.max(axis=0) - projected.min(axis=0)
    extents_sorted = np.sort(extents)[::-1]

    L = W = H = Rout = Rin = 0.0
    axis = np.zeros(3, dtype=np.float64)
    yaw = 0.0

    if primitive_type == "cuboid":
        L, W, H = [float(x) for x in extents_sorted]
        major = evecs[:, 0]
        yaw = math.atan2(major[1], major[0])
    elif primitive_type == "sphere":
        Rout = float(np.median(np.linalg.norm(centered, axis=1)))
        if not np.isfinite(Rout) or Rout <= 1e-4:
            Rout = float(extents_sorted[0] / 2.0)
    elif primitive_type == "hemisphere":
        Rout = float(max(extents_sorted[0], extents_sorted[1]) / 2.0)
        axis = _normalize(evecs[:, 2])
        if axis[2] < 0:
            axis = -axis
        if np.linalg.norm(axis[:2]) > 1e-5:
            yaw = math.atan2(axis[1], axis[0])
    elif primitive_type == "cylinder":
        candidates = [evecs[:, i] for i in range(3)]
        vertical_index = int(np.argmax([abs(v[2]) for v in candidates]))
        if abs(candidates[vertical_index][2]) > 0.70:
            axis = _normalize(candidates[vertical_index])
            remaining = [i for i in range(3) if i != vertical_index]
            H = float(extents[vertical_index])
            Rout = float(np.mean([extents[i] for i in remaining]) / 2.0)
        else:
            axis = _normalize(evecs[:, 0])
            H = float(extents_sorted[0])
            Rout = float((extents_sorted[1] + extents_sorted[2]) / 4.0)
        if axis[2] < 0:
            axis = -axis
        if np.linalg.norm(axis[:2]) > 1e-5:
            yaw = math.atan2(axis[1], axis[0])
    elif primitive_type == "ring":
        axis = _normalize(evecs[:, 2])
        if axis[2] < 0:
            axis = -axis
        H = float(extents_sorted[2])
        Rout = float(max(extents_sorted[0], extents_sorted[1]) / 2.0)
        Rin = float(max(0.35 * Rout, 0.008))
    elif primitive_type == "stick":
        axis = _normalize(evecs[:, 0])
        L = float(extents_sorted[0])
        Rout = float((extents_sorted[1] + extents_sorted[2]) / 4.0)
        yaw = math.atan2(axis[1], axis[0]) if np.linalg.norm(axis[:2]) > 1e-5 else 0.0
    elif primitive_type == "cone":
        axis = _normalize(evecs[:, 0])
        if axis[2] < 0:
            axis = -axis
        H = float(extents_sorted[0])
        Rout = float(max(extents_sorted[1], extents_sorted[2]) / 2.0)
        yaw = math.atan2(axis[1], axis[0]) if np.linalg.norm(axis[:2]) > 1e-5 else 0.0
    else:
        raise ValueError(f"Unsupported primitive type: {primitive_type}")

    geometry_vector = np.array([L, W, H, Rout, Rin, axis[0], axis[1], axis[2]], dtype=np.float32)
    state_vector = np.array([centroid[0], centroid[1], yaw, 0.0, 0.0, 0.0], dtype=np.float32)
    planar_radius = float(max(L, W, 2.0 * Rout, 0.02) / 2.0)
    return {
        "centroid_world": centroid.astype(np.float32),
        "axis_world": axis.astype(np.float32),
        "yaw": float(yaw),
        "geometry_vector": geometry_vector,
        "state_vector": state_vector,
        "pca_extents": extents_sorted.astype(np.float32),
        "planar_radius": planar_radius,
    }


# =============================================================================
# GEMINI INPUT IMAGES
# =============================================================================

def _clean_rgb_objects_only(rgb: np.ndarray, segmentation: np.ndarray, object_body_ids: List[int]):
    object_ids, _ = decode_pybullet_segmentation(segmentation)
    fg = np.isin(object_ids, np.asarray(object_body_ids, dtype=np.int32))
    out = np.zeros_like(rgb)
    out[fg] = rgb[fg]
    return out, fg


def _make_full_primitive_mask_image(instances: List[dict], full_shape):
    h, w = full_shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    palette = [
        (235, 70, 70), (70, 220, 70), (70, 100, 240), (235, 220, 70),
        (225, 70, 220), (70, 220, 220), (245, 150, 60), (175, 95, 235),
        (100, 200, 120), (220, 120, 150), (120, 180, 240), (240, 180, 100),
    ]
    for i, inst in enumerate(instances):
        color = np.asarray(palette[i % len(palette)], dtype=np.uint8)
        mask = np.asarray(inst["mask_full"], dtype=bool)
        canvas[mask] = color
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        cx, cy = int(round(xs.mean())), int(round(ys.mean()))
        label = f"P{inst['primitive_id']} {inst['primitive_type']}"
        cv2.putText(canvas, label, (max(0, cx - 35), max(15, cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    path = OUTPUT_DIR / "mrformer_primitive_masks_for_gemini.png"
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return canvas, path


# =============================================================================
# GEMINI GROUPING + COM/HOLLOWNESS
# =============================================================================

def _gemini_client():
    load_dotenv(ROOT / ".env")
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Create .env beside main.py and add:\n"
            "GEMINI_API_KEY=YOUR_KEY_HERE"
        )
    return genai.Client(api_key=key)


def _extract_json_object(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        first = text.find("{")
        last = text.rfind("}")
        if first >= 0 and last > first:
            return json.loads(text[first:last + 1])
        raise


def _fallback_property_for_record(record: dict, body_metadata: Dict[int, dict]):
    meta = body_metadata.get(int(record["body_id"]), {})
    name = str(meta.get("semantic_name", "")).lower()
    ptype = record["primitive_type"]
    if "tape" in name:
        return "hollow", True
    if "cup" in name and ptype in {"cylinder", "ring"}:
        return "hollow", True
    return "solid", False


def _simulation_grouping_fallback(primitive_records: List[dict], body_metadata: Optional[Dict[int, dict]] = None):
    body_metadata = body_metadata or {}
    by_body: Dict[int, List[dict]] = {}
    for r in primitive_records:
        by_body.setdefault(int(r["body_id"]), []).append(r)
    objects = []
    for object_number, (bid, recs) in enumerate(sorted(by_body.items()), start=1):
        pids = sorted(int(r["primitive_id"]) for r in recs)
        props = []
        for r in recs:
            occ, accessible = _fallback_property_for_record(r, body_metadata)
            props.append({
                "primitive_id": int(r["primitive_id"]),
                "occupancy": occ,
                "inner_accessible": bool(accessible),
                "reason": "simulation metadata fallback",
            })
        # Image COM fallback: mean primitive pixel centers, normalized later.
        centers = np.array([r["image_center_full"] for r in recs], dtype=np.float64)
        mean_uv = centers.mean(axis=0) if len(centers) else np.array([0.5, 0.5])
        meta = body_metadata.get(bid, {})
        objects.append({
            "object_number": object_number,
            "object_label": meta.get("semantic_name", f"object_{object_number}"),
            "primitive_ids": pids,
            "com_uv_norm": [float(mean_uv[0]), float(mean_uv[1])],  # temporarily pixel, normalized below
            "primitive_properties": props,
            "reason": "SIMULATION FALLBACK: exact rigid-body grouping used because Gemini was unavailable.",
        })
    return {"objects": objects, "source": "simulation_fallback"}


def _normalize_fallback_com_uv(graph: dict, width: int, height: int):
    if graph.get("source") != "simulation_fallback":
        return graph
    for obj in graph.get("objects", []):
        uv = obj.get("com_uv_norm", [width / 2.0, height / 2.0])
        # fallback stored pixel centers above
        obj["com_uv_norm"] = [float(np.clip(uv[0] / max(width - 1, 1), 0.0, 1.0)), float(np.clip(uv[1] / max(height - 1, 1), 0.0, 1.0))]
    return graph


def gemini_group_primitives(
    rgb_objects_only: np.ndarray,
    primitive_mask_image: np.ndarray,
    primitive_records: List[dict],
    body_metadata: Optional[Dict[int, dict]] = None,
    max_retries: int = 2,
):
    valid_ids = sorted(int(r["primitive_id"]) for r in primitive_records)
    h, w = rgb_objects_only.shape[:2]
    if not valid_ids:
        return {"objects": [], "source": "empty"}

    prompt = f"""
You are the visual object-part reasoning module of a robotic pushing system.

IMAGE 1: clean RGB containing only the physical scene objects.
IMAGE 2: MR-Former primitive masks labeled P1, P2, ... with primitive class names.

Use ONLY these two images. Do not use coordinates, point clouds, simulator IDs, forces, or robot state.
Valid primitive IDs are: {valid_ids}

Tasks:
1) Group primitive masks that belong to the SAME rigid physical object.
2) Give each object a short visual label such as cup, bottle, sphere, tape, cylinder, cone-like object, banana, or food carton.
3) Estimate the projected CENTER OF MASS of the ENTIRE object as [u_norm,v_norm], each in [0,1], where (0,0) is image top-left and (1,1) bottom-right. This is a visual estimate, not an exact physics measurement.
4) For every primitive, classify its material occupancy as "hollow", "solid", or "unknown" and say whether its interior is visually accessible to a gripper/pusher.
5) Every primitive ID must appear exactly once.

Return ONLY JSON in exactly this structure:
{{
  "objects": [
    {{
      "object_number": 1,
      "object_label": "cup",
      "primitive_ids": [1,2],
      "com_uv_norm": [0.42,0.55],
      "primitive_properties": [
        {{"primitive_id":1,"occupancy":"hollow","inner_accessible":true,"reason":"short reason"}}
      ],
      "reason": "short grouping reason"
    }}
  ]
}}
"""

    try:
        client = _gemini_client()
    except Exception as exc:
        print(f"WARNING: Gemini client unavailable: {type(exc).__name__}: {exc}")
        return _normalize_fallback_com_uv(_simulation_grouping_fallback(primitive_records, body_metadata), w, h)

    preferred = os.getenv("GEMINI_MODEL", "gemini-robotics-er-2-preview").strip()
    fallback_model = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash").strip()
    model_names = []
    for name in (preferred, fallback_model):
        if name and name not in model_names:
            model_names.append(name)

    last_exc = None
    for model_name in model_names:
        for attempt in range(1, max_retries + 1):
            try:
                print(f"Gemini object grouping/COM: model={model_name}, attempt={attempt}/{max_retries} ...")
                response = client.models.generate_content(
                    model=model_name,
                    contents=[prompt, Image.fromarray(rgb_objects_only), Image.fromarray(primitive_mask_image)],
                    config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json"),
                )
                data = _extract_json_object(getattr(response, "text", ""))
                groups = data.get("objects", [])
                returned = [int(pid) for g in groups for pid in g.get("primitive_ids", [])]
                if sorted(returned) != valid_ids or len(returned) != len(set(returned)):
                    raise ValueError(f"Gemini primitive IDs invalid. expected={valid_ids}, returned={returned}")

                normalized = []
                for g in groups:
                    pids = sorted(int(x) for x in g.get("primitive_ids", []))
                    prop_map = {}
                    for pp in g.get("primitive_properties", []):
                        pid = int(pp.get("primitive_id", -1))
                        if pid in pids:
                            occ = str(pp.get("occupancy", "unknown")).lower()
                            if occ not in {"hollow", "solid", "unknown"}:
                                occ = "unknown"
                            prop_map[pid] = {
                                "primitive_id": pid,
                                "occupancy": occ,
                                "inner_accessible": bool(pp.get("inner_accessible", False)),
                                "reason": str(pp.get("reason", "")),
                            }
                    for pid in pids:
                        prop_map.setdefault(pid, {"primitive_id": pid, "occupancy": "unknown", "inner_accessible": False, "reason": "not specified"})
                    uv = g.get("com_uv_norm", [0.5, 0.5])
                    uv = [float(np.clip(float(uv[0]), 0.0, 1.0)), float(np.clip(float(uv[1]), 0.0, 1.0))]
                    normalized.append({
                        "object_label": str(g.get("object_label", "object")),
                        "primitive_ids": pids,
                        "com_uv_norm": uv,
                        "primitive_properties": [prop_map[pid] for pid in pids],
                        "reason": str(g.get("reason", "")),
                    })
                normalized.sort(key=lambda g: min(g["primitive_ids"]) if g["primitive_ids"] else 10**9)
                for i, g in enumerate(normalized, start=1):
                    g["object_number"] = i
                print(f"Gemini grouping succeeded with {model_name}.")
                return {"objects": normalized, "source": "gemini", "model": model_name}
            except Exception as exc:
                last_exc = exc
                msg = str(exc)
                print(f"WARNING: Gemini {model_name} attempt {attempt} failed: {type(exc).__name__}: {msg[:350]}")
                if "403" in msg or "Forbidden" in msg:
                    break
                if attempt < max_retries:
                    time.sleep(float(attempt))

    print("WARNING: Gemini unavailable. Using clearly labeled simulation fallback so the pipeline remains runnable.")
    if last_exc is not None:
        print(f"Last Gemini error: {type(last_exc).__name__}: {str(last_exc)[:350]}")
    return _normalize_fallback_com_uv(_simulation_grouping_fallback(primitive_records, body_metadata), w, h)


def _lift_group_coms(graph, primitive_records, mask_by_pid, depth_m, intr, camera_eye, camera_target, camera_up, T_cw):
    h, w = depth_m.shape
    records_by_pid = {int(r["primitive_id"]): r for r in primitive_records}
    for obj in graph.get("objects", []):
        pids = [int(x) for x in obj.get("primitive_ids", [])]
        union = np.zeros((h, w), dtype=bool)
        for pid in pids:
            if pid in mask_by_pid:
                union |= mask_by_pid[pid]
        valid_depth = depth_m[union & np.isfinite(depth_m) & (depth_m > 0) & (depth_m < MR_DEPTH_MAX_M - 1e-5)]
        if valid_depth.size > 0:
            z = float(np.median(valid_depth))
        else:
            centers = [records_by_pid[pid]["center_world"] for pid in pids if pid in records_by_pid]
            if centers:
                c = np.mean(np.asarray(centers, dtype=np.float64), axis=0).astype(np.float32)
                obj["estimated_com_world"] = c.tolist()
                obj["estimated_com_camera"] = world_point_to_camera(c, T_cw).tolist()
                continue
            z = 1.0
        uvn = obj.get("com_uv_norm", [0.5, 0.5])
        u = float(np.clip(uvn[0], 0.0, 1.0)) * (w - 1)
        v = float(np.clip(uvn[1], 0.0, 1.0)) * (h - 1)
        world = _pixel_depth_to_world(u, v, z, intr, camera_eye, camera_target, camera_up)
        obj["estimated_com_world"] = world.tolist()
        obj["estimated_com_camera"] = world_point_to_camera(world, T_cw).tolist()
    return graph


# =============================================================================
# GEMINI SECOND PASS: SOFT PUSH-PRIMITIVE ADVICE ONLY
# =============================================================================

def gemini_push_advice(
    rgb_objects_only: np.ndarray,
    primitive_mask_image: np.ndarray,
    selected_object: dict,
    desired_delta_xytheta: List[float],
    max_retries: int = 1,
):
    pids = [int(x) for x in selected_object.get("primitive_ids", [])]
    prop_text = selected_object.get("primitive_properties", [])
    dx, dy, dtheta = [float(x) for x in desired_delta_xytheta]
    prompt = f"""
You are a high-level embodied reasoning assistant for ONE already-selected rigid object.
You see ONLY the clean RGB image and the MR-Former primitive-mask image.
Selected object number: {selected_object.get('object_number')}
Selected object visual label: {selected_object.get('object_label')}
Its primitive IDs: {pids}
Previously inferred primitive hollow/solid properties: {json.dumps(prop_text)}
Desired planar motion: dx={dx:+.3f} m, dy={dy:+.3f} m, dtheta={math.degrees(dtheta):+.1f} deg.

Give only a SOFT semantic prior. Do NOT output exact contact coordinates, distances, robot joints, or forces.
The mathematical COM/torque model and recurrent RMPPI will make the final action choice.
For hollow ring/tape-like parts, you may state whether inner-rim or outer-rim contact is semantically reasonable, but only if the interior is accessible.

Return ONLY JSON:
{{
  "preferred_translation_primitive_ids": [1],
  "preferred_rotation_primitive_ids": [2],
  "avoid_primitive_ids": [],
  "hollow_contact_preference": {{"P2":"inner|outer|either|not_applicable"}},
  "reason": "short explanation"
}}
"""
    try:
        client = _gemini_client()
    except Exception as exc:
        return {"source": "math_only_fallback", "preferred_translation_primitive_ids": [], "preferred_rotation_primitive_ids": [], "avoid_primitive_ids": [], "hollow_contact_preference": {}, "reason": f"Gemini unavailable: {type(exc).__name__}"}

    model_names = []
    for name in (os.getenv("GEMINI_MODEL", "gemini-robotics-er-2-preview").strip(), os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash").strip()):
        if name and name not in model_names:
            model_names.append(name)

    for model_name in model_names:
        for attempt in range(1, max_retries + 1):
            try:
                print(f"Gemini push-prior assistance: model={model_name} ...")
                response = client.models.generate_content(
                    model=model_name,
                    contents=[prompt, Image.fromarray(rgb_objects_only), Image.fromarray(primitive_mask_image)],
                    config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json"),
                )
                data = _extract_json_object(getattr(response, "text", ""))
                for key in ("preferred_translation_primitive_ids", "preferred_rotation_primitive_ids", "avoid_primitive_ids"):
                    data[key] = [int(x) for x in data.get(key, []) if int(x) in pids]
                data["hollow_contact_preference"] = dict(data.get("hollow_contact_preference", {}))
                data["reason"] = str(data.get("reason", ""))
                data["source"] = "gemini"
                data["model"] = model_name
                return data
            except Exception as exc:
                msg = str(exc)
                print(f"WARNING: Gemini push advice failed: {type(exc).__name__}: {msg[:280]}")
                if "403" in msg or "Forbidden" in msg:
                    break
                if attempt < max_retries:
                    time.sleep(1.0)

    return {"source": "math_only_fallback", "preferred_translation_primitive_ids": [], "preferred_rotation_primitive_ids": [], "avoid_primitive_ids": [], "hollow_contact_preference": {}, "reason": "Gemini API unavailable; mathematical COM/torque + RMPPI used."}


# =============================================================================
# OBJECT CATALOG
# =============================================================================

def build_object_catalog(perception_result: dict, body_pose_lookup: Dict[int, dict]):
    records_by_pid = {int(r["primitive_id"]): r for r in perception_result["primitive_records"]}
    catalog = []
    for obj in perception_result["object_primitive_graph"].get("objects", []):
        pids = [int(x) for x in obj.get("primitive_ids", [])]
        recs = [records_by_pid[pid] for pid in pids if pid in records_by_pid]
        if not recs:
            continue
        votes: Dict[int, int] = {}
        for r in recs:
            bid = int(r["body_id"])
            votes[bid] = votes.get(bid, 0) + int(r["pixel_count"])
        body_id = max(votes, key=votes.get)
        pose = body_pose_lookup.get(body_id, {})
        catalog.append({
            "object_number": int(obj["object_number"]),
            "object_label": str(obj.get("object_label", "object")),
            "primitive_ids": pids,
            "primitive_types": [r["primitive_type"] for r in recs],
            "primitive_properties": obj.get("primitive_properties", []),
            "group_reason": str(obj.get("reason", "")),
            "group_source": perception_result["object_primitive_graph"].get("source", "unknown"),
            "body_id": int(body_id),
            "body_name": str(pose.get("body_name", f"body_{body_id}")),
            "initial_world_position": list(pose.get("position_world", [np.nan, np.nan, np.nan])),
            "initial_world_yaw_rad": float(pose.get("yaw_world", np.nan)),
            "initial_world_yaw_deg": float(math.degrees(float(pose.get("yaw_world", 0.0)))),
            "true_com_world": list(pose.get("true_com_world", [np.nan, np.nan, np.nan])),
            "gemini_com_world": list(obj.get("estimated_com_world", [np.nan, np.nan, np.nan])),
            "gemini_com_camera": list(obj.get("estimated_com_camera", [np.nan, np.nan, np.nan])),
            "com_uv_norm": list(obj.get("com_uv_norm", [0.5, 0.5])),
        })
    catalog.sort(key=lambda x: x["object_number"])
    return catalog


def apply_object_semantics_to_records(records: List[dict], selected_object: dict):
    """Attach initial Gemini hollow/solid semantics to current MR-Former records.

    Primitive IDs can change after re-perception, so matching is primarily by
    primitive type. When several same-type parts exist, the same conservative
    occupancy label is applied to each.
    """
    initial_props = selected_object.get("primitive_properties", [])
    initial_types = selected_object.get("primitive_types", [])
    initial_pids = selected_object.get("primitive_ids", [])
    by_type: Dict[str, List[dict]] = {}
    for pid, ptype in zip(initial_pids, initial_types):
        prop = next((x for x in initial_props if int(x.get("primitive_id", -999)) == int(pid)), None)
        if prop is not None:
            by_type.setdefault(str(ptype), []).append(prop)

    out = []
    for r in records:
        rr = dict(r)
        props = by_type.get(rr["primitive_type"], [])
        if props:
            # Hollow wins if any same-type initial component was hollow.
            hollow = any(str(p.get("occupancy", "unknown")).lower() == "hollow" for p in props)
            rr["occupancy"] = "hollow" if hollow else str(props[0].get("occupancy", "unknown"))
            rr["inner_accessible"] = any(bool(p.get("inner_accessible", False)) for p in props)
        else:
            rr["occupancy"] = "unknown"
            rr["inner_accessible"] = False
        out.append(rr)
    return out




def _looks_like_planar_cap(points_world: np.ndarray) -> bool:
    """Return True for a nearly planar point patch (typical false hemisphere cap)."""
    pts = np.asarray(points_world, dtype=np.float64)
    if len(pts) < 20:
        return True
    centered = pts - pts.mean(axis=0)
    try:
        evals = np.sort(np.maximum(np.linalg.eigvalsh(np.cov(centered.T)), 0.0))
    except Exception:
        return False
    if evals[-1] <= 1e-12:
        return True
    # A cap/top-face has almost no variation normal to its plane.
    return float(evals[0] / evals[-1]) < 0.018

# =============================================================================
# ONLINE ENTRY POINT
# =============================================================================

def run_perception(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    segmentation_gt: np.ndarray,
    object_body_ids: List[int],
    camera_eye,
    camera_target,
    camera_up,
    fov_y_deg: float,
    mrformer: Optional[MRFormerFast] = None,
    body_id_to_name: Optional[Dict[int, str]] = None,
    body_metadata: Optional[Dict[int, dict]] = None,
    use_gemini: bool = True,
    allow_empty: bool = False,
):
    """Simulation-only perception with body-isolated MR-Former inputs.

    PyBullet segmentation is used ONLY as a foreground support mask for each
    rigid object. MR-Former supplies every primitive class/mask. Gemini sees only
    the clean RGB and final MR-Former primitive-mask image.
    """
    if mrformer is None:
        mrformer = load_mrformer()

    rgb = np.asarray(rgb, dtype=np.uint8)
    depth_m = np.asarray(depth_m, dtype=np.float32)
    h, w = depth_m.shape
    object_id_map, _ = decode_pybullet_segmentation(segmentation_gt)
    intr = pybullet_intrinsics(w, h, fov_y_deg)
    T_wc, T_cw = camera_transforms(camera_eye, camera_target, camera_up)

    clean_rgb, _ = _clean_rgb_objects_only(rgb, segmentation_gt, object_body_ids)
    cv2.imwrite(str(OUTPUT_DIR / "objects_only_rgb_for_gemini.png"), cv2.cvtColor(clean_rgb, cv2.COLOR_RGB2BGR))

    primitive_records: List[dict] = []
    full_instances: List[dict] = []
    mask_by_pid: Dict[int, np.ndarray] = {}
    next_pid = 1

    for body_id in object_body_ids:
        body_mask = object_id_map == int(body_id)
        if int(body_mask.sum()) < 20:
            continue
        prepared = _prepare_single_object_for_mrformer(rgb, depth_m, body_mask)
        if prepared is None:
            continue

        tensor = T.ToTensor()(prepared["rgbd"]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = mrformer(tensor)
        local_instances = _extract_instances_from_logits(logits, prepared["foreground224"], min_pixels=20)

        for inst in local_instances:
            mask_full = _mask224_to_full(inst["mask"], prepared["roi_full"], (h, w))
            mask_full &= body_mask
            if int(mask_full.sum()) < 18:
                continue
            points_world, colors = _full_mask_to_world_cloud(mask_full, rgb, depth_m, intr, camera_eye, camera_target, camera_up)
            if len(points_world) < 12:
                continue

            # Second-stage sanity check for the systematic hemisphere-cap artifact.
            # A genuine hemisphere is a curved 3-D patch.  A bottle/cylinder top
            # face mislabeled as hemisphere is nearly planar and is removed here.
            if (
                inst["primitive_type"] == "hemisphere"
                and float(inst.get("foreground_fraction", 1.0)) < 0.60
                and _looks_like_planar_cap(points_world)
            ):
                continue

            try:
                est = estimate_primitive_geometry(points_world, inst["primitive_type"])
            except Exception as exc:
                print(f"Skipping body {body_id} {inst['primitive_type']}: {type(exc).__name__}: {exc}")
                continue

            ys, xs = np.where(mask_full)
            image_center = [float(xs.mean()), float(ys.mean())]
            full_instances.append({
                "primitive_id": int(next_pid),
                "primitive_type": inst["primitive_type"],
                "mask_full": mask_full,
            })
            mask_by_pid[int(next_pid)] = mask_full
            cloud_path = CLOUD_DIR / f"P{next_pid}_{inst['primitive_type']}.npz"
            np.savez_compressed(cloud_path, points_world=points_world, colors=colors)

            primitive_records.append({
                "primitive_id": int(next_pid),
                "class_id": int(inst["class_id"]),
                "primitive_type": inst["primitive_type"],
                "pixel_count": int(mask_full.sum()),
                "foreground_fraction": float(inst.get("foreground_fraction", 0.0)),
                "mean_confidence": float(inst.get("mean_confidence", 0.0)),
                "peak_confidence": float(inst.get("peak_confidence", 0.0)),
                "bbox_full": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "image_center_full": image_center,
                "center_world": est["centroid_world"],
                "center_camera": world_point_to_camera(est["centroid_world"], T_cw),
                "axis_world": est["axis_world"],
                "yaw": est["yaw"],
                "geometry_vector": est["geometry_vector"],
                "state_vector": est["state_vector"],
                "pca_extents": est["pca_extents"],
                "planar_radius": est["planar_radius"],
                "point_cloud_path": str(cloud_path),
                "body_id": int(body_id),
                "body_name": (body_id_to_name or {}).get(int(body_id), ""),
            })
            next_pid += 1

    # Scene-level duplicate-cap cleanup.  The failure seen in the previous run was
    # a hemisphere fragment attached to nearly every isolated object.  If a body
    # already has a stronger non-hemisphere primitive, remove a weaker hemisphere
    # fragment rather than presenting it to Gemini as a real object part.
    records_by_body = {}
    for rr in primitive_records:
        records_by_body.setdefault(int(rr["body_id"]), []).append(rr)
    remove_pids = set()
    for _bid, _recs in records_by_body.items():
        non_hemi = [r for r in _recs if r["primitive_type"] != "hemisphere"]
        if not non_hemi:
            continue
        strongest_non = max(float(r.get("pixel_count", 0)) * (0.6 + 0.4 * float(r.get("mean_confidence", 0.0))) for r in non_hemi)
        for r in _recs:
            if r["primitive_type"] != "hemisphere":
                continue
            hq = float(r.get("pixel_count", 0)) * (0.6 + 0.4 * float(r.get("mean_confidence", 0.0)))
            if hq < 1.20 * strongest_non or float(r.get("foreground_fraction", 0.0)) < 0.45:
                remove_pids.add(int(r["primitive_id"]))
    if remove_pids:
        primitive_records = [r for r in primitive_records if int(r["primitive_id"]) not in remove_pids]
        full_instances = [r for r in full_instances if int(r["primitive_id"]) not in remove_pids]
        for pid in remove_pids:
            mask_by_pid.pop(int(pid), None)
        print("MR-Former postprocess suppressed spurious hemisphere-cap IDs:", sorted(remove_pids))

    if not primitive_records:
        primitive_mask_image, primitive_mask_path = _make_full_primitive_mask_image([], (h, w))
        if allow_empty:
            # A selected rigid object can temporarily disappear from the MR-Former
            # post-processing because of self-occlusion, thin hollow geometry, or
            # a low-confidence frame.  Return a valid empty observation instead
            # of aborting the entire manipulation task.  The caller can use the
            # last valid rigid primitive decomposition as a temporal fallback.
            print(
                "WARNING: MR-Former temporary selected-object dropout: no usable "
                "primitive instances in this frame. Returning an empty observation "
                "for rigid-tracking fallback instead of terminating."
            )
            graph = {
                "source": "mrformer_temporary_dropout",
                "objects": [],
            }
            return {
                "primitive_records": [],
                "object_primitive_graph": graph,
                "rgb_objects_only": clean_rgb,
                "primitive_mask_image": primitive_mask_image,
                "primitive_mask_path": str(primitive_mask_path),
                "T_world_from_camera": T_wc,
                "T_camera_from_world": T_cw,
                "camera_frame_convention": {
                    "x": "+Xc image right",
                    "y": "+Yc image down",
                    "z": "+Zc optical forward",
                },
            }
        raise RuntimeError(
            "MR-Former produced no usable primitive instances after per-object PyBullet isolation."
        )

    primitive_mask_image, primitive_mask_path = _make_full_primitive_mask_image(full_instances, (h, w))
    if use_gemini:
        graph = gemini_group_primitives(clean_rgb, primitive_mask_image, primitive_records, body_metadata=body_metadata)
    else:
        graph = _normalize_fallback_com_uv(_simulation_grouping_fallback(primitive_records, body_metadata), w, h)
        graph["source"] = "body_tracking_after_initial_selection"

    graph = _lift_group_coms(graph, primitive_records, mask_by_pid, depth_m, intr, camera_eye, camera_target, camera_up, T_cw)

    serializable_records = []
    for r in primitive_records:
        serializable_records.append({
            **{k: v for k, v in r.items() if k not in {"center_world", "center_camera", "axis_world", "geometry_vector", "state_vector", "pca_extents"}},
            "center_world": np.asarray(r["center_world"]).tolist(),
            "center_camera": np.asarray(r["center_camera"]).tolist(),
            "axis_world": np.asarray(r["axis_world"]).tolist(),
            "geometry_vector": np.asarray(r["geometry_vector"]).tolist(),
            "state_vector": np.asarray(r["state_vector"]).tolist(),
            "pca_extents": np.asarray(r["pca_extents"]).tolist(),
        })

    with open(OUTPUT_DIR / "online_perception_result.json", "w", encoding="utf-8") as f:
        json.dump({
            "object_primitive_graph": graph,
            "camera_frame_convention": {"x": "+Xc image right", "y": "+Yc image down", "z": "+Zc optical forward"},
            "T_world_from_camera": T_wc.tolist(),
            "T_camera_from_world": T_cw.tolist(),
            "primitive_mask_path": str(primitive_mask_path),
            "primitive_records": serializable_records,
        }, f, indent=2)

    return {
        "primitive_records": primitive_records,
        "object_primitive_graph": graph,
        "rgb_objects_only": clean_rgb,
        "primitive_mask_image": primitive_mask_image,
        "primitive_mask_path": str(primitive_mask_path),
        "T_world_from_camera": T_wc,
        "T_camera_from_world": T_cw,
        "camera_frame_convention": {"x": "+Xc image right", "y": "+Yc image down", "z": "+Zc optical forward"},
    }

# =============================================================================
# WHOLE-OBJECT MOMENT-OF-INERTIA ESTIMATION FROM PERCEIVED PRIMITIVES
# =============================================================================

def _primitive_volume_from_geometry(record: dict) -> float:
    """Approximate primitive volume from MR-Former/PCA geometry."""
    g = np.asarray(record.get("geometry_vector", np.zeros(8)), dtype=np.float64)
    L, W, H, Ro, Ri = [max(0.0, float(x)) for x in g[:5]]
    ptype = str(record.get("primitive_type", ""))
    if ptype == "cuboid":
        return max(L * W * H, 1e-9)
    if ptype == "sphere":
        return max((4.0 / 3.0) * math.pi * Ro**3, 1e-9)
    if ptype == "hemisphere":
        return max((2.0 / 3.0) * math.pi * Ro**3, 1e-9)
    if ptype == "cylinder":
        return max(math.pi * Ro**2 * max(H, 1e-5), 1e-9)
    if ptype == "ring":
        return max(math.pi * max(Ro**2 - Ri**2, 1e-10) * max(H, 1e-5), 1e-9)
    if ptype == "stick":
        return max(math.pi * Ro**2 * max(L, 1e-5), 1e-9)
    if ptype == "cone":
        return max((1.0 / 3.0) * math.pi * Ro**2 * max(H, 1e-5), 1e-9)
    return 1e-9


def _primitive_izz_about_own_com_per_kg(record: dict) -> float:
    """Return approximate Izz/m about a primitive's own COM in world Z.

    For axially symmetric primitives, the inertia tensor is projected onto world Z
    using the perceived primitive axis.  The result is intentionally an analytic
    prior, not a replacement for the learned recurrent dynamics.
    """
    g = np.asarray(record.get("geometry_vector", np.zeros(8)), dtype=np.float64)
    L, W, H, Ro, Ri, Ax, Ay, Az = [float(x) for x in g]
    L, W, H, Ro, Ri = [max(0.0, x) for x in (L, W, H, Ro, Ri)]
    ptype = str(record.get("primitive_type", ""))

    if ptype == "cuboid":
        # Assuming the perceived cuboid's L/W axes are in the tabletop plane.
        return max((L * L + W * W) / 12.0, 1e-10)
    if ptype == "sphere":
        return max((2.0 / 5.0) * Ro * Ro, 1e-10)
    if ptype == "hemisphere":
        # Around the symmetry axis through its COM, same z-axis expression as the
        # corresponding solid spherical half. This is sufficient as a prior.
        return max((2.0 / 5.0) * Ro * Ro, 1e-10)

    axis = np.asarray([Ax, Ay, Az], dtype=np.float64)
    n = float(np.linalg.norm(axis))
    if n < 1e-9:
        axis = np.array([0.0, 0.0, 1.0])
    else:
        axis /= n
    cos2 = float(np.clip(axis[2] * axis[2], 0.0, 1.0))

    if ptype in {"cylinder", "stick"}:
        length = H if ptype == "cylinder" else L
        I_axis = 0.5 * Ro * Ro
        I_perp = (3.0 * Ro * Ro + length * length) / 12.0
        return max(I_perp + (I_axis - I_perp) * cos2, 1e-10)
    if ptype == "ring":
        rr = Ro * Ro + Ri * Ri
        I_axis = 0.5 * rr
        I_perp = (3.0 * rr + H * H) / 12.0
        return max(I_perp + (I_axis - I_perp) * cos2, 1e-10)
    if ptype == "cone":
        I_axis = 3.0 * Ro * Ro / 10.0
        I_perp = 3.0 * Ro * Ro / 20.0 + 3.0 * H * H / 80.0
        return max(I_perp + (I_axis - I_perp) * cos2, 1e-10)
    return 1e-6


def estimate_whole_object_inertia_from_primitives(
    records: List[dict],
    object_com_world,
    total_mass_kg: float = 1.0,
) -> dict:
    """Estimate whole-object planar inertia Izz about the supplied object COM.

    Important: COM alone is not sufficient to determine inertia.  This function
    combines the whole-object COM with MR-Former primitive geometry and a total
    mass estimate.  Primitive masses are distributed in proportion to approximate
    primitive volumes, then the parallel-axis theorem moves each primitive inertia
    to the whole-object COM.

    In simulation ``total_mass_kg`` can be obtained from PyBullet.  On the real
    robot it can come from a scale, CAD, or a mass estimator.
    """
    valid = [r for r in records if np.all(np.isfinite(np.asarray(r.get("center_world", [np.nan]*3), dtype=float)))]
    mass = max(float(total_mass_kg), 1e-6)
    com = np.asarray(object_com_world, dtype=np.float64).reshape(-1)
    if com.size < 3:
        com = np.pad(com, (0, 3 - com.size))
    if not valid:
        return {
            "mass_kg": mass,
            "com_world": com[:3].astype(np.float32),
            "Izz_kgm2": 1e-6,
            "Izz_per_kg_m2": 1e-6 / mass,
            "primitive_terms": [],
            "source": "no_primitives_fallback",
        }

    volumes = np.asarray([_primitive_volume_from_geometry(r) for r in valid], dtype=np.float64)
    volumes = np.maximum(volumes, 1e-12)
    fractions = volumes / volumes.sum()

    total_izz = 0.0
    terms = []
    for r, frac in zip(valid, fractions):
        mi = mass * float(frac)
        center = np.asarray(r["center_world"], dtype=np.float64)
        dxy = center[:2] - com[:2]
        own_per_kg = _primitive_izz_about_own_com_per_kg(r)
        own = mi * own_per_kg
        parallel = mi * float(np.dot(dxy, dxy))
        term = own + parallel
        total_izz += term
        terms.append({
            "primitive_id": int(r.get("primitive_id", -1)),
            "primitive_type": str(r.get("primitive_type", "unknown")),
            "mass_fraction": float(frac),
            "estimated_mass_kg": float(mi),
            "center_to_object_com_m": float(np.linalg.norm(dxy)),
            "own_Izz_kgm2": float(own),
            "parallel_axis_Izz_kgm2": float(parallel),
            "total_term_Izz_kgm2": float(term),
        })

    total_izz = max(float(total_izz), 1e-8)
    return {
        "mass_kg": float(mass),
        "com_world": com[:3].astype(np.float32),
        "Izz_kgm2": total_izz,
        "Izz_per_kg_m2": float(total_izz / mass),
        "primitive_terms": terms,
        "source": "MRFormer_geometry+whole_object_COM+mass+parallel_axis",
    }
