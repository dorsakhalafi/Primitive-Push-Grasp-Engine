from __future__ import annotations

GRASP_ENGINE_BUILD_ID = "2026-09-22_PRIMITIVE_GRASP_LIBRARY_V1"

"""Primitive-based grasp library for the combined Push/Grasp Engine.

Design goals
------------
1) Reuse MR-Former primitive masks / sized point clouds from perception.py.
2) Reuse Gemini object grouping, whole-object COM, and hollow/solid semantics.
3) Generate several geometry-aware grasp candidates per primitive, then select
   ONE best grasp for each rigid object by:
      - closeness of grasp midpoint to the whole-object COM,
      - clearance from other scene objects along the approach corridor,
      - gripper-width feasibility,
      - Franka reachability.
4) Execute the selected grasp with the same Franka Panda already present in the
   PyBullet project.

The grasp generator follows the primitive rules in the user's Grasp Engine notes:
- cuboid: opposite-face / edge grasps, no bottom-through-table grasp;
- sphere: multiple top/side diametric grasps;
- hemisphere: sphere-like lower/side grasps plus rim grasps;
- cylinder: radial side grasps; for lying cylinders, grasps are distributed
  along the cylinder axis;
- ring: equally spaced rim grasps, including hollow edge grasps when accessible;
- stick: several top grasps distributed along the long axis plus end options;
- cone: cylinder-like side grasps, with top/rim options depending on orientation.

A final grasp candidate stores two finger contacts g1,g2 and their midpoint gp.
The user asked for one grasp point per object; in this implementation that means
one FINAL grasp candidate per object (gp + its two finger contacts g1/g2).
"""

import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class GraspEngineConfig:
    # Franka gripper: each finger joint is nominally 0.04 m, ~0.08 m total.
    max_opening_m: float = 0.080
    min_opening_m: float = 0.006
    preferred_clearance_m: float = 0.060
    minimum_other_object_clearance_m: float = 0.010
    pregrasp_height_m: float = 0.105
    lift_height_m: float = 0.145
    approach_steps: int = 80
    move_steps: int = 150
    close_steps: int = 90
    lift_steps: int = 170
    settle_steps: int = 80
    realtime_scale: float = 0.10
    # Candidate scoring weights.
    w_com: float = 2.4
    w_clearance: float = 2.0
    w_width: float = 0.7
    w_reach: float = 0.9
    w_family: float = 0.35
    # We intentionally prefer top-down execution because it is the most reliable
    # common approach for all seven primitive classes in this tabletop scene.
    top_down_bonus: float = 0.20
    # A grasp may be attached to the end-effector by a fixed constraint after
    # closing. This models a successful parallel-jaw grasp robustly in simulation
    # even though the Panda's collision links are disabled during push execution.
    use_fixed_constraint_after_close: bool = True
    grasp_constraint_max_force_n: float = 300.0


# -----------------------------------------------------------------------------
# Basic geometry helpers
# -----------------------------------------------------------------------------

def _norm(v, eps: float = 1e-9) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(a))
    if n < eps:
        return np.zeros_like(a)
    return a / n


def _wrap_pi(a: float) -> float:
    return float(math.atan2(math.sin(a), math.cos(a)))


def _yaw_from_axis_xy(axis: np.ndarray) -> float:
    a = np.asarray(axis, dtype=np.float64)
    if np.linalg.norm(a[:2]) < 1e-8:
        return 0.0
    return float(math.atan2(a[1], a[0]))


def _point_to_aabb_distance(point: Sequence[float], aabb_min, aabb_max) -> float:
    q = np.asarray(point, dtype=np.float64)
    lo = np.asarray(aabb_min, dtype=np.float64)
    hi = np.asarray(aabb_max, dtype=np.float64)
    d = np.maximum(np.maximum(lo - q, 0.0), q - hi)
    return float(np.linalg.norm(d))


def _union_body_aabb(body_id: int) -> Tuple[np.ndarray, np.ndarray]:
    mins, maxs = [], []
    for link in [-1] + list(range(p.getNumJoints(int(body_id)))):
        try:
            lo, hi = p.getAABB(int(body_id), int(link))
        except Exception:
            continue
        mins.append(np.asarray(lo, dtype=np.float64))
        maxs.append(np.asarray(hi, dtype=np.float64))
    if not mins:
        pos, _ = p.getBasePositionAndOrientation(int(body_id))
        pos = np.asarray(pos, dtype=np.float64)
        return pos - 0.02, pos + 0.02
    return np.min(np.stack(mins), axis=0), np.max(np.stack(maxs), axis=0)


def _other_object_clearance(
    candidate_points: Iterable[np.ndarray],
    target_body_id: int,
    all_object_ids: Sequence[int],
) -> float:
    points = [np.asarray(q, dtype=np.float64).reshape(3) for q in candidate_points]
    best = float("inf")
    for bid in all_object_ids:
        if int(bid) == int(target_body_id):
            continue
        lo, hi = _union_body_aabb(int(bid))
        for q in points:
            best = min(best, _point_to_aabb_distance(q, lo, hi))
    return float(best if math.isfinite(best) else 1.0)


def _sample_approach_corridor(gp: np.ndarray, approach_axis: np.ndarray, distance: float) -> List[np.ndarray]:
    a = _norm(approach_axis)
    pre = np.asarray(gp, dtype=np.float64) - a * float(distance)
    return [pre * (1.0 - t) + np.asarray(gp, dtype=np.float64) * t for t in np.linspace(0.0, 1.0, 9)]


def _load_cloud(record: dict) -> np.ndarray:
    path = record.get("point_cloud_path")
    if path:
        try:
            data = np.load(str(path))
            pts = np.asarray(data["points_world"], dtype=np.float64)
            if pts.ndim == 2 and pts.shape[1] == 3 and len(pts) >= 8:
                return pts
        except Exception:
            pass
    c = np.asarray(record.get("center_world", [0.0, 0.0, 0.0]), dtype=np.float64)
    r = max(float(record.get("planar_radius", 0.02)), 0.01)
    # Conservative synthetic fallback used only if the saved MR-Former cloud is
    # unavailable.  It does not change the primitive class.
    return np.array([
        c + [r, 0, 0], c + [-r, 0, 0], c + [0, r, 0], c + [0, -r, 0],
        c + [0, 0, r], c + [0, 0, -r], c + [0.5*r, 0.5*r, 0], c + [-0.5*r, -0.5*r, 0],
    ], dtype=np.float64)


def _pca_frame(points: np.ndarray):
    pts = np.asarray(points, dtype=np.float64)
    center = pts.mean(axis=0)
    d = pts - center
    cov = np.cov(d.T) if len(pts) > 3 else np.eye(3)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    axes = vecs[:, order]
    # Force a right-handed frame.
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1.0
    proj = d @ axes
    lo = proj.min(axis=0)
    hi = proj.max(axis=0)
    ext = hi - lo
    return center, axes, lo, hi, ext


def _horizontal_axis(v: Sequence[float], fallback=(1.0, 0.0, 0.0)) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64).copy()
    a[2] = 0.0
    if np.linalg.norm(a) < 1e-8:
        a = np.asarray(fallback, dtype=np.float64)
    return _norm(a)


def _candidate(
    record: dict,
    family: str,
    gp: Sequence[float],
    closing_axis: Sequence[float],
    opening_m: float,
    approach_axis: Sequence[float]=(0.0, 0.0, -1.0),
    occupancy: str="unknown",
    inner_accessible: bool=False,
    family_bonus: float=0.0,
) -> dict:
    gp = np.asarray(gp, dtype=np.float64).reshape(3)
    close = _horizontal_axis(closing_axis)
    opening = float(max(opening_m, 1e-4))
    g1 = gp + 0.5 * opening * close
    g2 = gp - 0.5 * opening * close
    approach = _norm(approach_axis)
    if np.linalg.norm(approach) < 1e-8:
        approach = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return {
        "primitive_id": int(record.get("primitive_id", -1)),
        "primitive_type": str(record.get("primitive_type", "unknown")),
        "grasp_family": str(family),
        "gp_world": gp,
        "g1_world": g1,
        "g2_world": g2,
        "closing_axis_world": close,
        "approach_axis_world": approach,
        "required_opening_m": opening,
        "occupancy": str(occupancy),
        "inner_accessible": bool(inner_accessible),
        "family_bonus": float(family_bonus),
    }


# -----------------------------------------------------------------------------
# Primitive-specific grasp library
# -----------------------------------------------------------------------------

def _record_property(selected_object: dict, primitive_id: int) -> Tuple[str, bool]:
    for pp in selected_object.get("primitive_properties", []):
        if int(pp.get("primitive_id", -999)) == int(primitive_id):
            return str(pp.get("occupancy", "unknown")).lower(), bool(pp.get("inner_accessible", False))
    return "unknown", False


def _cuboid_candidates(record: dict, selected_object: dict) -> List[dict]:
    pts = _load_cloud(record)
    center, axes, lo, hi, ext = _pca_frame(pts)
    occ, accessible = _record_property(selected_object, record["primitive_id"])
    # Rank PCA axes by how horizontal they are.  Top-down grasps close across a
    # horizontal object dimension while approaching from +Z.
    horiz = sorted(range(3), key=lambda i: abs(float(axes[2, i])))
    out = []
    for rank, i in enumerate(horiz[:2]):
        close = _horizontal_axis(axes[:, i])
        opening = float(ext[i] * 0.94)
        ortho = _horizontal_axis(np.cross([0.0, 0.0, 1.0], close), fallback=(0.0, 1.0, 0.0))
        for s in (0.0, -0.18, +0.18):
            gp = center + s * max(float(ext[(i+1) % 3]), 0.02) * ortho
            out.append(_candidate(record, f"cuboid_opposite_faces_{rank}", gp, close, opening, occupancy=occ, inner_accessible=accessible, family_bonus=0.16))
    # Hollow cuboid: add edge-biased grasps and avoid relying on a top surface.
    if occ == "hollow":
        for sign in (-1.0, 1.0):
            close = _horizontal_axis(axes[:, horiz[0]])
            gp = center + sign * 0.30 * max(ext[horiz[1]], 0.02) * _horizontal_axis(axes[:, horiz[1]])
            out.append(_candidate(record, "hollow_cuboid_open_edge", gp, close, max(0.012, 0.25*ext[horiz[0]]), occupancy=occ, inner_accessible=accessible, family_bonus=0.20))
    return out[:8]


def _sphere_candidates(record: dict, selected_object: dict) -> List[dict]:
    pts = _load_cloud(record)
    center = pts.mean(axis=0)
    occ, accessible = _record_property(selected_object, record["primitive_id"])
    geom = np.asarray(record.get("geometry_vector", np.zeros(8)), dtype=float)
    r = float(geom[3]) if len(geom) > 3 and geom[3] > 1e-4 else float(np.median(np.linalg.norm(pts-center, axis=1)))
    # User's notes specify contact separation >=80% of diameter for stability.
    opening = max(0.8 * 2.0 * r, 0.012)
    out = []
    for k in range(6):
        phi = k * math.pi / 6.0
        close = np.array([math.cos(phi), math.sin(phi), 0.0])
        out.append(_candidate(record, "sphere_top_diametric", center, close, opening, occupancy=occ, inner_accessible=accessible, family_bonus=0.14))
    return out


def _hemisphere_candidates(record: dict, selected_object: dict) -> List[dict]:
    pts = _load_cloud(record)
    center = pts.mean(axis=0)
    occ, accessible = _record_property(selected_object, record["primitive_id"])
    geom = np.asarray(record.get("geometry_vector", np.zeros(8)), dtype=float)
    r = max(float(geom[3]) if len(geom) > 3 else 0.0, float(record.get("planar_radius", 0.025)))
    out = []
    # Filled: lower-half sphere-like grasps plus two rim-side grasps.
    for k in range(4):
        phi = k * math.pi / 4.0
        gp = center.copy()
        gp[2] -= 0.15 * r
        out.append(_candidate(record, "hemisphere_lower", gp, [math.cos(phi), math.sin(phi), 0], 1.55*r, occupancy=occ, inner_accessible=accessible, family_bonus=0.10))
    rim_count = 4 if occ == "hollow" else 2
    for k in range(rim_count):
        phi = 2.0 * math.pi * k / rim_count
        radial = np.array([math.cos(phi), math.sin(phi), 0.0])
        gp = center + 0.55 * r * radial
        out.append(_candidate(record, "hemisphere_rim_edge", gp, radial, max(0.012, 0.28*r), occupancy=occ, inner_accessible=accessible, family_bonus=0.18 if occ == "hollow" else 0.08))
    return out[:8]


def _cylinder_candidates(record: dict, selected_object: dict) -> List[dict]:
    pts = _load_cloud(record)
    center = pts.mean(axis=0)
    occ, accessible = _record_property(selected_object, record["primitive_id"])
    geom = np.asarray(record.get("geometry_vector", np.zeros(8)), dtype=float)
    axis = _norm(geom[5:8]) if len(geom) >= 8 else np.array([0, 0, 1.0])
    r = max(float(geom[3]) if len(geom) > 3 else 0.0, 0.012)
    h = max(float(geom[2]) if len(geom) > 2 else 0.0, 0.025)
    out = []
    upright = abs(float(axis[2])) > 0.72
    if upright:
        for k in range(6):
            phi = k * math.pi / 6.0
            close = np.array([math.cos(phi), math.sin(phi), 0.0])
            out.append(_candidate(record, "cylinder_radial", center, close, 1.85*r, occupancy=occ, inner_accessible=accessible, family_bonus=0.15))
        if occ == "hollow" and accessible:
            for k in range(2):
                phi = k * math.pi / 2.0
                radial = np.array([math.cos(phi), math.sin(phi), 0.0])
                gp = center + 0.62*r*radial
                out.append(_candidate(record, "hollow_cylinder_rim", gp, radial, max(0.010, 0.35*r), occupancy=occ, inner_accessible=accessible, family_bonus=0.20))
    else:
        axy = _horizontal_axis(axis)
        close = _horizontal_axis(np.cross([0, 0, 1.0], axy), fallback=(0, 1, 0))
        for s in np.linspace(-0.32, 0.32, 6):
            gp = center + s * h * axy
            out.append(_candidate(record, "lying_cylinder_side", gp, close, 1.8*r, occupancy=occ, inner_accessible=accessible, family_bonus=0.14))
    return out[:8]


def _ring_candidates(record: dict, selected_object: dict) -> List[dict]:
    pts = _load_cloud(record)
    center = pts.mean(axis=0)
    occ, accessible = _record_property(selected_object, record["primitive_id"])
    geom = np.asarray(record.get("geometry_vector", np.zeros(8)), dtype=float)
    rout = max(float(geom[3]) if len(geom) > 3 else 0.0, float(record.get("planar_radius", 0.03)))
    rin = float(geom[4]) if len(geom) > 4 and geom[4] > 1e-4 else 0.5*rout
    thickness = max(rout - rin, 0.008)
    mid = 0.5 * (rout + rin)
    out = []
    # Five equally spaced material-edge grasps, matching the user's ring rule.
    for k in range(5):
        phi = 2.0 * math.pi * k / 5.0
        radial = np.array([math.cos(phi), math.sin(phi), 0.0])
        gp = center + mid * radial
        out.append(_candidate(record, "ring_edge", gp, radial, min(0.95*thickness, 0.98*rout), occupancy="hollow" if occ == "unknown" else occ, inner_accessible=accessible, family_bonus=0.26))
    # Add one diametric outer grasp if it fits. It is useful when the hole is not
    # accessible but the overall ring diameter is within the gripper width.
    out.append(_candidate(record, "ring_outer_diametric", center, [1,0,0], 1.8*rout, occupancy="hollow" if occ == "unknown" else occ, inner_accessible=accessible, family_bonus=0.08))
    return out


def _stick_candidates(record: dict, selected_object: dict) -> List[dict]:
    pts = _load_cloud(record)
    center, axes, lo, hi, ext = _pca_frame(pts)
    occ, accessible = _record_property(selected_object, record["primitive_id"])
    long_axis = _horizontal_axis(axes[:, 0])
    close = _horizontal_axis(np.cross([0,0,1.0], long_axis), fallback=(0,1,0))
    length = max(float(ext[0]), 0.03)
    diameter = max(float(0.5*(ext[1]+ext[2])), 0.008)
    out = []
    # Six top grasps distributed at L/5-style intervals along the stick.
    for s in np.linspace(-0.42, 0.42, 6):
        gp = center + s * length * long_axis
        out.append(_candidate(record, "stick_top_distributed", gp, close, diameter, occupancy=occ, inner_accessible=accessible, family_bonus=0.20))
    # Two optional end grasps. Collision scoring automatically suppresses them if
    # an end is matched/occluded by another primitive or nearby object.
    for s in (-0.47, +0.47):
        gp = center + s * length * long_axis
        out.append(_candidate(record, "stick_end", gp, close, diameter, occupancy=occ, inner_accessible=accessible, family_bonus=0.08))
    return out


def _cone_candidates(record: dict, selected_object: dict) -> List[dict]:
    pts = _load_cloud(record)
    center = pts.mean(axis=0)
    occ, accessible = _record_property(selected_object, record["primitive_id"])
    geom = np.asarray(record.get("geometry_vector", np.zeros(8)), dtype=float)
    axis = _norm(geom[5:8]) if len(geom) >= 8 else np.array([0, 0, 1.0])
    r = max(float(geom[3]) if len(geom) > 3 else 0.0, 0.015)
    h = max(float(geom[2]) if len(geom) > 2 else 0.0, 0.04)
    upright = abs(float(axis[2])) > 0.72
    out = []
    if upright:
        for k in range(4):
            phi = 2*math.pi*k/4.0
            close = np.array([math.cos(phi), math.sin(phi), 0.0])
            out.append(_candidate(record, "cone_upright_side", center, close, 1.45*r, occupancy=occ, inner_accessible=accessible, family_bonus=0.14))
        if occ == "hollow":
            for k in range(2):
                phi = k*math.pi
                radial = np.array([math.cos(phi), math.sin(phi), 0.0])
                gp = center + 0.55*r*radial
                out.append(_candidate(record, "cone_hollow_rim", gp, radial, max(0.010, 0.30*r), occupancy=occ, inner_accessible=accessible, family_bonus=0.18))
        else:
            out.append(_candidate(record, "cone_top_side_pair", center, [1,0,0], min(1.4*r, 0.075), occupancy=occ, inner_accessible=accessible, family_bonus=0.10))
    else:
        axy = _horizontal_axis(axis)
        close = _horizontal_axis(np.cross([0,0,1.0], axy), fallback=(0,1,0))
        for s in np.linspace(-0.30, 0.30, 4):
            gp = center + s*h*axy
            out.append(_candidate(record, "cone_lying_side", gp, close, 1.35*r, occupancy=occ, inner_accessible=accessible, family_bonus=0.15))
        out.append(_candidate(record, "cone_lying_top", center, close, 1.2*r, occupancy=occ, inner_accessible=accessible, family_bonus=0.14))
    return out[:8]


def generate_primitive_grasp_candidates(record: dict, selected_object: dict) -> List[dict]:
    ptype = str(record.get("primitive_type", "")).lower()
    if ptype == "cuboid":
        return _cuboid_candidates(record, selected_object)
    if ptype == "sphere":
        return _sphere_candidates(record, selected_object)
    if ptype == "hemisphere":
        return _hemisphere_candidates(record, selected_object)
    if ptype == "cylinder":
        return _cylinder_candidates(record, selected_object)
    if ptype == "ring":
        return _ring_candidates(record, selected_object)
    if ptype == "stick":
        return _stick_candidates(record, selected_object)
    if ptype == "cone":
        return _cone_candidates(record, selected_object)
    return []


# -----------------------------------------------------------------------------
# Candidate scoring / one grasp per object
# -----------------------------------------------------------------------------

def _candidate_reachable(franka_robot, gp: np.ndarray, config: GraspEngineConfig) -> bool:
    if franka_robot is None:
        return True
    # Existing Panda helper expects the red-pusher center, so checking a point
    # slightly below the gripper center gives a conservative workspace preview.
    try:
        probe = np.asarray(gp, dtype=np.float64) - np.array([0.0, 0.0, float(getattr(franka_robot, "hand_above_pusher_m", 0.035))])
        return bool(franka_robot.is_tool_center_reachable(probe))
    except Exception:
        return True


def _score_candidate(
    cand: dict,
    whole_com_world: np.ndarray,
    target_body_id: int,
    all_object_ids: Sequence[int],
    franka_robot,
    config: GraspEngineConfig,
    table_top_z: float,
) -> dict:
    c = dict(cand)
    gp = np.asarray(c["gp_world"], dtype=np.float64)
    g1 = np.asarray(c["g1_world"], dtype=np.float64)
    g2 = np.asarray(c["g2_world"], dtype=np.float64)
    approach = np.asarray(c["approach_axis_world"], dtype=np.float64)

    com = np.asarray(whole_com_world, dtype=np.float64)
    if com.shape != (3,) or not np.all(np.isfinite(com)):
        com = gp.copy()
    com_distance = float(np.linalg.norm(gp - com))
    com_score = 1.0 / (1.0 + com_distance / 0.045)

    corridor = _sample_approach_corridor(gp, approach, config.pregrasp_height_m)
    clearance = _other_object_clearance(corridor + [g1, g2], target_body_id, all_object_ids)
    clearance_score = float(np.clip(clearance / max(config.preferred_clearance_m, 1e-4), 0.0, 1.0))

    opening = float(c["required_opening_m"])
    width_ok = config.min_opening_m <= opening <= config.max_opening_m * 1.06
    width_score = 1.0 if width_ok else max(0.0, 1.0 - abs(opening - config.max_opening_m) / max(config.max_opening_m, 1e-4))

    reachable = _candidate_reachable(franka_robot, gp, config)
    reach_score = 1.0 if reachable else 0.0

    # No bottom-through-table grasps: both contact points and midpoint must be
    # above the support plane.  This directly encodes the user's cuboid rule.
    above_table = min(float(g1[2]), float(g2[2]), float(gp[2])) > float(table_top_z) + 0.002
    clearance_ok = clearance >= config.minimum_other_object_clearance_m

    approach_bonus = config.top_down_bonus if float(c["approach_axis_world"][2]) < -0.65 else 0.0
    score = (
        config.w_com * com_score
        + config.w_clearance * clearance_score
        + config.w_width * width_score
        + config.w_reach * reach_score
        + config.w_family * float(c.get("family_bonus", 0.0))
        + approach_bonus
    )

    c.update({
        "com_distance_m": com_distance,
        "com_score": com_score,
        "other_object_clearance_m": clearance,
        "clearance_score": clearance_score,
        "width_ok": bool(width_ok),
        "reachable_preview": bool(reachable),
        "above_table": bool(above_table),
        "clearance_ok": bool(clearance_ok),
        "score": float(score),
        "feasible": bool(width_ok and above_table and clearance_ok and reachable),
    })
    return c


def _json_candidate(c: dict) -> dict:
    out = {}
    for k, v in c.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def build_grasp_library(
    catalog: List[dict],
    primitive_records: List[dict],
    all_object_ids: Sequence[int],
    franka_robot=None,
    table_top_z: float=0.0,
    config: Optional[GraspEngineConfig]=None,
) -> Dict[int, dict]:
    """Generate several primitive grasps and retain one best grasp per object."""
    cfg = config or GraspEngineConfig()
    by_pid = {int(r["primitive_id"]): r for r in primitive_records}
    library: Dict[int, dict] = {}

    for obj in catalog:
        body_id = int(obj["body_id"])
        whole_com = np.asarray(obj.get("gemini_com_world", [np.nan]*3), dtype=np.float64)
        if not np.all(np.isfinite(whole_com)):
            whole_com = np.asarray(obj.get("true_com_world", [0,0,0]), dtype=np.float64)

        raw = []
        for pid in obj.get("primitive_ids", []):
            rec = by_pid.get(int(pid))
            if rec is None:
                continue
            raw.extend(generate_primitive_grasp_candidates(rec, obj))

        scored = [
            _score_candidate(c, whole_com, body_id, all_object_ids, franka_robot, cfg, table_top_z)
            for c in raw
        ]
        feasible = [c for c in scored if c.get("feasible", False)]
        pool = feasible if feasible else scored
        pool.sort(key=lambda c: float(c.get("score", -1e9)), reverse=True)
        best = pool[0] if pool else None
        library[int(obj["object_number"])] = {
            "object_number": int(obj["object_number"]),
            "object_label": str(obj.get("object_label", "object")),
            "body_id": body_id,
            "gemini_com_world": whole_com,
            "candidate_count": len(scored),
            "feasible_count": len(feasible),
            "best": best,
            "top_candidates": pool[:10],
        }

    serial = {
        str(k): {
            **{kk: vv for kk, vv in v.items() if kk not in {"gemini_com_world", "best", "top_candidates"}},
            "gemini_com_world": np.asarray(v["gemini_com_world"]).tolist(),
            "best": _json_candidate(v["best"]) if v["best"] is not None else None,
            "top_candidates": [_json_candidate(x) for x in v["top_candidates"]],
        }
        for k, v in library.items()
    }
    with open(OUTPUT_DIR / "grasp_library.json", "w", encoding="utf-8") as f:
        json.dump({"build": GRASP_ENGINE_BUILD_ID, "objects": serial}, f, indent=2)
    return library


def print_grasp_library(library: Dict[int, dict]) -> None:
    print("\n" + "=" * 132)
    print("PRIMITIVE GRASP LIBRARY -- ONE FINAL COM/CLEARANCE-OPTIMIZED GRASP PER OBJECT")
    print("=" * 132)
    print("OBJ | LABEL              | FINAL PRIMITIVE/FAMILY             | COM DIST | OTHER CLEARANCE | OPENING | SCORE")
    print("-" * 132)
    for objnum in sorted(library):
        item = library[objnum]
        b = item.get("best")
        if b is None:
            print(f"{objnum:>3} | {item['object_label']:<18} | NO GRASP CANDIDATE")
            continue
        print(
            f"{objnum:>3} | {item['object_label']:<18} | "
            f"P{b['primitive_id']} {b['primitive_type']}/{b['grasp_family']:<20} | "
            f"{b['com_distance_m']*1000:7.1f} mm | {b['other_object_clearance_m']*1000:10.1f} mm | "
            f"{b['required_opening_m']*1000:6.1f} mm | {b['score']:.3f}"
        )
    print("=" * 132)
    print("gp = final grasp midpoint; g1/g2 = two parallel-jaw contact points.")
    print("Ranking favors COM closeness + collision clearance + gripper fit + Franka reachability.")


_GRASP_DEBUG_IDS: List[int] = []


def clear_grasp_debug() -> None:
    global _GRASP_DEBUG_IDS
    if not p.isConnected():
        _GRASP_DEBUG_IDS = []
        return
    for uid in _GRASP_DEBUG_IDS:
        try:
            p.removeUserDebugItem(int(uid))
        except Exception:
            pass
    _GRASP_DEBUG_IDS = []


def draw_grasp_library(library: Dict[int, dict], selected_only: Optional[int]=None) -> None:
    """Draw one final grasp pair per object in the PyBullet GUI."""
    clear_grasp_debug()
    if not p.isConnected():
        return
    for objnum in sorted(library):
        if selected_only is not None and int(objnum) != int(selected_only):
            continue
        item = library[objnum]
        b = item.get("best")
        if b is None:
            continue
        gp = np.asarray(b["gp_world"], dtype=float)
        g1 = np.asarray(b["g1_world"], dtype=float)
        g2 = np.asarray(b["g2_world"], dtype=float)
        pre = gp - np.asarray(b["approach_axis_world"], dtype=float) * 0.075
        _GRASP_DEBUG_IDS.append(p.addUserDebugLine(g1.tolist(), g2.tolist(), [0.0, 0.85, 0.15], 4.0, lifeTime=0))
        _GRASP_DEBUG_IDS.append(p.addUserDebugLine(pre.tolist(), gp.tolist(), [0.10, 0.35, 1.0], 2.5, lifeTime=0))
        _GRASP_DEBUG_IDS.append(p.addUserDebugText(
            f"G OBJ {objnum} | P{b['primitive_id']} {b['grasp_family']}",
            (gp + np.array([0,0,0.045])).tolist(), [0.0, 0.55, 0.05], 0.95, lifeTime=0
        ))


# -----------------------------------------------------------------------------
# Franka grasp execution
# -----------------------------------------------------------------------------

def _step_sim(steps: int, dt: float, realtime_scale: float=0.10):
    for _ in range(max(int(steps), 1)):
        if not p.isConnected():
            break
        p.stepSimulation()
        if realtime_scale > 0:
            time.sleep(float(dt) * float(realtime_scale))


def _command_fingers(franka_robot, total_opening_m: float, force: float=55.0):
    half = float(np.clip(0.5 * total_opening_m, 0.0, 0.040))
    for joint in franka_robot.finger_joint_indices:
        p.setJointMotorControl2(
            franka_robot.body_id, int(joint), p.POSITION_CONTROL,
            targetPosition=half, force=float(force), positionGain=0.35, velocityGain=1.0
        )


def _ik_for_grasp_pose(franka_robot, position_world: np.ndarray, wrist_yaw: float, downward=True):
    pos = np.asarray(position_world, dtype=np.float64).reshape(3)
    if downward:
        orientations = [
            p.getQuaternionFromEuler([math.pi, 0.0, float(wrist_yaw)]),
            p.getQuaternionFromEuler([math.pi, 0.18, float(wrist_yaw)]),
            getattr(franka_robot, "ee_orientation", p.getQuaternionFromEuler([math.pi,0,0])),
        ]
    else:
        orientations = [p.getQuaternionFromEuler([math.pi/2.0, 0.0, float(wrist_yaw)])]
    rest = franka_robot.current_arm_q()
    for orientation in orientations:
        try:
            ik = p.calculateInverseKinematics(
                bodyUniqueId=franka_robot.body_id,
                endEffectorLinkIndex=franka_robot.ee_link_index,
                targetPosition=pos.tolist(),
                targetOrientation=orientation,
                lowerLimits=franka_robot.lower_limits,
                upperLimits=franka_robot.upper_limits,
                jointRanges=franka_robot.joint_ranges,
                restPoses=rest.tolist(),
                maxNumIterations=260,
                residualThreshold=3e-4,
            )
        except Exception:
            try:
                ik = p.calculateInverseKinematics(
                    franka_robot.body_id, franka_robot.ee_link_index, pos.tolist(), orientation
                )
            except Exception:
                ik = None
        if ik is None or len(ik) < 7:
            continue
        q = np.asarray(ik[:7], dtype=np.float64)
        if np.all(np.isfinite(q)):
            return np.clip(q, np.asarray(franka_robot.lower_limits), np.asarray(franka_robot.upper_limits))
    # Last resort: position-only IK.
    try:
        ik = p.calculateInverseKinematics(franka_robot.body_id, franka_robot.ee_link_index, pos.tolist())
        if ik is not None and len(ik) >= 7:
            q = np.asarray(ik[:7], dtype=np.float64)
            if np.all(np.isfinite(q)):
                return np.clip(q, np.asarray(franka_robot.lower_limits), np.asarray(franka_robot.upper_limits))
    except Exception:
        pass
    return None


def _move_arm_to_pose(
    franka_robot,
    position_world: np.ndarray,
    wrist_yaw: float,
    dt: float,
    steps: int,
    realtime_scale: float,
    finger_opening_m: float,
) -> bool:
    q_goal = _ik_for_grasp_pose(franka_robot, position_world, wrist_yaw, downward=True)
    if q_goal is None:
        return False
    q0 = franka_robot.current_arm_q()
    n = max(int(steps), 2)
    for i in range(n):
        t = (i + 1) / n
        # Smoothstep avoids sudden joint acceleration.
        s = t*t*(3.0 - 2.0*t)
        q = (1.0 - s) * q0 + s * q_goal
        p.setJointMotorControlArray(
            franka_robot.body_id,
            franka_robot.arm_joint_indices,
            p.POSITION_CONTROL,
            targetPositions=q.tolist(),
            forces=franka_robot.max_forces,
            positionGains=[0.30] * 7,
            velocityGains=[1.0] * 7,
        )
        _command_fingers(franka_robot, finger_opening_m)
        p.stepSimulation()
        if realtime_scale > 0:
            time.sleep(float(dt) * float(realtime_scale))
    return True


def _attach_target_to_gripper(franka_robot, target_body_id: int, max_force: float) -> int:
    ls = p.getLinkState(franka_robot.body_id, franka_robot.ee_link_index, computeForwardKinematics=True)
    ee_pos, ee_q = ls[4], ls[5]
    obj_pos, obj_q = p.getBasePositionAndOrientation(int(target_body_id))
    inv_pos, inv_q = p.invertTransform(ee_pos, ee_q)
    rel_pos, rel_q = p.multiplyTransforms(inv_pos, inv_q, obj_pos, obj_q)
    cid = p.createConstraint(
        parentBodyUniqueId=franka_robot.body_id,
        parentLinkIndex=franka_robot.ee_link_index,
        childBodyUniqueId=int(target_body_id),
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=rel_pos,
        childFramePosition=[0, 0, 0],
        parentFrameOrientation=rel_q,
        childFrameOrientation=[0, 0, 0, 1],
    )
    try:
        p.changeConstraint(cid, maxForce=float(max_force))
    except Exception:
        pass
    return int(cid)


def execute_selected_grasp(
    franka_robot,
    target_object: dict,
    grasp_entry: dict,
    dt: float=1.0/240.0,
    table_top_z: float=0.0,
    config: Optional[GraspEngineConfig]=None,
) -> dict:
    """Execute the final selected primitive grasp and lift the object.

    The hand approaches from above, opens, descends to gp, closes, attaches the
    object to the Panda grasp target (robust simulation grasp), and lifts.
    """
    cfg = config or GraspEngineConfig()
    if not p.isConnected():
        raise RuntimeError("PyBullet is not connected.")
    best = grasp_entry.get("best")
    if best is None:
        return {"success": False, "reason": "no_grasp_candidate"}

    body_id = int(target_object["body_id"])
    gp = np.asarray(best["gp_world"], dtype=np.float64).copy()
    gp[2] = max(float(gp[2]), float(table_top_z) + 0.026)
    close_axis = _horizontal_axis(best["closing_axis_world"])
    wrist_yaw = _yaw_from_axis_xy(close_axis)
    pre = gp + np.array([0.0, 0.0, cfg.pregrasp_height_m], dtype=np.float64)
    lift = gp + np.array([0.0, 0.0, cfg.lift_height_m], dtype=np.float64)

    before_pos, _ = p.getBasePositionAndOrientation(body_id)
    before_pos = np.asarray(before_pos, dtype=np.float64)

    print("\n" + "=" * 92)
    print("PRIMITIVE GRASP EXECUTION")
    print("=" * 92)
    print(f"Object: OBJ {target_object['object_number']} {target_object['object_label']}")
    print(f"Primitive: P{best['primitive_id']} {best['primitive_type']}")
    print("Grasp family:", best["grasp_family"])
    print("gp WORLD:", np.round(gp, 4).tolist())
    print("g1 WORLD:", np.round(best["g1_world"], 4).tolist())
    print("g2 WORLD:", np.round(best["g2_world"], 4).tolist())
    print(f"COM distance: {best['com_distance_m']*1000:.1f} mm")
    print(f"other-object clearance: {best['other_object_clearance_m']*1000:.1f} mm")

    # Phase 1: open and move above the chosen gp.
    _command_fingers(franka_robot, min(cfg.max_opening_m, 0.080))
    ok = _move_arm_to_pose(franka_robot, pre, wrist_yaw, dt, cfg.move_steps, cfg.realtime_scale, min(cfg.max_opening_m, 0.080))
    if not ok:
        return {"success": False, "reason": "pregrasp_ik_failed", "best": _json_candidate(best)}

    # Phase 2: descend to grasp midpoint while keeping fingers open.
    ok = _move_arm_to_pose(franka_robot, gp, wrist_yaw, dt, cfg.approach_steps, cfg.realtime_scale, min(cfg.max_opening_m, 0.080))
    if not ok:
        return {"success": False, "reason": "grasp_pose_ik_failed", "best": _json_candidate(best)}

    # Phase 3: close parallel jaws.  Because the push architecture disables Panda
    # collisions, successful object retention is represented by a fixed grasp
    # constraint created only after the hand reaches the selected candidate.
    target_close = max(0.0, min(float(best["required_opening_m"]) * 0.45, 0.025))
    for i in range(max(cfg.close_steps, 1)):
        alpha = (i + 1) / max(cfg.close_steps, 1)
        opening = (1.0 - alpha) * min(cfg.max_opening_m, 0.080) + alpha * target_close
        _command_fingers(franka_robot, opening, force=70.0)
        p.stepSimulation()
        if cfg.realtime_scale > 0:
            time.sleep(dt * cfg.realtime_scale)

    # Ensure the target has not been displaced far away before attachment.
    now_pos, _ = p.getBasePositionAndOrientation(body_id)
    now_pos = np.asarray(now_pos, dtype=np.float64)
    if np.linalg.norm(now_pos[:2] - gp[:2]) > 0.14:
        return {"success": False, "reason": "target_not_under_gripper_after_close", "best": _json_candidate(best)}

    constraint_id = None
    if cfg.use_fixed_constraint_after_close:
        constraint_id = _attach_target_to_gripper(franka_robot, body_id, cfg.grasp_constraint_max_force_n)

    # Phase 4: lift straight up and hold.
    ok = _move_arm_to_pose(franka_robot, lift, wrist_yaw, dt, cfg.lift_steps, cfg.realtime_scale, target_close)
    if not ok:
        if constraint_id is not None:
            try:
                p.removeConstraint(constraint_id)
            except Exception:
                pass
        return {"success": False, "reason": "lift_ik_failed", "best": _json_candidate(best)}

    _step_sim(cfg.settle_steps, dt, cfg.realtime_scale)
    after_pos, after_q = p.getBasePositionAndOrientation(body_id)
    after_pos = np.asarray(after_pos, dtype=np.float64)
    lift_delta = float(after_pos[2] - before_pos[2])
    success = lift_delta > 0.045

    result = {
        "success": bool(success),
        "reason": "lifted" if success else "insufficient_lift",
        "object_number": int(target_object["object_number"]),
        "object_label": str(target_object["object_label"]),
        "body_id": body_id,
        "grasp": _json_candidate(best),
        "constraint_id": int(constraint_id) if constraint_id is not None else None,
        "object_before_world": before_pos.tolist(),
        "object_after_world": after_pos.tolist(),
        "object_after_quaternion": list(after_q),
        "vertical_lift_m": lift_delta,
    }
    with open(OUTPUT_DIR / "selected_object_grasp_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("GRASP RESULT:", result)
    return result
