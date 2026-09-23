from __future__ import annotations
MAIN_BUILD_ID = "2026-09-22_PUSH_GRASP_ENGINE_V1"

import json
import math
import inspect
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import pybullet as p
import pybullet_data

from perception import (
    load_mrformer,
    run_perception,
    build_object_catalog,
    gemini_push_advice,
    apply_object_semantics_to_records,
    estimate_whole_object_inertia_from_primitives,
)
import push as push_module
from push import (
    PushModelRuntime,
    FrankaPandaRobot,
    rank_primitives_for_goal,
    score_candidates,
    choose_one_shot_if_available,
    choose_monotonic_correction,
    choose_yaw_correction_candidate,
    rmppi_plan,
    prepare_executable_push,
    execute_push_pybullet,
    advance_hidden,
    retarget_candidate_to_goal,
)
from force_feedback import ForceFeedbackConfig, compute_pybullet_body_mass_com_inertia
from grasp import (
    GraspEngineConfig,
    build_grasp_library,
    print_grasp_library,
    draw_grasp_library,
    clear_grasp_debug,
    execute_selected_grasp,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
RAW_DIR = OUTPUT_DIR / "raw_pybullet"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# PYBULLET SCENE -- EIGHT PURPOSEFUL OBJECTS
# =============================================================================

USE_GUI = True
DT = 1.0 / 240.0
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
NEAR_PLANE = 0.10
FAR_PLANE = 3.0
FOV = 40.0

TABLE_HALF_X = 0.70
TABLE_HALF_Y = 0.52
TABLE_HALF_Z = 0.025
TABLE_CENTER_Z = 0.50
TABLE_TOP_Z = TABLE_CENTER_Z + TABLE_HALF_Z

PILE_CENTER_X = 0.02
PILE_CENTER_Y = 0.02
PUSH_FRAME_ORIGIN_XY = np.array([PILE_CENTER_X, PILE_CENTER_Y], dtype=np.float32)

# Overhead RGB-D. The actual 3-D Franka Panda is temporarily hidden during capture,
# so MR-Former/Gemini see ONLY the eight scene objects.
CAMERA_EYE = [PILE_CENTER_X + 0.08, PILE_CENTER_Y - 0.10, 1.65]
CAMERA_TARGET = [PILE_CENTER_X, PILE_CENTER_Y, 0.57]
CAMERA_UP = [0.0, 1.0, 0.0]

GUI_CAMERA_DISTANCE = 1.85
GUI_CAMERA_YAW = 40.0
GUI_CAMERA_PITCH = -48.0
GUI_CAMERA_TARGET = [-0.12, PILE_CENTER_Y, 0.72]

MAX_CLOSED_LOOP_PUSHES = 10
PUSHER_RADIUS = 0.008
POSITION_TOLERANCE_M = 0.015
YAW_TOLERANCE_DEG = 1.0
# Position-first / yaw-second controller. Once XY is close enough, translation
# goal tracking is suspended and a signed-torque yaw push is used.
YAW_PHASE_POSITION_GATE_M = 0.022
YAW_CORRECTION_MAX_XY_DRIFT_M = 0.018
YAW_CORRECTION_EFFECTIVE_MIN_DEG = 0.35
# Fine-yaw mode is entered near the requested angle.  It uses short, low-force
# tangential pulses and a tighter stop band so the object is not considered
# finished with the old +/-5 deg residual error.
YAW_FINE_BAND_DEG = 8.0
YAW_FINE_FORCE_SCALE = 0.50
GOAL_EDGE_MARGIN_M = 0.035

# First push should do most of the translational work.  Later pushes are deliberately
# shorter correction pushes.  The learned model still selects primitive/family/
# direction; this layer only scales the executed travel after the selected action
# has passed the network + geometric planning stages.
FIRST_PUSH_MIN_LENGTH_M = 0.140
FIRST_PUSH_MAX_LENGTH_M = 0.280
FIRST_PUSH_GOAL_FRACTION = 1.55
CORRECTION_MIN_LENGTH_M = 0.012
CORRECTION_MAX_LENGTH_M = 0.075
CORRECTION_GOAL_FRACTION = 1.05
MAX_CONSECUTIVE_EXECUTION_FAILURES = 4

# =============================================================================
# PAPER-INSPIRED PUSH FORCE FEEDBACK
# =============================================================================
# The attached paper used 13 N on a 260 g bushing in a different in-hand setup.
# For these small tabletop PyBullet objects we start lower.  Increase gradually
# only if the object does not move despite stable contact.
FORCE_FEEDBACK = ForceFeedbackConfig(
    trigger_force_n=0.8,
    desired_force_n=6.0,
    hard_force_limit_n=14.0,
    force_filter_alpha=0.25,
    kp_speed=0.0060,
    ki_speed=0.0008,
    kd_speed=0.00010,
    min_push_speed_mps=0.004,
    max_push_speed_mps=0.095,
    smoothing_time_s=0.20,
    smoothing_backoff_m=0.004,
    save_logs=True,
)

# =============================================================================
# PRIMITIVE GRASP ENGINE
# =============================================================================
GRASP_CONFIG = GraspEngineConfig(
    max_opening_m=0.080,
    min_opening_m=0.006,
    preferred_clearance_m=0.060,
    minimum_other_object_clearance_m=0.010,
    pregrasp_height_m=0.105,
    lift_height_m=0.145,
)


# =============================================================================
# BASIC HELPERS
# =============================================================================

def require_connected():
    if not p.isConnected():
        raise RuntimeError(
            "PyBullet physics server is not connected. Keep the PyBullet GUI open "
            "until the program prints the final result."
        )


def wrap_angle(a: float) -> float:
    return float(math.atan2(math.sin(a), math.cos(a)))


def rot2(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def body_pose_world(body_id: int):
    require_connected()
    pos, q = p.getBasePositionAndOrientation(int(body_id))
    yaw = p.getEulerFromQuaternion(q)[2]
    return np.asarray(pos, dtype=np.float32), float(yaw)


def world_state_to_push_frame(state_xyztheta: np.ndarray) -> np.ndarray:
    s = np.asarray(state_xyztheta, dtype=np.float32).copy()
    s[:2] -= PUSH_FRAME_ORIGIN_XY
    return s


def record_world_to_push_frame(record: dict) -> dict:
    out = dict(record)
    state = np.asarray(record["state_vector"], dtype=np.float32).copy()
    state[:2] -= PUSH_FRAME_ORIGIN_XY
    out["state_vector"] = state
    center = np.asarray(record["center_world"], dtype=np.float32).copy()
    center[:2] -= PUSH_FRAME_ORIGIN_XY
    out["center_world"] = center
    return out


def candidate_push_to_world(candidate: dict) -> dict:
    out = dict(candidate)
    c = np.asarray(candidate["contact_world"], dtype=np.float32).copy()
    c[:2] += PUSH_FRAME_ORIGIN_XY
    out["contact_world"] = c
    return out




def candidate_world_to_push_frame(candidate: dict) -> dict:
    """Convert an actually executed WORLD candidate back to the model push frame."""
    out = dict(candidate)
    c = np.asarray(candidate["contact_world"], dtype=np.float32).copy()
    c[:2] -= PUSH_FRAME_ORIGIN_XY
    out["contact_world"] = c
    if "execution_tool_contact_world" in out:
        tc = np.asarray(out["execution_tool_contact_world"], dtype=np.float32).copy()
        tc[:2] -= PUSH_FRAME_ORIGIN_XY
        out["execution_tool_contact_world"] = tc
    if "execution_start_world" in out:
        st = np.asarray(out["execution_start_world"], dtype=np.float32).copy()
        st[:2] -= PUSH_FRAME_ORIGIN_XY
        out["execution_start_world"] = st
    if "execution_end_world" in out:
        en = np.asarray(out["execution_end_world"], dtype=np.float32).copy()
        en[:2] -= PUSH_FRAME_ORIGIN_XY
        out["execution_end_world"] = en
    return out

def actual_object_error(body_id: int, goal_world: np.ndarray):
    pos, yaw = body_pose_world(body_id)
    dx = float(pos[0] - goal_world[0])
    dy = float(pos[1] - goal_world[1])
    dtheta = wrap_angle(yaw - float(goal_world[2]))
    return {
        "actual_pose_world": [float(pos[0]), float(pos[1]), float(yaw)],
        "dx_m": dx,
        "dy_m": dy,
        "position_error_m": float(math.hypot(dx, dy)),
        "yaw_error_rad": dtheta,
        "yaw_error_deg": float(math.degrees(dtheta)),
    }


def goal_reached(body_id: int, goal_world: np.ndarray):
    e = actual_object_error(body_id, goal_world)
    return e["position_error_m"] <= POSITION_TOLERANCE_M and abs(e["yaw_error_deg"]) <= YAW_TOLERANCE_DEG



def _candidate_direction_xy(candidate: dict) -> np.ndarray:
    d = np.asarray(candidate.get("direction_world", [0.0, 0.0, 0.0]), dtype=np.float64)[:2]
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return np.zeros(2, dtype=np.float64)
    return d / n


def choose_macro_first_candidate(step9: dict, current_object_xy: np.ndarray, goal_world: np.ndarray):
    """Choose a strong FIRST push only from network-ranked candidates.

    We do not replace the trained model.  We use its top candidates, then prefer a
    candidate whose commanded direction points toward the desired translation and
    whose primitive family already received a good Step-9 score.  This prevents a
    far-away goal from starting with a tiny correction-style action.
    """
    current_xy = np.asarray(current_object_xy, dtype=np.float64)[:2]
    goal_xy = np.asarray(goal_world, dtype=np.float64)[:2]
    delta = goal_xy - current_xy
    dist = float(np.linalg.norm(delta))
    if dist < 0.045:
        return None
    u_goal = delta / max(dist, 1e-12)
    top = [int(i) for i in np.asarray(step9["top_indices"]).reshape(-1)[:16]]
    if not top:
        return None
    raw_cost = np.asarray(step9["cost"], dtype=np.float64)
    c0 = float(np.min(raw_cost[top]))
    span = max(float(np.max(raw_cost[top]) - c0), 1e-6)
    best = None
    best_score = float("inf")
    for idx in top:
        cand = step9["candidates"][idx]
        d = _candidate_direction_xy(cand)
        if np.linalg.norm(d) < 1e-8:
            continue
        alignment = float(np.dot(d, u_goal))
        if alignment < 0.35:
            continue
        normalized_network_cost = (float(raw_cost[idx]) - c0) / span
        length_bonus = float(cand.get("push_length", 0.0)) / 0.08
        # network ranking is primary; goal alignment breaks ties in favor of a
        # decisive first translational push.
        score = normalized_network_cost - 1.85 * alignment - 0.35 * length_bonus
        if score < best_score:
            best_score = score
            best = {
                "candidate_index": idx,
                "candidate": cand,
                "predicted_next": np.asarray(step9["pred_next"][idx], dtype=np.float32),
                "alignment": alignment,
            }
    return best


def tune_execution_candidate(
    candidate: dict,
    step_index: int,
    object_error: dict,
    current_object_xy: np.ndarray,
    goal_world: np.ndarray,
    stagnation_count: int = 0,
    execution_gain_estimate=None,
) -> dict:
    """Scale physical travel while preserving planner primitive/family/direction.

    First push = decisive macro push. Later pushes = calibrated corrections.
    The correction scale is adapted from the measured ratio of object displacement
    to pusher travel from previous effective pushes.
    """
    out = dict(candidate)
    pos_err = float(object_error["position_error_m"])
    yaw_err = abs(float(object_error["yaw_error_deg"]))
    d = _candidate_direction_xy(out)
    delta = np.asarray(goal_world, dtype=np.float64)[:2] - np.asarray(current_object_xy, dtype=np.float64)[:2]
    delta_n = float(np.linalg.norm(delta))
    alignment = 0.0 if delta_n < 1e-9 else float(np.dot(d, delta / delta_n))
    base_length = float(out.get("push_length", 0.04))
    base_speed = float(out.get("push_speed", 0.04))

    if int(step_index) == 0 and pos_err > 0.040 and alignment > 0.10:
        desired = FIRST_PUSH_GOAL_FRACTION * pos_err
        desired = float(np.clip(desired, FIRST_PUSH_MIN_LENGTH_M, FIRST_PUSH_MAX_LENGTH_M))
        out["push_length"] = max(base_length, desired)
        out["push_speed"] = max(base_speed, 0.070)
        out["execution_phase"] = "MACRO_FIRST_PUSH"
    else:
        gain = 0.60 if execution_gain_estimate is None else float(np.clip(execution_gain_estimate, 0.25, 1.15))
        desired = CORRECTION_GOAL_FRACTION * pos_err / max(gain, 0.25)
        desired = float(np.clip(desired, CORRECTION_MIN_LENGTH_M, CORRECTION_MAX_LENGTH_M))
        if stagnation_count > 0:
            desired = min(0.058, desired * (1.10 + 0.08 * min(stagnation_count, 2)))
        if pos_err < 0.025 and yaw_err > 7.0:
            desired = min(desired, 0.030)
        if alignment > 0.05 or pos_err < 0.025:
            out["push_length"] = desired
        else:
            out["push_length"] = min(base_length, desired)
        out["push_speed"] = max(base_speed, 0.050)
        out["execution_phase"] = "CALIBRATED_CORRECTION"

    out["goal_alignment"] = float(alignment)
    out["execution_gain_used"] = None if execution_gain_estimate is None else float(execution_gain_estimate)
    return out

# =============================================================================
# DYNAMIC MULTIBODY COM
# =============================================================================

def compute_multibody_com(body_id: int):
    """Mass-weighted COM of the whole rigid multibody in world coordinates."""
    require_connected()
    weighted = np.zeros(3, dtype=np.float64)
    total = 0.0

    base_mass = float(p.getDynamicsInfo(body_id, -1)[0])
    base_pos = np.asarray(p.getBasePositionAndOrientation(body_id)[0], dtype=np.float64)
    if base_mass > 0:
        weighted += base_mass * base_pos
        total += base_mass

    for link in range(p.getNumJoints(body_id)):
        mass = float(p.getDynamicsInfo(body_id, link)[0])
        if mass <= 0:
            continue
        state = p.getLinkState(body_id, link, computeForwardKinematics=True)
        com_pos = np.asarray(state[0], dtype=np.float64)
        weighted += mass * com_pos
        total += mass

    if total <= 1e-12:
        return base_pos.astype(np.float32)
    return (weighted / total).astype(np.float32)


# =============================================================================
# PROCEDURAL OBJECT BUILDERS
# =============================================================================

def _box_shape(half, color):
    c = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(half))
    v = p.createVisualShape(p.GEOM_BOX, halfExtents=list(half), rgbaColor=list(color))
    return c, v


def _cylinder_shape(radius, height, color):
    c = p.createCollisionShape(p.GEOM_CYLINDER, radius=float(radius), height=float(height))
    v = p.createVisualShape(p.GEOM_CYLINDER, radius=float(radius), length=float(height), rgbaColor=list(color))
    return c, v


def _sphere_shape(radius, color):
    c = p.createCollisionShape(p.GEOM_SPHERE, radius=float(radius))
    v = p.createVisualShape(p.GEOM_SPHERE, radius=float(radius), rgbaColor=list(color))
    return c, v


def _create_fixed_multibody(
    base_mass,
    base_collision,
    base_visual,
    base_position,
    base_orientation,
    links,
):
    if not links:
        return p.createMultiBody(
            baseMass=float(base_mass),
            baseCollisionShapeIndex=base_collision,
            baseVisualShapeIndex=base_visual,
            basePosition=list(base_position),
            baseOrientation=list(base_orientation),
        )

    return p.createMultiBody(
        baseMass=float(base_mass),
        baseCollisionShapeIndex=base_collision,
        baseVisualShapeIndex=base_visual,
        basePosition=list(base_position),
        baseOrientation=list(base_orientation),
        linkMasses=[float(x["mass"]) for x in links],
        linkCollisionShapeIndices=[int(x["collision"]) for x in links],
        linkVisualShapeIndices=[int(x["visual"]) for x in links],
        linkPositions=[list(x["position"]) for x in links],
        linkOrientations=[list(x["orientation"]) for x in links],
        linkInertialFramePositions=[[0, 0, 0] for _ in links],
        linkInertialFrameOrientations=[[0, 0, 0, 1] for _ in links],
        linkParentIndices=[0 for _ in links],
        linkJointTypes=[p.JOINT_FIXED for _ in links],
        linkJointAxis=[[0, 0, 1] for _ in links],
    )


def create_cup(position, yaw):
    color = [0.90, 0.22, 0.20, 1.0]
    outer_r, wall_t, wall_h, bottom_h = 0.045, 0.006, 0.090, 0.008
    base_c, base_v = _cylinder_shape(outer_r, bottom_h, color)
    links = []
    n = 16
    arc = 2.0 * math.pi * (outer_r - wall_t / 2.0) / n * 1.08
    wall_c, wall_v = _box_shape([arc / 2.0, wall_t / 2.0, wall_h / 2.0], color)
    for i in range(n):
        phi = 2.0 * math.pi * i / n
        r = outer_r - wall_t / 2.0
        links.append({
            "mass": 0.012,
            "collision": wall_c,
            "visual": wall_v,
            "position": [r * math.cos(phi), r * math.sin(phi), bottom_h / 2.0 + wall_h / 2.0],
            "orientation": p.getQuaternionFromEuler([0, 0, phi + math.pi / 2.0]),
        })
    # Visible handle made from small rigid spheres.
    hs_c, hs_v = _sphere_shape(0.008, color)
    handle_center_x = outer_r + 0.006
    for a in np.linspace(-math.pi / 2.0, math.pi / 2.0, 9):
        links.append({
            "mass": 0.004,
            "collision": hs_c,
            "visual": hs_v,
            "position": [handle_center_x + 0.030 * math.cos(a), 0.0, 0.052 + 0.030 * math.sin(a)],
            "orientation": [0, 0, 0, 1],
        })
    q = p.getQuaternionFromEuler([0, 0, yaw])
    return _create_fixed_multibody(0.09, base_c, base_v, position, q, links)


def create_bottle(position, yaw):
    color = [0.15, 0.48, 0.92, 1.0]
    base_c, base_v = _cylinder_shape(0.035, 0.105, color)
    neck_c, neck_v = _cylinder_shape(0.018, 0.050, color)
    cap_c, cap_v = _cylinder_shape(0.020, 0.014, [0.08, 0.20, 0.45, 1.0])
    links = [
        {"mass": 0.05, "collision": neck_c, "visual": neck_v, "position": [0, 0, 0.0775], "orientation": [0, 0, 0, 1]},
        {"mass": 0.015, "collision": cap_c, "visual": cap_v, "position": [0, 0, 0.1095], "orientation": [0, 0, 0, 1]},
    ]
    q = p.getQuaternionFromEuler([0, 0, yaw])
    return _create_fixed_multibody(0.24, base_c, base_v, position, q, links)


def create_sphere(position):
    c, v = _sphere_shape(0.043, [0.95, 0.80, 0.12, 1.0])
    return _create_fixed_multibody(0.18, c, v, position, [0, 0, 0, 1], [])


def create_tape(position, yaw):
    color = [0.18, 0.75, 0.34, 1.0]
    Rout, Rin, height, n = 0.052, 0.027, 0.018, 24
    rmid = 0.5 * (Rout + Rin)
    radial_t = Rout - Rin
    arc = 2.0 * math.pi * rmid / n * 1.08
    seg_c, seg_v = _box_shape([arc / 2.0, radial_t / 2.0, height / 2.0], color)
    links = []
    for i in range(1, n):
        phi = 2.0 * math.pi * i / n
        links.append({
            "mass": 0.006,
            "collision": seg_c,
            "visual": seg_v,
            "position": [rmid * math.cos(phi), rmid * math.sin(phi), 0.0],
            "orientation": p.getQuaternionFromEuler([0, 0, phi + math.pi / 2.0]),
        })
    # First segment is the base collision and is offset from the reference origin.
    # We use a tiny central base and put all ring segments on fixed links so the hole is real.
    tiny_c, tiny_v = _sphere_shape(0.002, [0.18, 0.75, 0.34, 0.0])
    phi0 = 0.0
    links.insert(0, {
        "mass": 0.006,
        "collision": seg_c,
        "visual": seg_v,
        "position": [rmid, 0.0, 0.0],
        "orientation": p.getQuaternionFromEuler([0, 0, math.pi / 2.0]),
    })
    q = p.getQuaternionFromEuler([0, 0, yaw])
    return _create_fixed_multibody(0.003, tiny_c, tiny_v, position, q, links)


def create_cylinder(position, yaw):
    c, v = _cylinder_shape(0.037, 0.105, [0.62, 0.25, 0.82, 1.0])
    q = p.getQuaternionFromEuler([0, 0, yaw])
    return _create_fixed_multibody(0.26, c, v, position, q, [])


def create_cone_like(position, yaw, lying=True):
    color = [0.96, 0.48, 0.10, 1.0]
    total_h = 0.120
    n = 6
    slice_h = total_h / n
    radii = np.linspace(0.046, 0.010, n)
    base_c, base_v = _cylinder_shape(float(radii[0]), slice_h, color)
    links = []
    for i in range(1, n):
        c, v = _cylinder_shape(float(radii[i]), slice_h, color)
        links.append({
            "mass": 0.025,
            "collision": c,
            "visual": v,
            "position": [0, 0, i * slice_h],
            "orientation": [0, 0, 0, 1],
        })
    euler = [math.pi / 2.0 if lying else 0.0, 0.0, yaw]
    q = p.getQuaternionFromEuler(euler)
    return _create_fixed_multibody(0.05, base_c, base_v, position, q, links)



def create_banana(position, yaw):
    """Create one curved, rigid banana-like object without external assets.

    The banana is represented by seven short cylindrical segments arranged on
    an in-plane arc.  PyBullet treats all segments as one rigid multibody,
    while MR-Former is still responsible for assigning primitive classes.
    """
    color = [0.98, 0.82, 0.08, 1.0]
    dark_tip = [0.45, 0.27, 0.06, 1.0]
    seg_radius = 0.014
    seg_length = 0.031
    n = 7
    arc_radius = 0.090
    phis = np.linspace(-0.72, +0.72, n)

    # Tiny transparent base at the object's reference origin.  All visible
    # banana segments are fixed links, so the complete banana is one body.
    base_c, base_v = _sphere_shape(0.002, [1.0, 0.82, 0.08, 0.0])
    seg_c, seg_v = _cylinder_shape(seg_radius, seg_length, color)
    tip_c, tip_v = _sphere_shape(0.015, dark_tip)
    links = []

    for phi in phis:
        # Arc centered so phi=0 is the banana's local origin.
        x = arc_radius * math.sin(float(phi))
        y = arc_radius * (1.0 - math.cos(float(phi)))
        tangent_yaw = float(phi)
        q_seg = p.getQuaternionFromEuler([0.0, math.pi / 2.0, tangent_yaw])
        links.append({
            "mass": 0.022,
            "collision": seg_c,
            "visual": seg_v,
            "position": [x, y, 0.0],
            "orientation": q_seg,
        })

    # Small darker end caps make the object visually banana-like while staying
    # one rigid body.
    for phi in (float(phis[0]), float(phis[-1])):
        x = arc_radius * math.sin(phi)
        y = arc_radius * (1.0 - math.cos(phi))
        links.append({
            "mass": 0.006,
            "collision": tip_c,
            "visual": tip_v,
            "position": [x, y, 0.0],
            "orientation": [0, 0, 0, 1],
        })

    q = p.getQuaternionFromEuler([0.0, 0.0, yaw])
    return _create_fixed_multibody(0.004, base_c, base_v, position, q, links)


def create_food_carton(position, yaw):
    """Create a single rigid food-carton / cereal-box-like cuboid."""
    color = [0.88, 0.58, 0.20, 1.0]
    c, v = _box_shape([0.036, 0.027, 0.060], color)
    q = p.getQuaternionFromEuler([0.0, 0.0, yaw])
    return _create_fixed_multibody(0.24, c, v, position, q, [])

def create_scene():
    require_connected()
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(DT)
    p.resetDebugVisualizerCamera(
        cameraDistance=GUI_CAMERA_DISTANCE,
        cameraYaw=GUI_CAMERA_YAW,
        cameraPitch=GUI_CAMERA_PITCH,
        cameraTargetPosition=GUI_CAMERA_TARGET,
    )

    table_c = p.createCollisionShape(p.GEOM_BOX, halfExtents=[TABLE_HALF_X, TABLE_HALF_Y, TABLE_HALF_Z])
    table_v = p.createVisualShape(p.GEOM_BOX, halfExtents=[TABLE_HALF_X, TABLE_HALF_Y, TABLE_HALF_Z], rgbaColor=[0.92, 0.92, 0.92, 1.0])
    table_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=table_c, baseVisualShapeIndex=table_v, basePosition=[0, 0, TABLE_CENTER_Z])
    plane_id = p.loadURDF("plane.urdf")

    # Eight purposeful objects. Positions are close enough to form clutter but remain readable.
    cup = create_cup([-0.135, +0.090, TABLE_TOP_Z + 0.004], math.radians(+18))
    bottle = create_bottle([-0.005, +0.095, TABLE_TOP_Z + 0.0525], math.radians(-12))
    sphere = create_sphere([+0.125, +0.095, TABLE_TOP_Z + 0.043])
    tape = create_tape([-0.115, -0.075, TABLE_TOP_Z + 0.011], math.radians(+28))
    cylinder = create_cylinder([+0.025, -0.070, TABLE_TOP_Z + 0.0525], math.radians(+8))
    # Lying cone-like object: supports rolling/offset rotational pushes.
    cone = create_cone_like([+0.155, -0.060, TABLE_TOP_Z + 0.052], math.radians(-25), lying=True)

    # Added requested objects.  Both are deliberately placed inside the same
    # clutter workspace but with enough separation for clean initial RGB-D masks.
    banana = create_banana([-0.285, +0.005, TABLE_TOP_Z + 0.018], math.radians(-18))
    food_carton = create_food_carton([+0.275, +0.005, TABLE_TOP_Z + 0.060], math.radians(+14))

    object_ids = [cup, bottle, sphere, tape, cylinder, cone, banana, food_carton]

    # Keep execution physics inside the domain used by the randomized training
    # set and make contact motion clearly visible. These are not network inputs.
    p.changeDynamics(table_id, -1, lateralFriction=0.62, restitution=0.0)
    for _bid in object_ids:
        for _link in [-1] + list(range(p.getNumJoints(int(_bid)))):
            try:
                p.changeDynamics(
                    int(_bid), int(_link),
                    lateralFriction=0.42,
                    rollingFriction=0.0015,
                    spinningFriction=0.0015,
                    restitution=0.0,
                    linearDamping=0.035,
                    angularDamping=0.035,
                )
            except Exception:
                pass

    names = {
        cup: "CUP",
        bottle: "BOTTLE",
        sphere: "SPHERICAL_OBJECT",
        tape: "TAPE",
        cylinder: "CYLINDRICAL_OBJECT",
        cone: "CONE_LIKE_OBJECT",
        banana: "BANANA",
        food_carton: "FOOD_CARTON",
    }
    body_metadata = {
        cup: {"semantic_name": "cup", "expected_hollow": True},
        bottle: {"semantic_name": "bottle", "expected_hollow": False},
        sphere: {"semantic_name": "spherical object", "expected_hollow": False},
        tape: {"semantic_name": "tape", "expected_hollow": True},
        cylinder: {"semantic_name": "cylindrical object", "expected_hollow": False},
        cone: {"semantic_name": "cone-like object", "expected_hollow": False},
        banana: {"semantic_name": "banana", "expected_hollow": False},
        food_carton: {"semantic_name": "food carton", "expected_hollow": False},
    }

    # Moderate settling only; no huge randomized pile.
    for _ in range(500):
        p.stepSimulation()
        if USE_GUI:
            time.sleep(DT * 0.08)

    print("Created purposeful eight-object clutter scene:")
    for i, bid in enumerate(object_ids, start=1):
        print(f"  scene body {i}: {names[bid]} (body_id={bid})")

    return {
        "table_id": int(table_id),
        "plane_id": int(plane_id),
        "object_ids": [int(x) for x in object_ids],
        "body_id_to_name": {int(k): v for k, v in names.items()},
        "body_metadata": {int(k): v for k, v in body_metadata.items()},
    }


# =============================================================================
# CAMERA
# =============================================================================

def camera_matrices():
    view = p.computeViewMatrix(CAMERA_EYE, CAMERA_TARGET, CAMERA_UP)
    proj = p.computeProjectionMatrixFOV(FOV, IMAGE_WIDTH / IMAGE_HEIGHT, NEAR_PLANE, FAR_PLANE)
    return view, proj


VIEW_MATRIX, PROJECTION_MATRIX = camera_matrices()


def capture_rgbd(franka_robot=None):
    """Capture RGB-D with the Franka Panda hidden from the camera image.

    The Franka Panda remains present and visible in the interactive PyBullet GUI, but it
    is made transparent only for the instant of RGB-D rendering. This preserves
    the user's original requirement that the perception camera sees the objects,
    not the execution robot.
    """
    require_connected()
    if franka_robot is not None:
        franka_robot.set_visible(False)
    try:
        out = p.getCameraImage(
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            viewMatrix=VIEW_MATRIX,
            projectionMatrix=PROJECTION_MATRIX,
            renderer=(p.ER_BULLET_HARDWARE_OPENGL if USE_GUI else p.ER_TINY_RENDERER),
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            shadow=1,
        )
    finally:
        if franka_robot is not None and p.isConnected():
            franka_robot.set_visible(True)

    w, h = int(out[0]), int(out[1])
    rgb = np.asarray(out[2], dtype=np.uint8).reshape(h, w, 4)[..., :3].copy()
    depth_buffer = np.asarray(out[3], dtype=np.float32).reshape(h, w)
    seg = np.asarray(out[4], dtype=np.int32).reshape(h, w)
    depth_m = FAR_PLANE * NEAR_PLANE / (FAR_PLANE - (FAR_PLANE - NEAR_PLANE) * depth_buffer)
    return rgb, depth_m.astype(np.float32), seg


def save_raw_frame(rgb, depth_m, segmentation, frame_index):
    cv2.imwrite(str(RAW_DIR / f"rgb_{frame_index:03d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(RAW_DIR / f"depth_{frame_index:03d}_mm.png"), np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16))
    np.save(RAW_DIR / f"segmentation_{frame_index:03d}.npy", segmentation)


# =============================================================================
# REPORTS + USER INPUT
# =============================================================================

def get_body_pose_lookup(scene: dict):
    lookup = {}
    for bid in scene["object_ids"]:
        pos, yaw = body_pose_world(bid)
        lookup[int(bid)] = {
            "position_world": pos.tolist(),
            "yaw_world": float(yaw),
            "true_com_world": compute_multibody_com(bid).tolist(),
            "body_name": scene["body_id_to_name"][int(bid)],
        }
    return lookup


def print_camera_report(result: dict):
    print("\n" + "=" * 92)
    print("CAMERA COORDINATE SYSTEM")
    print("=" * 92)
    print("World: PyBullet world; table top Z =", TABLE_TOP_Z)
    print("Camera: +Xc image right, +Yc image down, +Zc optical forward")
    print("Camera eye WORLD:", CAMERA_EYE)
    print("Camera target WORLD:", CAMERA_TARGET)
    print("T_world_from_camera =")
    print(np.array2string(result["T_world_from_camera"], precision=5, suppress_small=True))
    print("T_camera_from_world =")
    print(np.array2string(result["T_camera_from_world"], precision=5, suppress_small=True))


def print_mrformer_results(result: dict):
    print("\n" + "=" * 110)
    print("MR-FORMER RESULTS -- EACH PYBULLET OBJECT WAS ISOLATED BEFORE RGB-D INFERENCE")
    print("=" * 110)
    for r in result["primitive_records"]:
        cw = np.asarray(r["center_world"], dtype=float)
        print(
            f"P{r['primitive_id']:02d} {r['primitive_type']:<10} | "
            f"body={r['body_name']:<20} | center W=({cw[0]:+.3f},{cw[1]:+.3f},{cw[2]:+.3f})"
        )


def augment_catalog_with_inertia(catalog: list, primitive_records: list):
    """Attach whole-object mass/COM/inertia to each catalog entry.

    Planning uses the perception-derived Izz: MR-Former primitive geometry + the
    whole-object COM + total mass + parallel-axis theorem.  Exact PyBullet Izz is
    kept only as simulation ground truth for evaluation.
    """
    by_body = {}
    for r in primitive_records:
        by_body.setdefault(int(r["body_id"]), []).append(r)

    for obj in catalog:
        bid = int(obj["body_id"])
        truth = compute_pybullet_body_mass_com_inertia(bid)
        mass = float(truth["mass_kg"])
        com = np.asarray(obj.get("gemini_com_world", [np.nan, np.nan, np.nan]), dtype=np.float64)
        if com.size < 3 or not np.all(np.isfinite(com[:3])):
            com = np.asarray(obj.get("true_com_world", truth["com_world"]), dtype=np.float64)
        est = estimate_whole_object_inertia_from_primitives(
            by_body.get(bid, []), com, total_mass_kg=max(mass, 1e-6)
        )
        obj["object_mass_kg"] = float(mass)
        obj["estimated_Izz_kgm2"] = float(est["Izz_kgm2"])
        obj["estimated_Izz_per_kg_m2"] = float(est["Izz_per_kg_m2"])
        obj["inertia_source"] = str(est["source"])
        obj["true_sim_Izz_kgm2"] = float(truth["Izz_kgm2"])
        obj["true_sim_inertia_world_kgm2"] = np.asarray(truth["inertia_world_kgm2"]).tolist()
    return catalog


def print_object_catalog(catalog: list, source: str):
    print("\n" + "=" * 145)
    print(f"OBJECT NUMBERS + INITIAL COORDINATES + WHOLE-OBJECT COM | grouping source={source}")
    print("=" * 145)
    print("OBJ | LABEL / BODY             | PRIMITIVES        | INITIAL WORLD (x,y,theta°) | GEMINI COM WORLD (x,y,z)       | TRUE SIM COM (x,y,z)")
    print("-" * 145)
    for obj in catalog:
        w = obj["initial_world_position"]
        gc = obj["gemini_com_world"]
        tc = obj["true_com_world"]
        plist = ",".join(f"P{x}" for x in obj["primitive_ids"])
        print(
            f"{obj['object_number']:>3d} | "
            f"{(obj['object_label'] + ' / ' + obj['body_name'])[:24]:<24} | "
            f"{plist[:16]:<16} | "
            f"({w[0]:+.3f},{w[1]:+.3f},{obj['initial_world_yaw_deg']:+7.2f}) | "
            f"({gc[0]:+.3f},{gc[1]:+.3f},{gc[2]:+.3f}) | "
            f"({tc[0]:+.3f},{tc[1]:+.3f},{tc[2]:+.3f})"
        )
        props = {f"P{p['primitive_id']}": (p.get("occupancy", "unknown"), bool(p.get("inner_accessible", False))) for p in obj.get("primitive_properties", [])}
        print("    hollow/solid:", props)
        if "estimated_Izz_kgm2" in obj:
            print(
                f"    mass={obj['object_mass_kg']:.4f} kg | "
                f"estimated Izz@whole-COM={obj['estimated_Izz_kgm2']:.6e} kg*m^2 | "
                f"true sim Izz={obj['true_sim_Izz_kgm2']:.6e} kg*m^2"
            )
    print("=" * 145)



# =============================================================================
# VISUAL INFORMATION: OBJECT NUMBERS, COORDINATES, COM, HOLLOW/SOLID
# =============================================================================

_OBJECT_LABEL_DEBUG_IDS = []
_WORLD_DEBUG_IDS = []
_GOAL_DEBUG_IDS = []


def draw_world_coordinate_frame():
    """Draw world X/Y axes and a safe target rectangle directly in PyBullet GUI."""
    require_connected()
    global _WORLD_DEBUG_IDS
    for uid in _WORLD_DEBUG_IDS:
        try:
            p.removeUserDebugItem(uid)
        except Exception:
            pass
    _WORLD_DEBUG_IDS = []
    z = TABLE_TOP_Z + 0.006
    _WORLD_DEBUG_IDS.append(p.addUserDebugLine([0, 0, z], [0.30, 0, z], [1, 0, 0], 4.0))
    _WORLD_DEBUG_IDS.append(p.addUserDebugLine([0, 0, z], [0, 0.30, z], [0, 0.75, 0], 4.0))
    _WORLD_DEBUG_IDS.append(p.addUserDebugText("+X WORLD", [0.31, 0, z + 0.015], [1, 0, 0], 1.2))
    _WORLD_DEBUG_IDS.append(p.addUserDebugText("+Y WORLD", [0, 0.31, z + 0.015], [0, 0.65, 0], 1.2))
    _WORLD_DEBUG_IDS.append(p.addUserDebugText("WORLD (0,0)", [0.015, 0.015, z + 0.015], [0.1, 0.1, 0.1], 1.1))

    xmin = -TABLE_HALF_X + GOAL_EDGE_MARGIN_M
    xmax = TABLE_HALF_X - GOAL_EDGE_MARGIN_M
    ymin = -TABLE_HALF_Y + GOAL_EDGE_MARGIN_M
    ymax = TABLE_HALF_Y - GOAL_EDGE_MARGIN_M
    corners = [[xmin, ymin, z], [xmax, ymin, z], [xmax, ymax, z], [xmin, ymax, z]]
    for i in range(4):
        _WORLD_DEBUG_IDS.append(
            p.addUserDebugLine(corners[i], corners[(i + 1) % 4], [0.95, 0.72, 0.08], 1.4)
        )
    _WORLD_DEBUG_IDS.append(
        p.addUserDebugText("TARGET INPUT AREA (approx.)", [xmin + 0.02, ymax - 0.02, z + 0.012], [0.7, 0.45, 0.0], 0.9)
    )


def label_objects_in_pybullet(catalog: list):
    """Put OBJ numbers and detected information directly above each object."""
    require_connected()
    global _OBJECT_LABEL_DEBUG_IDS
    for uid in _OBJECT_LABEL_DEBUG_IDS:
        try:
            p.removeUserDebugItem(uid)
        except Exception:
            pass
    _OBJECT_LABEL_DEBUG_IDS = []

    palette = [
        [0.85, 0.05, 0.05], [0.05, 0.25, 0.90], [0.85, 0.65, 0.05],
        [0.05, 0.65, 0.20], [0.60, 0.12, 0.75], [0.95, 0.35, 0.05],
    ]
    for i, obj in enumerate(catalog):
        pos = np.asarray(obj["initial_world_position"], dtype=float)
        plist = ",".join(f"P{x}" for x in obj["primitive_ids"])
        text = (
            f"OBJ {obj['object_number']}  {obj['object_label'].upper()}\n"
            f"{plist}   x={pos[0]:+.3f} y={pos[1]:+.3f} th={obj['initial_world_yaw_deg']:+.1f}deg"
        )
        uid = p.addUserDebugText(
            text,
            [float(pos[0]), float(pos[1]), float(pos[2] + 0.16)],
            textColorRGB=palette[i % len(palette)],
            textSize=1.05,
            lifeTime=0,
        )
        _OBJECT_LABEL_DEBUG_IDS.append(uid)

        com = np.asarray(obj["gemini_com_world"], dtype=float)
        if np.all(np.isfinite(com)):
            dz = 0.025
            _OBJECT_LABEL_DEBUG_IDS.append(
                p.addUserDebugLine(
                    [com[0] - 0.015, com[1], com[2] + dz],
                    [com[0] + 0.015, com[1], com[2] + dz],
                    [1.0, 0.0, 1.0], 3.0,
                )
            )
            _OBJECT_LABEL_DEBUG_IDS.append(
                p.addUserDebugLine(
                    [com[0], com[1] - 0.015, com[2] + dz],
                    [com[0], com[1] + 0.015, com[2] + dz],
                    [1.0, 0.0, 1.0], 3.0,
                )
            )


def _catalog_property_string(obj: dict) -> str:
    by_pid = {int(p.get("primitive_id", -1)): p for p in obj.get("primitive_properties", [])}
    parts = []
    for pid, ptype in zip(obj.get("primitive_ids", []), obj.get("primitive_types", [])):
        prop = by_pid.get(int(pid), {})
        occupancy = str(prop.get("occupancy", "unknown"))
        inner = "innerOK" if bool(prop.get("inner_accessible", False)) else "outer"
        parts.append(f"P{pid}:{ptype}/{occupancy}/{inner}")
    return " | ".join(parts)


def show_object_coordinate_board(catalog: list, grouping_source: str):
    """Show and save a top-down coordinate map plus the full detected object table."""
    H, W = 1320, 1580
    canvas = np.full((H, W, 3), 248, dtype=np.uint8)
    cv2.putText(canvas, "PRIMITIVE PUSH ENGINE - OBJECT NUMBERS + PYBULLET WORLD COORDINATES",
                (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.88, (10, 10, 10), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Grouping/COM source: {grouping_source}",
                (30, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (45, 45, 45), 1, cv2.LINE_AA)

    # ---------------- top-down map ----------------
    left, top, map_w, map_h = 40, 110, 710, 690
    cv2.rectangle(canvas, (left, top), (left + map_w, top + map_h), (30, 30, 30), 2)
    xmin, xmax = -TABLE_HALF_X, TABLE_HALF_X
    ymin, ymax = -TABLE_HALF_Y, TABLE_HALF_Y

    def w2p(x, y):
        px = int(left + (float(x) - xmin) / (xmax - xmin) * map_w)
        py = int(top + map_h - (float(y) - ymin) / (ymax - ymin) * map_h)
        return px, py

    # 10-cm grid.
    for x in np.arange(math.ceil(xmin * 10) / 10, xmax + 1e-9, 0.1):
        a = w2p(x, ymin); b = w2p(x, ymax)
        cv2.line(canvas, a, b, (220, 220, 220), 1)
        cv2.putText(canvas, f"{x:+.1f}", (a[0] - 18, top + map_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1, cv2.LINE_AA)
    for y in np.arange(math.ceil(ymin * 10) / 10, ymax + 1e-9, 0.1):
        a = w2p(xmin, y); b = w2p(xmax, y)
        cv2.line(canvas, a, b, (220, 220, 220), 1)
        cv2.putText(canvas, f"{y:+.1f}", (left - 38, a[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1, cv2.LINE_AA)

    # Axes.
    o = w2p(0, 0)
    xp = w2p(0.30, 0)
    yp = w2p(0, 0.30)
    cv2.arrowedLine(canvas, o, xp, (0, 0, 230), 3, tipLength=0.08)
    cv2.arrowedLine(canvas, o, yp, (0, 150, 0), 3, tipLength=0.08)
    cv2.putText(canvas, "+X", (xp[0] + 5, xp[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 230), 2)
    cv2.putText(canvas, "+Y", (yp[0] + 5, yp[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 150, 0), 2)

    # Safe target rectangle.
    sa = w2p(-TABLE_HALF_X + GOAL_EDGE_MARGIN_M, -TABLE_HALF_Y + GOAL_EDGE_MARGIN_M)
    sb = w2p(TABLE_HALF_X - GOAL_EDGE_MARGIN_M, TABLE_HALF_Y - GOAL_EDGE_MARGIN_M)
    x1, x2 = sorted([sa[0], sb[0]]); y1, y2 = sorted([sa[1], sb[1]])
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 180, 220), 2)

    palette = [
        (40, 40, 220), (220, 90, 30), (20, 180, 220), (30, 180, 60),
        (180, 50, 180), (30, 110, 245), (30, 190, 235), (95, 150, 220),
    ]
    for i, obj in enumerate(catalog):
        pos = obj["initial_world_position"]
        px, py = w2p(pos[0], pos[1])
        col = palette[i % len(palette)]
        cv2.circle(canvas, (px, py), 18, col, -1)
        cv2.circle(canvas, (px, py), 19, (20, 20, 20), 2)
        cv2.putText(canvas, str(obj["object_number"]), (px - 6, py + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        theta = math.radians(float(obj["initial_world_yaw_deg"]))
        tip = (int(px + 35 * math.cos(theta)), int(py - 35 * math.sin(theta)))
        cv2.arrowedLine(canvas, (px, py), tip, col, 3, tipLength=0.25)
        cv2.putText(canvas, obj["object_label"], (px + 22, py - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (20, 20, 20), 1, cv2.LINE_AA)

    cv2.putText(canvas, "Target input uses WORLD x,y coordinates on this map.",
                (45, 835), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Theta is tabletop yaw: 0 deg = +X, +90 deg = +Y.",
                (45, 866), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 1, cv2.LINE_AA)

    # ---------------- detailed table ----------------
    x0 = 790
    y = 120
    cv2.putText(canvas, "DETECTED OBJECT CATALOG", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (10, 10, 10), 2)
    y += 34
    for obj in catalog:
        pos = obj["initial_world_position"]
        comw = obj["gemini_com_world"]
        comc = obj["gemini_com_camera"]
        cv2.putText(canvas,
                    f"OBJ {obj['object_number']}  {obj['object_label'].upper()} / {obj['body_name']}",
                    (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2, cv2.LINE_AA)
        y += 24
        cv2.putText(canvas,
                    f"initial WORLD: x={pos[0]:+.3f}  y={pos[1]:+.3f}  theta={obj['initial_world_yaw_deg']:+.1f} deg",
                    (x0 + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (30, 30, 30), 1, cv2.LINE_AA)
        y += 22
        cv2.putText(canvas,
                    f"detected COM WORLD: ({comw[0]:+.3f}, {comw[1]:+.3f}, {comw[2]:+.3f})",
                    (x0 + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (80, 0, 110), 1, cv2.LINE_AA)
        y += 22
        cv2.putText(canvas,
                    f"detected COM CAMERA: ({comc[0]:+.3f}, {comc[1]:+.3f}, {comc[2]:+.3f})",
                    (x0 + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (80, 0, 110), 1, cv2.LINE_AA)
        y += 22
        if "estimated_Izz_kgm2" in obj:
            cv2.putText(canvas,
                        f"mass={obj['object_mass_kg']:.3f} kg  |  estimated Izz@COM={obj['estimated_Izz_kgm2']:.3e} kg*m^2",
                        (x0 + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 55, 10), 1, cv2.LINE_AA)
            y += 22
        prop = _catalog_property_string(obj)
        # Split long primitive description across two lines.
        chunks = [prop[j:j + 82] for j in range(0, len(prop), 82)] or ["no primitives"]
        for chunk in chunks[:2]:
            cv2.putText(canvas, chunk, (x0 + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (20, 70, 120), 1, cv2.LINE_AA)
            y += 20
        y += 12

    out_path = OUTPUT_DIR / "object_coordinate_map.png"
    cv2.imwrite(str(out_path), canvas)
    try:
        cv2.imshow("OBJECT NUMBERS + WORLD COORDINATES", canvas)
        cv2.waitKey(1)
    except Exception:
        pass
    print("\nSaved visual object/coordinate board:", out_path)
    return canvas


def selected_goal_bounds(body_id: int):
    lo, hi = body_union_aabb(int(body_id))
    half = 0.5 * (hi - lo)
    return {
        "xmin": float(-TABLE_HALF_X + half[0] + GOAL_EDGE_MARGIN_M),
        "xmax": float(TABLE_HALF_X - half[0] - GOAL_EDGE_MARGIN_M),
        "ymin": float(-TABLE_HALF_Y + half[1] + GOAL_EDGE_MARGIN_M),
        "ymax": float(TABLE_HALF_Y - half[1] - GOAL_EDGE_MARGIN_M),
    }


def draw_goal_marker(selected: dict, goal_world: np.ndarray):
    require_connected()
    global _GOAL_DEBUG_IDS
    for uid in _GOAL_DEBUG_IDS:
        try:
            p.removeUserDebugItem(uid)
        except Exception:
            pass
    _GOAL_DEBUG_IDS = []
    x, y, th = [float(v) for v in goal_world]
    z = TABLE_TOP_Z + 0.012
    s = 0.035
    _GOAL_DEBUG_IDS.append(p.addUserDebugLine([x - s, y, z], [x + s, y, z], [0.95, 0.0, 0.0], 4.0))
    _GOAL_DEBUG_IDS.append(p.addUserDebugLine([x, y - s, z], [x, y + s, z], [0.95, 0.0, 0.0], 4.0))
    tip = [x + 0.075 * math.cos(th), y + 0.075 * math.sin(th), z]
    _GOAL_DEBUG_IDS.append(p.addUserDebugLine([x, y, z], tip, [0.95, 0.0, 0.0], 5.0))
    _GOAL_DEBUG_IDS.append(
        p.addUserDebugText(
            f"GOAL OBJ {selected['object_number']}\nx={x:+.3f} y={y:+.3f} th={math.degrees(th):+.1f}deg",
            [x, y, z + 0.065], [0.90, 0.0, 0.0], 1.1, lifeTime=0
        )
    )

def _console_prompt_selected_object(catalog: list):
    valid = {int(x["object_number"]): x for x in catalog}
    while True:
        try:
            raw = input("\nChoose ONE entire object number to move: ").strip()
        except EOFError as exc:
            raise RuntimeError(
                "VS Code did not provide interactive stdin. The GUI input dialog also failed. "
                "Run with 'python main.py' in the integrated Terminal."
            ) from exc
        try:
            n = int(raw)
        except ValueError:
            print("Enter an integer object number from the table.")
            continue
        if n in valid:
            return valid[n]
        print("Object number is not in the table.")


def _console_prompt_goal_world():
    print("\nEnter desired pose for the ENTIRE selected object in PyBullet WORLD frame.")
    print("x,y in meters; theta is planar yaw in degrees.")
    while True:
        try:
            x = float(input("Desired x [m]: ").strip())
            y = float(input("Desired y [m]: ").strip())
            th = float(input("Desired theta [deg]: ").strip())
            return np.array([x, y, math.radians(th)], dtype=np.float32)
        except EOFError as exc:
            raise RuntimeError(
                "VS Code did not provide interactive stdin. Run with 'python main.py' in the integrated Terminal."
            ) from exc
        except ValueError:
            print("Invalid number. Try again.")


def prompt_selected_object_and_goal(catalog: list):
    """Always give the user a real input opportunity.

    On Windows/VS Code we use Tk dialogs first, so pressing the VS Code Run button
    cannot cause input() to immediately receive EOF and close PyBullet.  Console
    input remains as a fallback when Tk is unavailable.
    """
    valid = {int(x["object_number"]): x for x in catalog}
    valid_ids = sorted(valid)
    try:
        import tkinter as tk
        from tkinter import simpledialog, messagebox

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.update()

        selected = None
        while selected is None:
            n = simpledialog.askinteger(
                "Select object",
                "Choose ONE entire object number to move.\nValid object numbers: "
                + ", ".join(str(x) for x in valid_ids),
                parent=root,
            )
            if n is None:
                root.destroy()
                raise RuntimeError("Object selection was cancelled by the user.")
            if int(n) in valid:
                selected = valid[int(n)]
            else:
                messagebox.showerror(
                    "Invalid object",
                    f"Object {n} is not in the catalog. Valid: {valid_ids}",
                    parent=root,
                )

        def ask_float(title, prompt):
            while True:
                value = simpledialog.askfloat(title, prompt, parent=root)
                if value is None:
                    root.destroy()
                    raise RuntimeError("Goal entry was cancelled by the user.")
                if math.isfinite(float(value)):
                    return float(value)

        current = selected["initial_world_position"]
        bounds = selected_goal_bounds(int(selected["body_id"]))
        messagebox.showinfo(
            "Selected object coordinates",
            f"OBJ {selected['object_number']}  {selected['object_label']}\n\n"
            f"CURRENT WORLD pose:\n"
            f"x = {current[0]:+.4f} m\n"
            f"y = {current[1]:+.4f} m\n"
            f"theta = {selected['initial_world_yaw_deg']:+.2f} deg\n\n"
            f"Enter the DESIRED center/base pose in the SAME PyBullet WORLD frame.\n"
            f"Safe X approx: [{bounds['xmin']:+.3f}, {bounds['xmax']:+.3f}] m\n"
            f"Safe Y approx: [{bounds['ymin']:+.3f}, {bounds['ymax']:+.3f}] m\n"
            f"theta: -180 ... +180 deg\n\n"
            f"0 deg points along +X; +90 deg points along +Y.",
            parent=root,
        )

        while True:
            x = ask_float("Desired X", f"Desired X [m]\nSafe approx: {bounds['xmin']:+.3f} ... {bounds['xmax']:+.3f}")
            y = ask_float("Desired Y", f"Desired Y [m]\nSafe approx: {bounds['ymin']:+.3f} ... {bounds['ymax']:+.3f}")
            if bounds["xmin"] <= x <= bounds["xmax"] and bounds["ymin"] <= y <= bounds["ymax"]:
                break
            messagebox.showerror(
                "Goal outside safe table area",
                f"Entered x={x:+.3f}, y={y:+.3f}.\n"
                f"Use X in [{bounds['xmin']:+.3f},{bounds['xmax']:+.3f}] and "
                f"Y in [{bounds['ymin']:+.3f},{bounds['ymax']:+.3f}].",
                parent=root,
            )
        th_deg = ask_float("Desired theta", "Desired tabletop yaw theta [deg]\n0 deg = +X, +90 deg = +Y")
        root.destroy()
        goal = np.array([x, y, math.radians(th_deg)], dtype=np.float32)
        return selected, goal

    except RuntimeError:
        raise
    except Exception as exc:
        print(f"GUI input dialog unavailable ({type(exc).__name__}: {exc}). Falling back to terminal input.")
        selected = _console_prompt_selected_object(catalog)
        goal = _console_prompt_goal_world()
        return selected, goal



def _console_prompt_selected_object_and_action(catalog: list):
    valid = {int(x["object_number"]): x for x in catalog}
    while True:
        selected = _console_prompt_selected_object(catalog)
        try:
            raw = input("Action for this object -- GRASP or PUSH? [g/p]: ").strip().lower()
        except EOFError as exc:
            raise RuntimeError("Interactive stdin unavailable. Run with 'python main.py' in the VS Code integrated Terminal.") from exc
        if raw in {"g", "grasp"}:
            return selected, "grasp"
        if raw in {"p", "push"}:
            return selected, "push"
        print("Enter 'g'/'grasp' or 'p'/'push'.")


def prompt_selected_object_and_action(catalog: list):
    """Ask first WHICH entire object, then whether to GRASP or PUSH it."""
    valid = {int(x["object_number"]): x for x in catalog}
    valid_ids = sorted(valid)
    try:
        import tkinter as tk
        from tkinter import simpledialog, messagebox
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.update()

        selected = None
        while selected is None:
            n = simpledialog.askinteger(
                "Select object",
                "Choose ONE entire object number.\nValid object numbers: "
                + ", ".join(str(x) for x in valid_ids),
                parent=root,
            )
            if n is None:
                root.destroy()
                raise RuntimeError("Object selection was cancelled by the user.")
            if int(n) in valid:
                selected = valid[int(n)]
            else:
                messagebox.showerror("Invalid object", f"Object {n} is not in the catalog. Valid: {valid_ids}", parent=root)

        action = None
        while action is None:
            raw = simpledialog.askstring(
                "Choose action",
                f"OBJ {selected['object_number']}  {selected['object_label']}\n\n"
                "Type GRASP or PUSH:",
                parent=root,
            )
            if raw is None:
                root.destroy()
                raise RuntimeError("Action selection was cancelled by the user.")
            value = raw.strip().lower()
            if value in {"g", "grasp"}:
                action = "grasp"
            elif value in {"p", "push"}:
                action = "push"
            else:
                messagebox.showerror("Invalid action", "Enter GRASP or PUSH.", parent=root)
        root.destroy()
        return selected, action
    except RuntimeError:
        raise
    except Exception as exc:
        print(f"GUI object/action dialog unavailable ({type(exc).__name__}: {exc}). Falling back to terminal input.")
        return _console_prompt_selected_object_and_action(catalog)


def prompt_goal_for_selected(selected: dict):
    """Ask for X,Y,theta only after the user selected PUSH."""
    bounds = selected_goal_bounds(int(selected["body_id"]))
    current = selected["initial_world_position"]
    try:
        import tkinter as tk
        from tkinter import simpledialog, messagebox
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.update()
        messagebox.showinfo(
            "Push target coordinates",
            f"OBJ {selected['object_number']}  {selected['object_label']}\n\n"
            f"CURRENT WORLD pose:\n"
            f"x = {current[0]:+.4f} m\n"
            f"y = {current[1]:+.4f} m\n"
            f"theta = {selected['initial_world_yaw_deg']:+.2f} deg\n\n"
            f"Enter the desired whole-object pose.\n"
            f"Safe X approx: [{bounds['xmin']:+.3f}, {bounds['xmax']:+.3f}] m\n"
            f"Safe Y approx: [{bounds['ymin']:+.3f}, {bounds['ymax']:+.3f}] m",
            parent=root,
        )
        def ask_float(title, prompt):
            while True:
                value = simpledialog.askfloat(title, prompt, parent=root)
                if value is None:
                    root.destroy()
                    raise RuntimeError("Push goal entry was cancelled by the user.")
                if math.isfinite(float(value)):
                    return float(value)
        while True:
            x = ask_float("Desired X", f"Desired X [m]\nSafe approx: {bounds['xmin']:+.3f} ... {bounds['xmax']:+.3f}")
            y = ask_float("Desired Y", f"Desired Y [m]\nSafe approx: {bounds['ymin']:+.3f} ... {bounds['ymax']:+.3f}")
            if bounds["xmin"] <= x <= bounds["xmax"] and bounds["ymin"] <= y <= bounds["ymax"]:
                break
            messagebox.showerror("Goal outside safe table area", "Choose an X,Y inside the displayed safe bounds.", parent=root)
        th = ask_float("Desired theta", "Desired tabletop yaw theta [deg]\n0 deg = +X, +90 deg = +Y")
        root.destroy()
        return np.array([x, y, math.radians(th)], dtype=np.float32)
    except RuntimeError:
        raise
    except Exception as exc:
        print(f"GUI goal dialog unavailable ({type(exc).__name__}: {exc}). Falling back to terminal input.")
        return _console_prompt_goal_world()

def wait_before_close(message="Final result is printed. Close this dialog to close PyBullet."):
    if not USE_GUI or not p.isConnected():
        return
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        messagebox.showinfo("Primitive Push Engine", message, parent=root)
        root.destroy()
    except Exception:
        try:
            input("\nPress ENTER to close PyBullet...")
        except EOFError:
            # Keep the GUI alive for a short inspection window rather than
            # immediately shutting it down when stdin is unavailable.
            for _ in range(int(8.0 / DT)):
                if not p.isConnected():
                    break
                p.stepSimulation()
                time.sleep(DT)


def primitive_goal_from_object_goal(record_world: dict, target_body_id: int, object_goal_world: np.ndarray):
    body_pos, body_yaw = body_pose_world(target_body_id)
    prim_center = np.asarray(record_world["center_world"], dtype=np.float64)[:2]
    offset_local = rot2(-body_yaw) @ (prim_center - body_pos[:2])
    desired_xy = np.asarray(object_goal_world[:2], dtype=np.float64) + rot2(float(object_goal_world[2])) @ offset_local
    delta_yaw = wrap_angle(float(object_goal_world[2]) - body_yaw)
    desired_prim_yaw = wrap_angle(float(record_world["state_vector"][2]) + delta_yaw)
    return world_state_to_push_frame(np.array([desired_xy[0], desired_xy[1], desired_prim_yaw], dtype=np.float32))


def body_union_aabb(body_id: int):
    lows, highs = [], []
    for link in [-1] + list(range(p.getNumJoints(int(body_id)))):
        try:
            lo, hi = p.getAABB(int(body_id), int(link))
        except Exception:
            continue
        lows.append(np.asarray(lo, dtype=np.float64))
        highs.append(np.asarray(hi, dtype=np.float64))
    if not lows:
        raise RuntimeError(f"No AABB for body {body_id}")
    return np.min(np.stack(lows), axis=0), np.max(np.stack(highs), axis=0)



class LiveXYTracker:
    """Continuously updated top-down view of the SELECTED object's real motion.

    PyBullet GUI shows the 3-D physical simulation.  This window shows the same
    selected rigid body in WORLD X-Y coordinates, including target, orientation,
    executed path, current pusher location and the currently planned push vector.
    """

    def __init__(self, catalog, scene, selected_body_id, selected_object_number, goal_world):
        self.catalog = list(catalog)
        self.scene = scene
        self.selected_body_id = int(selected_body_id)
        self.selected_object_number = int(selected_object_number)
        self.goal_world = np.asarray(goal_world, dtype=np.float64)
        self.path_xy = []
        self.settled_path_xy = []
        self.planned_candidate = None
        self.frame_counter = 0
        self.window_name = "LIVE PUSH - SELECTED OBJECT WORLD XY"
        self.out_path = OUTPUT_DIR / "live_selected_object_xy.png"

    def set_plan(self, candidate_world):
        self.planned_candidate = dict(candidate_world) if candidate_world is not None else None

    def _w2p(self, x, y, width=940, height=720, margin=65):
        xmin, xmax = -TABLE_HALF_X, TABLE_HALF_X
        ymin, ymax = -TABLE_HALF_Y, TABLE_HALF_Y
        px = margin + (float(x) - xmin) / max(xmax - xmin, 1e-9) * (width - 2 * margin)
        py = height - margin - (float(y) - ymin) / max(ymax - ymin, 1e-9) * (height - 2 * margin)
        return int(round(px)), int(round(py))

    def _draw_body_footprint(self, canvas, body_id, color, thickness=2):
        try:
            lo, hi = body_union_aabb(int(body_id))
            a = self._w2p(lo[0], lo[1])
            b = self._w2p(hi[0], hi[1])
            x1, x2 = sorted((a[0], b[0]))
            y1, y2 = sorted((a[1], b[1]))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        except Exception:
            pass

    def update(self, stage, pusher_xyz=None):
        if not p.isConnected():
            return

        width, height = 940, 720
        canvas = np.full((height, width, 3), 248, dtype=np.uint8)

        # Grid / axes in the exact PyBullet WORLD frame.
        for x in np.arange(-TABLE_HALF_X, TABLE_HALF_X + 1e-9, 0.10):
            a = self._w2p(x, -TABLE_HALF_Y, width, height)
            b = self._w2p(x, TABLE_HALF_Y, width, height)
            cv2.line(canvas, a, b, (225, 225, 225), 1)
        for y in np.arange(-TABLE_HALF_Y, TABLE_HALF_Y + 1e-9, 0.10):
            a = self._w2p(-TABLE_HALF_X, y, width, height)
            b = self._w2p(TABLE_HALF_X, y, width, height)
            cv2.line(canvas, a, b, (225, 225, 225), 1)

        origin = self._w2p(0.0, 0.0, width, height)
        x_tip = self._w2p(0.16, 0.0, width, height)
        y_tip = self._w2p(0.0, 0.16, width, height)
        cv2.arrowedLine(canvas, origin, x_tip, (30, 30, 220), 3, tipLength=0.18)
        cv2.arrowedLine(canvas, origin, y_tip, (30, 160, 30), 3, tipLength=0.18)
        cv2.putText(canvas, "+X", (x_tip[0] + 4, x_tip[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 220), 2)
        cv2.putText(canvas, "+Y", (y_tip[0] + 4, y_tip[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 160, 30), 2)

        # All scene bodies in their CURRENT simulation coordinates.
        number_by_body = {int(o["body_id"]): int(o["object_number"]) for o in self.catalog}
        label_by_body = {int(o["body_id"]): str(o["object_label"]) for o in self.catalog}
        for bid in self.scene["object_ids"]:
            try:
                pos, yaw = body_pose_world(int(bid))
            except Exception:
                continue
            is_selected = int(bid) == self.selected_body_id
            col = (30, 90, 235) if is_selected else (150, 150, 150)
            self._draw_body_footprint(canvas, bid, col, 4 if is_selected else 1)
            q = self._w2p(pos[0], pos[1], width, height)
            cv2.circle(canvas, q, 9 if is_selected else 5, col, -1)
            number = number_by_body.get(int(bid), -1)
            label = label_by_body.get(int(bid), self.scene["body_id_to_name"].get(int(bid), str(bid)))
            cv2.putText(canvas, f"{number}:{label}", (q[0] + 8, q[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (45, 45, 45), 1, cv2.LINE_AA)

            if is_selected:
                tip = self._w2p(pos[0] + 0.070 * math.cos(yaw), pos[1] + 0.070 * math.sin(yaw), width, height)
                cv2.arrowedLine(canvas, q, tip, (30, 90, 235), 3, tipLength=0.22)
                pushing_stage = ("PUSH" in str(stage).upper()) and ("PLANNED" not in str(stage).upper())
                if pushing_stage:
                    if not self.path_xy or np.linalg.norm(np.asarray(self.path_xy[-1]) - pos[:2]) > 0.0020:
                        self.path_xy.append([float(pos[0]), float(pos[1])])
                if str(stage).upper() in {"AFTER PUSH", "FINAL", "TARGET SELECTED"}:
                    if not self.settled_path_xy or np.linalg.norm(np.asarray(self.settled_path_xy[-1]) - pos[:2]) > 0.0010:
                        self.settled_path_xy.append([float(pos[0]), float(pos[1])])

        # Selected-object path.
        if len(self.path_xy) >= 2:
            pts = np.asarray([self._w2p(x, y, width, height) for x, y in self.path_xy], dtype=np.int32)
            cv2.polylines(canvas, [pts], False, (185, 205, 225), 2, cv2.LINE_AA)
        if len(self.settled_path_xy) >= 2:
            pts2 = np.asarray([self._w2p(x, y, width, height) for x, y in self.settled_path_xy], dtype=np.int32)
            cv2.polylines(canvas, [pts2], False, (0, 120, 255), 4, cv2.LINE_AA)

        # Desired target pose.
        gx, gy, gth = [float(v) for v in self.goal_world]
        g = self._w2p(gx, gy, width, height)
        cv2.drawMarker(canvas, g, (0, 0, 230), cv2.MARKER_TILTED_CROSS, 26, 4)
        gt = self._w2p(gx + 0.085 * math.cos(gth), gy + 0.085 * math.sin(gth), width, height)
        cv2.arrowedLine(canvas, g, gt, (0, 0, 230), 4, tipLength=0.22)
        cv2.putText(canvas, "GOAL", (g[0] + 12, g[1] + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 190), 2)

        # Current planned push, before/during execution.
        if self.planned_candidate is not None:
            try:
                c = np.asarray(self.planned_candidate["contact_world"], dtype=float)
                d = np.asarray(self.planned_candidate["direction_world"], dtype=float)
                L = float(self.planned_candidate["push_length"])
                c0 = self._w2p(c[0], c[1], width, height)
                c1 = self._w2p(c[0] + L * d[0], c[1] + L * d[1], width, height)
                cv2.arrowedLine(canvas, c0, c1, (190, 30, 190), 4, tipLength=0.22)
                cv2.circle(canvas, c0, 6, (190, 30, 190), -1)
            except Exception:
                pass

        if pusher_xyz is not None:
            pp = np.asarray(pusher_xyz, dtype=float)
            if pp.size >= 2 and np.all(np.isfinite(pp[:2])):
                q = self._w2p(pp[0], pp[1], width, height)
                cv2.circle(canvas, q, 7, (180, 0, 180), -1)
                cv2.putText(canvas, "TOOL", (q[0] + 8, q[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 0, 140), 1)

        # Numeric status for the exact selected object.
        pos, yaw = body_pose_world(self.selected_body_id)
        dx = float(pos[0] - gx)
        dy = float(pos[1] - gy)
        dth = wrap_angle(float(yaw - gth))
        cv2.rectangle(canvas, (18, 15), (width - 18, 86), (255, 255, 255), -1)
        cv2.rectangle(canvas, (18, 15), (width - 18, 86), (60, 60, 60), 1)
        cv2.putText(canvas, f"STAGE: {stage}", (30, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"OBJ {self.selected_object_number} ACTUAL x={pos[0]:+.3f} y={pos[1]:+.3f} th={math.degrees(yaw):+.1f}deg   |   "
            f"GOAL x={gx:+.3f} y={gy:+.3f} th={math.degrees(gth):+.1f}deg   |   "
            f"ERR={math.hypot(dx,dy)*1000:.1f}mm, {math.degrees(dth):+.1f}deg",
            (30, 69), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (35, 35, 35), 1, cv2.LINE_AA,
        )

        self.frame_counter += 1
        try:
            cv2.imshow(self.window_name, canvas)
            cv2.waitKey(1)
        except Exception:
            pass
        if self.frame_counter % 12 == 0 or stage in {"TARGET SELECTED", "PLANNED PUSH", "AFTER PUSH", "FINAL"}:
            cv2.imwrite(str(self.out_path), canvas)


def build_pybullet_obstacles(scene: dict, target_body_id: int):
    obstacles = []
    for bid in scene["object_ids"]:
        if int(bid) == int(target_body_id):
            continue
        lo, hi = body_union_aabb(int(bid))
        center = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        radius = float(math.hypot(half[0], half[1]))
        obstacles.append({
            "body_id": int(bid),
            "center": (center[:2].astype(np.float32) - PUSH_FRAME_ORIGIN_XY).astype(np.float32),
            "radius": max(radius, 0.025),
        })
    return obstacles


def _advice_with_types(advice: dict, initial_perception: dict):
    records = {int(r["primitive_id"]): r["primitive_type"] for r in initial_perception["primitive_records"]}
    out = dict(advice)
    out["preferred_translation_types"] = sorted({records[pid] for pid in out.get("preferred_translation_primitive_ids", []) if pid in records})
    out["preferred_rotation_types"] = sorted({records[pid] for pid in out.get("preferred_rotation_primitive_ids", []) if pid in records})
    out["avoid_types"] = sorted({records[pid] for pid in out.get("avoid_primitive_ids", []) if pid in records})
    return out


# =============================================================================
# RIGID SELECTED-OBJECT PRIMITIVE TRACKING FALLBACK
# =============================================================================

def _body_rotation_matrix(body_id: int):
    """Return current body position, 3x3 rotation, and tabletop yaw."""
    pos, quat = p.getBasePositionAndOrientation(int(body_id))
    R = np.asarray(p.getMatrixFromQuaternion(quat), dtype=np.float64).reshape(3, 3)
    yaw = float(p.getEulerFromQuaternion(quat)[2])
    return np.asarray(pos, dtype=np.float64), R, yaw


def _build_selected_primitive_tracker(initial_records: list, body_id: int):
    """Freeze the last valid MR-Former primitive decomposition in body coordinates.

    The selected PyBullet object is rigid. Primitive type, size, and local placement
    therefore do not change just because one later RGB-D frame is weak. We retain the
    last valid MR-Former decomposition in the object's local frame and only use it when
    a later selected-object MR-Former frame returns no usable primitives.
    """
    pos, R, body_yaw = _body_rotation_matrix(body_id)
    tracker = []
    for rec in initial_records:
        if int(rec.get("body_id", -1)) != int(body_id):
            continue
        center_w = np.asarray(rec["center_world"], dtype=np.float64)
        axis_w = np.asarray(rec.get("axis_world", [0.0, 0.0, 0.0]), dtype=np.float64)
        item = {
            "record": dict(rec),
            "center_local": R.T @ (center_w - pos),
            "axis_local": R.T @ axis_w,
            "yaw_offset": wrap_angle(float(rec.get("yaw", 0.0)) - body_yaw),
        }
        tracker.append(item)
    return tracker


def _tracked_records_from_selected_object(tracker: list, body_id: int):
    """Transform cached MR-Former primitives with the current rigid-body pose."""
    if not tracker:
        return []
    pos, R, body_yaw = _body_rotation_matrix(body_id)
    out = []
    for item in tracker:
        base = dict(item["record"])
        center_w = pos + R @ np.asarray(item["center_local"], dtype=np.float64)
        axis_w = R @ np.asarray(item["axis_local"], dtype=np.float64)
        n = float(np.linalg.norm(axis_w))
        if n > 1e-9:
            axis_w = axis_w / n
        yaw = wrap_angle(body_yaw + float(item["yaw_offset"]))

        geometry = np.asarray(base.get("geometry_vector", np.zeros(8)), dtype=np.float32).copy()
        if geometry.size >= 8:
            geometry[5:8] = axis_w.astype(np.float32)
        state = np.asarray(base.get("state_vector", np.zeros(6)), dtype=np.float32).copy()
        if state.size >= 6:
            state[0] = float(center_w[0])
            state[1] = float(center_w[1])
            state[2] = float(yaw)
            state[3:6] = 0.0

        base["center_world"] = center_w.astype(np.float32)
        base["axis_world"] = axis_w.astype(np.float32)
        base["yaw"] = float(yaw)
        base["geometry_vector"] = geometry
        base["state_vector"] = state
        base["tracking_source"] = "rigid_last_valid_mrformer"
        out.append(base)
    return out


def _refresh_selected_primitive_tracker(current_records: list, body_id: int, old_tracker: list):
    """Refresh cache only with a non-empty valid MR-Former result."""
    if not current_records:
        return old_tracker
    new_tracker = _build_selected_primitive_tracker(current_records, body_id)
    return new_tracker if new_tracker else old_tracker


# =============================================================================
# MAIN
# =============================================================================


def _execute_push_compat(**kwargs):
    """Call push.execute_push_pybullet without ever dying on optional kw drift.

    This is deliberately defensive because VS Code projects often retain one
    older file while another file is replaced. Required physical arguments are
    never removed; only optional keywords unsupported by the loaded push.py are
    filtered.
    """
    fn = push_module.execute_push_pybullet
    sig = inspect.signature(fn)
    params = sig.parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_var_kw:
        return fn(**kwargs)
    allowed = set(params.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        print("WARNING: loaded push.py does not support optional kwargs; dropping:", dropped)
        print("Loaded push.py:", getattr(push_module, "__file__", "unknown"))
        print("Loaded signature:", sig)
    return fn(**filtered)

def main():
    print("Loaded synchronized execution build:", MAIN_BUILD_ID)
    print("push.py:", getattr(push_module, "__file__", "unknown"))
    print("push build:", getattr(push_module, "PUSH_ENGINE_BUILD_ID", "legacy/unknown"))
    print("execute_push_pybullet signature:", inspect.signature(push_module.execute_push_pybullet))
    client_id = p.connect(p.GUI if USE_GUI else p.DIRECT)
    if client_id < 0:
        raise RuntimeError("Could not connect to PyBullet.")

    try:
        scene = create_scene()

        # ------------------------------------------------------------------
        # ORIGINAL SERIAL FRANKA PANDA ROBOT
        # ------------------------------------------------------------------
        # Exact first-simulation base pose and home joint configuration:
        #   base = [-0.65, 0.00, 0.525]
        #   q    = [0.0, -0.5, 0.0, -2.5, 0.0, 2.0, 0.8]
        #   finger opening = 0.04 m
        #
        # The Panda follows the SAME Cartesian pusher trajectory through IK.
        # Its collisions are disabled so the trained cylindrical pusher remains
        # the only contact body. It is hidden only during RGB-D capture.
        franka_robot = FrankaPandaRobot(
            base_position=(-0.65, 0.00, TABLE_TOP_Z),
            initial_arm_q=(0.0, -0.5, 0.0, -2.5, 0.0, 2.0, 0.8),
            finger_opening=0.04,
        )
        print("Serial Franka Panda restored at the original first-simulation position.")
        draw_world_coordinate_frame()

        print("\nLoading MR-Former...")
        mrformer = load_mrformer()
        # Push-network weights are loaded only if the user selects PUSH.
        push_runtime = None

        # ------------------------------------------------------------------
        # INITIAL PERCEPTION OF ALL EIGHT OBJECTS + GEMINI GROUPING/COM.
        # ------------------------------------------------------------------
        rgb, depth_m, segmentation = capture_rgbd(franka_robot)
        save_raw_frame(rgb, depth_m, segmentation, 0)
        perception0 = run_perception(
            rgb=rgb,
            depth_m=depth_m,
            segmentation_gt=segmentation,
            object_body_ids=scene["object_ids"],
            camera_eye=CAMERA_EYE,
            camera_target=CAMERA_TARGET,
            camera_up=CAMERA_UP,
            fov_y_deg=FOV,
            mrformer=mrformer,
            body_id_to_name=scene["body_id_to_name"],
            body_metadata=scene["body_metadata"],
            use_gemini=True,
        )

        print_camera_report(perception0)
        print_mrformer_results(perception0)
        catalog = build_object_catalog(perception0, get_body_pose_lookup(scene))
        catalog = augment_catalog_with_inertia(catalog, perception0["primitive_records"])
        if not catalog:
            raise RuntimeError("No visible grouped objects were available.")
        grouping_source = perception0["object_primitive_graph"].get("source", "unknown")
        print_object_catalog(catalog, grouping_source)
        print("\nHOW TO CHOOSE DESIRED SIMULATION COORDINATES")
        print("  * Use the INITIAL WORLD x,y,theta columns above as your current reference.")
        print("  * Desired x,y are the target whole-object/base pose in PyBullet WORLD meters.")
        print("  * theta is tabletop yaw: 0 deg = +X, +90 deg = +Y.")
        print(f"  * Table bounds: X=[{-TABLE_HALF_X:+.3f},{TABLE_HALF_X:+.3f}], Y=[{-TABLE_HALF_Y:+.3f},{TABLE_HALF_Y:+.3f}].")
        print("  * The yellow rectangle and object numbers are also drawn directly in the PyBullet GUI.")
        label_objects_in_pybullet(catalog)
        show_object_coordinate_board(catalog, grouping_source)

        with open(OUTPUT_DIR / "initial_object_catalog.json", "w", encoding="utf-8") as f:
            json.dump({
                "grouping": perception0["object_primitive_graph"],
                "objects": catalog,
                "T_world_from_camera": perception0["T_world_from_camera"].tolist(),
                "T_camera_from_world": perception0["T_camera_from_world"].tolist(),
            }, f, indent=2)

        # ------------------------------------------------------------------
        # BUILD ONE FINAL GRASP CANDIDATE PER OBJECT.
        # MR-Former supplies primitive geometry/point clouds; Gemini supplies the
        # rigid-object grouping, whole-object COM and hollow/solid semantics.
        # The grasp library then ranks candidates by COM closeness + collision
        # clearance + gripper fit + Franka reachability.
        # ------------------------------------------------------------------
        grasp_library = build_grasp_library(
            catalog=catalog,
            primitive_records=perception0["primitive_records"],
            all_object_ids=scene["object_ids"],
            franka_robot=franka_robot,
            table_top_z=TABLE_TOP_Z,
            config=GRASP_CONFIG,
        )
        print_grasp_library(grasp_library)
        draw_grasp_library(grasp_library)

        # ------------------------------------------------------------------
        # USER PICKS ONE ENTIRE OBJECT, THEN CHOOSES GRASP OR PUSH.
        # ------------------------------------------------------------------
        selected, task_mode = prompt_selected_object_and_action(catalog)
        target_body_id = int(selected["body_id"])
        print(f"\nSELECTED TASK: OBJ {selected['object_number']} {selected['object_label']} -> {task_mode.upper()}")

        if task_mode == "grasp":
            draw_grasp_library(grasp_library, selected_only=int(selected["object_number"]))
            entry = grasp_library.get(int(selected["object_number"]))
            if entry is None or entry.get("best") is None:
                raise RuntimeError(
                    f"No primitive grasp candidate was available for OBJ {selected['object_number']}."
                )
            grasp_result = execute_selected_grasp(
                franka_robot=franka_robot,
                target_object=selected,
                grasp_entry=entry,
                dt=DT,
                table_top_z=TABLE_TOP_Z,
                config=GRASP_CONFIG,
            )
            print("\n" + "=" * 92)
            print("FINAL GRASP RESULT")
            print("=" * 92)
            print(json.dumps(grasp_result, indent=2))
            wait_before_close(
                "Grasp action finished. The selected object is held/lifted if successful. "
                "Close this dialog to close PyBullet."
            )
            return

        # PUSH path: remove grasp preview clutter, ask for desired X,Y,theta, and
        # then load the recurrent dynamics checkpoint/RMPPI machinery.
        clear_grasp_debug()
        goal_world = prompt_goal_for_selected(selected)
        print("Loading trained recurrent push model...")
        push_runtime = PushModelRuntime()

        # Whole-object yaw is physically unobservable/irrelevant for rotationally
        # symmetric objects.  Do not keep chasing an impossible theta goal for a
        # sphere, flat tape/ring, or upright circular cylinder.
        _label_sym = str(selected.get("object_label", "")).strip().lower()
        _body_sym = str(selected.get("body_name", "")).strip().lower()
        _yaw_irrelevant = (
            "spherical" in _label_sym or "sphere" in _label_sym
            or "tape" in _label_sym or "ring" in _label_sym
            or "cylindrical object" in _label_sym
            or _body_sym in {"spherical_object", "tape", "cylindrical_object"}
        )
        if _yaw_irrelevant:
            _requested_deg = float(math.degrees(float(goal_world[2])))
            goal_world = np.asarray(goal_world, dtype=np.float32).copy()
            goal_world[2] = math.radians(float(selected["initial_world_yaw_deg"]))
            print(
                f"NOTE: OBJ {selected['object_number']} is rotationally symmetric in tabletop yaw. "
                f"Requested theta={_requested_deg:+.1f} deg is ignored; planning/evaluation uses XY only."
            )
        target_body_id = int(selected["body_id"])

        # Cache the INITIAL valid MR-Former decomposition of the selected rigid
        # object. Thin/hollow objects (especially tape/ring) can occasionally
        # produce an empty re-perception frame after confidence cleanup. That is
        # a temporary perception dropout, not a reason to terminate a manipulation
        # that already has a valid primitive decomposition from frame 0.
        selected_initial_records = [
            r for r in perception0["primitive_records"]
            if int(r.get("body_id", -1)) == target_body_id
        ]
        selected_primitive_tracker = _build_selected_primitive_tracker(
            selected_initial_records, target_body_id
        )
        if not selected_primitive_tracker:
            raise RuntimeError(
                "Selected object had no valid initial MR-Former primitive decomposition; "
                "cannot initialize rigid primitive tracking."
            )
        print(
            f"Selected-object primitive tracker initialized with "
            f"{len(selected_primitive_tracker)} MR-Former primitive(s)."
        )

        draw_goal_marker(selected, goal_world)

        # Live XY window: this is refreshed throughout every physical push while
        # the PyBullet GUI simultaneously shows the 3-D Panda/object motion.
        live_xy = LiveXYTracker(
            catalog=catalog,
            scene=scene,
            selected_body_id=target_body_id,
            selected_object_number=int(selected["object_number"]),
            goal_world=goal_world,
        )
        live_xy.update("TARGET SELECTED")

        print("\nUSER TARGET ACCEPTED IN PYBULLET WORLD FRAME:")
        print(
            f"  OBJ {selected['object_number']} {selected['object_label']} -> "
            f"x={goal_world[0]:+.4f} m, y={goal_world[1]:+.4f} m, "
            f"theta={math.degrees(float(goal_world[2])):+.2f} deg"
        )

        initial_pos, initial_yaw = body_pose_world(target_body_id)
        initial_pose = np.array([initial_pos[0], initial_pos[1], initial_yaw], dtype=np.float32)
        delta = np.array([
            goal_world[0] - initial_pose[0],
            goal_world[1] - initial_pose[1],
            wrap_angle(float(goal_world[2] - initial_pose[2])),
        ], dtype=np.float32)

        # ------------------------------------------------------------------
        # GEMINI ROBOTICS ASSISTANCE AGAIN -- SOFT PRIOR ONLY.
        # Exact primitive/contact is still decided by COM math + network + RMPPI.
        # ------------------------------------------------------------------
        advice = gemini_push_advice(
            perception0["rgb_objects_only"],
            perception0["primitive_mask_image"],
            selected,
            delta.tolist(),
        )
        advice = _advice_with_types(advice, perception0)

        print("\n" + "=" * 92)
        print("SELECTED ENTIRE OBJECT")
        print("=" * 92)
        print("Object number:", selected["object_number"])
        print("Label/body:", selected["object_label"], "/", selected["body_name"])
        print("Initial WORLD [x,y,theta_deg]:", [float(initial_pose[0]), float(initial_pose[1]), float(math.degrees(initial_pose[2]))])
        print("Desired WORLD [x,y,theta_deg]:", [float(goal_world[0]), float(goal_world[1]), float(math.degrees(goal_world[2]))])
        print("Gemini estimated whole-object COM WORLD:", np.round(selected["gemini_com_world"], 4).tolist())
        print("True simulation COM WORLD (evaluation only):", np.round(selected["true_com_world"], 4).tolist())
        print(f"Whole-object mass used for inertia: {selected['object_mass_kg']:.4f} kg")
        print(f"Perception-derived planar Izz about whole-object COM: {selected['estimated_Izz_kgm2']:.6e} kg*m^2")
        print(f"True PyBullet Izz (evaluation only): {selected['true_sim_Izz_kgm2']:.6e} kg*m^2")
        print("Gemini soft push prior:", advice)
        print("FINAL decision = COM + moment-of-inertia/torque math + trained recurrent dynamics + RMPPI + force feedback.")

        # Keep Gemini COM rigidly attached to the object through closed-loop motion.
        gemini_com0 = np.asarray(selected["gemini_com_world"], dtype=np.float64)
        if not np.all(np.isfinite(gemini_com0[:2])):
            gemini_com0 = np.asarray(selected["true_com_world"], dtype=np.float64)
        com_offset_local = rot2(-initial_yaw) @ (gemini_com0[:2] - initial_pos[:2])

        rnn_hidden = None
        previous_primitive_type = None
        run_log = []
        object_goal_push = world_state_to_push_frame(goal_world)
        consecutive_execution_failures = 0
        stagnation_count = 0
        temporarily_avoid_types = set()
        previous_position_error = actual_object_error(target_body_id, goal_world)["position_error_m"]
        execution_gain_estimate = None
        last_effective_direction = None
        effective_pushes = 0

        for step in range(MAX_CLOSED_LOOP_PUSHES):
            require_connected()
            print("\n" + "=" * 92)
            print(f"CLOSED LOOP {step + 1}/{MAX_CLOSED_LOOP_PUSHES} -- ONLY OBJECT {selected['object_number']}")
            print("=" * 92)

            if goal_reached(target_body_id, goal_world):
                print("Goal reached within tolerance. No more pushes needed.")
                break

            curr_pos, curr_yaw = body_pose_world(target_body_id)
            curr_object_state_world = np.array([curr_pos[0], curr_pos[1], curr_yaw, 0, 0, 0], dtype=np.float32)
            curr_object_state_push = curr_object_state_world.copy()
            curr_object_state_push[:2] -= PUSH_FRAME_ORIGIN_XY
            curr_com_world_xy = curr_pos[:2].astype(np.float64) + rot2(curr_yaw) @ com_offset_local
            curr_com_push_xy = curr_com_world_xy - PUSH_FRAME_ORIGIN_XY

            # ------------------------------------------------------------------
            # TWO-STAGE CLOSED LOOP
            # ------------------------------------------------------------------
            # 1) While XY is not close enough, push primarily for translation.
            # 2) Once XY is close, stop using the XY stop condition and perform a
            #    dedicated signed-torque yaw correction about the whole-object COM.
            # This fixes the old failure mode where XY was already within tolerance,
            # so every rotational push stopped almost immediately and theta never
            # converged.
            phase_err = actual_object_error(target_body_id, goal_world)
            orientation_phase = (
                (not _yaw_irrelevant)
                and phase_err["position_error_m"] <= YAW_PHASE_POSITION_GATE_M
                and abs(phase_err["yaw_error_deg"]) > YAW_TOLERANCE_DEG
            )
            if orientation_phase:
                print(
                    f"ORIENTATION PHASE: XY error={phase_err['position_error_m']*1000:.1f} mm, "
                    f"yaw error={phase_err['yaw_error_deg']:+.2f} deg -> signed-torque correction."
                )
            else:
                print(
                    f"TRANSLATION PHASE: XY error={phase_err['position_error_m']*1000:.1f} mm, "
                    f"yaw error={phase_err['yaw_error_deg']:+.2f} deg."
                )

            # Re-perceive ONLY the selected object. No repeated Gemini grouping.
            rgb, depth_m, segmentation = capture_rgbd(franka_robot)
            save_raw_frame(rgb, depth_m, segmentation, step + 1)
            current = run_perception(
                rgb=rgb,
                depth_m=depth_m,
                segmentation_gt=segmentation,
                object_body_ids=[target_body_id],
                camera_eye=CAMERA_EYE,
                camera_target=CAMERA_TARGET,
                camera_up=CAMERA_UP,
                fov_y_deg=FOV,
                mrformer=mrformer,
                body_id_to_name=scene["body_id_to_name"],
                body_metadata=scene["body_metadata"],
                use_gemini=False,
                allow_empty=True,
            )

            if current["primitive_records"]:
                current_records = apply_object_semantics_to_records(
                    current["primitive_records"], selected
                )
                selected_primitive_tracker = _refresh_selected_primitive_tracker(
                    current_records, target_body_id, selected_primitive_tracker
                )
                print(
                    f"Selected-object MR-Former refresh valid: "
                    f"{len(current_records)} primitive(s)."
                )
            else:
                # IMPORTANT: do not terminate. The object is rigid and its primitive
                # decomposition was already detected successfully in the initial frame.
                # Transform that last valid decomposition with the object's CURRENT
                # PyBullet pose and continue planning/execution normally.
                current_records = _tracked_records_from_selected_object(
                    selected_primitive_tracker, target_body_id
                )
                current_records = apply_object_semantics_to_records(
                    current_records, selected
                )
                print(
                    "MR-FORMER TEMPORARY DROPOUT -> using rigidly tracked last-valid "
                    f"primitive decomposition ({len(current_records)} primitive(s)); "
                    "pipeline continues to ACTION instead of terminating."
                )

            if not current_records:
                raise RuntimeError(
                    "Selected-object perception and rigid primitive tracking both returned empty records."
                )

            # Which primitive should be pushed?
            # Translation: prefer near COM. Rotation: prefer larger lever arm.
            chosen_record_world, ranking = rank_primitives_for_goal(
                records=current_records,
                object_com_xy=curr_com_world_xy,
                object_state=curr_object_state_world,
                object_goal=np.r_[goal_world, [0, 0, 0]][:6],
                gemini_advice=advice,
                object_inertia_zz_kgm2=selected["estimated_Izz_kgm2"],
                object_mass_kg=selected["object_mass_kg"],
                planning_force_n=FORCE_FEEDBACK.desired_force_n,
            )

            # In the final-yaw phase, explicitly prefer the primitive with the
            # largest COM lever arm / inertia authority.  Translation-oriented
            # primitive ranking is still used in the normal phase.
            if orientation_phase:
                rotation_ranked = sorted(
                    ranking,
                    key=lambda item: (
                        1.35 * float(item.get("rotation_lever_score", 0.0))
                        + 1.25 * float(item.get("inertia_rotation_authority", 0.0))
                        + 0.20 * float(item.get("combined_score", 0.0))
                    ),
                    reverse=True,
                )
                for _ri in rotation_ranked:
                    _rr = next((r for r in current_records
                                if int(r["primitive_id"]) == int(_ri["primitive_id"])), None)
                    if _rr is not None:
                        chosen_record_world = _rr
                        break

            # If the previous execution failed on one primitive type, use the next
            # ranked primitive for one replanning cycle instead of repeating the
            # same unreachable action forever.
            if temporarily_avoid_types:
                for item in ranking:
                    if item["primitive_type"] in temporarily_avoid_types:
                        continue
                    alt = next((r for r in current_records
                                if int(r["primitive_id"]) == int(item["primitive_id"])), None)
                    if alt is not None:
                        chosen_record_world = alt
                        break

            print("Current primitive ranking (math + Gemini soft prior):")
            for item in ranking:
                print(
                    f"  P{item['primitive_id']} {item['primitive_type']:<10} | "
                    f"center->COM={item['center_to_com_m']*1000:5.1f} mm | "
                    f"translation={item['translation_near_com_score']:.3f} | "
                    f"rotation={item['rotation_lever_score']:.3f} | "
                    f"I-authority={item.get('inertia_rotation_authority', 0.0):.3f} | "
                    f"combined={item['combined_score']:.3f}"
                )
            print(f"Chosen primitive for this iteration: P{chosen_record_world['primitive_id']} {chosen_record_world['primitive_type']}")

            if previous_primitive_type is not None and previous_primitive_type != chosen_record_world["primitive_type"]:
                print("Primitive type changed -> recurrent hidden state reset for safety.")
                rnn_hidden = None
            previous_primitive_type = chosen_record_world["primitive_type"]

            planner_record = record_world_to_push_frame(chosen_record_world)
            planner_record["obstacles"] = build_pybullet_obstacles(scene, target_body_id)
            planner_record["object_inertia_zz_kgm2"] = float(selected["estimated_Izz_kgm2"])
            planner_record["object_mass_kg"] = float(selected["object_mass_kg"])
            planner_record["whole_object_com_xy"] = np.asarray(curr_com_push_xy, dtype=np.float32)
            planner_record["planning_push_force_n"] = float(FORCE_FEEDBACK.desired_force_n)
            primitive_goal_push = primitive_goal_from_object_goal(chosen_record_world, target_body_id, goal_world)

            step9 = score_candidates(
                runtime=push_runtime,
                record=planner_record,
                goal_state=primitive_goal_push,
                hidden=rnn_hidden,
                top_k=24,
                object_com_xy=curr_com_push_xy,
                object_state=curr_object_state_push,
                object_goal=object_goal_push,
                gemini_advice=advice,
                pusher_radius=PUSHER_RADIUS,
                object_inertia_zz_kgm2=selected["estimated_Izz_kgm2"],
                object_mass_kg=selected["object_mass_kg"],
                planning_force_n=FORCE_FEEDBACK.desired_force_n,
            )

            best_idx = int(step9["top_indices"][0])
            best_one = step9["candidates"][best_idx]
            best_pred_local = np.asarray(step9["pred_next"][best_idx], dtype=np.float32)
            best_pred_world = best_pred_local.copy()
            best_pred_world[:2] += PUSH_FRAME_ORIGIN_XY

            print("\nNETWORK STEP-9 BEST ONE-STEP SUGGESTION")
            print("family:", best_one["family"])
            print("predicted next primitive WORLD [x,y,theta_deg]:", [float(best_pred_world[0]), float(best_pred_world[1]), float(math.degrees(best_pred_world[2]))])
            print("cost:", float(step9["cost"][best_idx]))
            print(
                "inertia prior: torque_z=",
                f"{float(step9['candidate_torque_z_nm'][best_idx]):+.4f} N*m,",
                "alpha_z~=",
                f"{float(step9['candidate_angular_accel_est_rad_s2'][best_idx]):+.3f} rad/s^2"
            )

            # FIRST action: one decisive network-guided macro push. Later actions
            # are monotonic one-step corrections whenever possible. Short-horizon
            # RMPPI is only a fallback.
            current_err_before = actual_object_error(target_body_id, goal_world)
            one_shot = choose_one_shot_if_available(step9)
            macro_choice = None
            correction_choice = None

            if one_shot is None and step == 0:
                macro_choice = choose_macro_first_candidate(step9, curr_pos[:2], goal_world)
            elif step > 0:
                correction_choice = choose_monotonic_correction(
                    step9,
                    previous_direction=last_effective_direction,
                )

            yaw_choice = None
            if orientation_phase:
                yaw_choice = choose_yaw_correction_candidate(
                    step9=step9,
                    whole_object_com_xy=curr_com_push_xy,
                    current_object_yaw=curr_yaw,
                    goal_object_yaw=float(goal_world[2]),
                    object_inertia_zz_kgm2=selected["estimated_Izz_kgm2"],
                    planning_force_n=FORCE_FEEDBACK.desired_force_n,
                )

            if yaw_choice is not None:
                chosen_local = yaw_choice["candidate"]
                plan_mode = yaw_choice["mode"]
                predicted_terminal_local = yaw_choice["predicted_next"]
                predicted_families = [chosen_local["family"]]
                print(
                    f"Yaw correction: err={math.degrees(yaw_choice['yaw_error_rad']):+.2f} deg | "
                    f"moment arm={yaw_choice['moment_arm_signed_m']*1000:+.1f} mm | "
                    f"torque~={yaw_choice['torque_z_nm']:+.3f} N*m | "
                    f"alpha~={yaw_choice['angular_accel_est_rad_s2']:+.2f} rad/s^2"
                )
            elif one_shot is not None and not orientation_phase:
                chosen_local = one_shot["candidate"]
                plan_mode = "ONE-SHOT NETWORK SOLUTION"
                predicted_terminal_local = one_shot["predicted_next"]
                predicted_families = [chosen_local["family"]]
            elif macro_choice is not None and not orientation_phase:
                chosen_local = macro_choice["candidate"]
                plan_mode = "FIRST MACRO PUSH: NETWORK TOP-K + STRONG GOAL ALIGNMENT"
                predicted_terminal_local = macro_choice["predicted_next"]
                predicted_families = [chosen_local["family"]]
                print(f"Macro first-push alignment to desired translation: {macro_choice['alignment']:.3f}")
            elif correction_choice is not None and not orientation_phase:
                chosen_local = correction_choice["candidate"]
                plan_mode = "MONOTONIC ONE-STEP CORRECTION"
                predicted_terminal_local = correction_choice["predicted_next"]
                predicted_families = [chosen_local["family"]]
                print(
                    f"Predicted correction progress: {correction_choice['position_gain_m']*1000:.1f} mm, "
                    f"predicted motion={correction_choice['predicted_motion_m']*1000:.1f} mm"
                )
            else:
                # If position error is still significant, do NOT allow another
                # multi-step rollout to send the object on a curved/zig-zag path.
                # Use the best learned Step-9 family and let the goal-tracking
                # execution layer enforce a straight monotonic translation.
                if current_err_before["position_error_m"] > POSITION_TOLERANCE_M * 1.35:
                    idx_direct = int(step9["top_indices"][0])
                    chosen_local = step9["candidates"][idx_direct]
                    plan_mode = "GOAL-TRACKED STEP-9 CORRECTION"
                    predicted_terminal_local = np.asarray(step9["pred_next"][idx_direct], dtype=np.float32)
                    predicted_families = [chosen_local["family"]]
                else:
                    # Only near the requested XY position do we permit a short
                    # RMPPI fallback, primarily for coupled yaw correction.
                    plan = rmppi_plan(
                        runtime=push_runtime,
                        step9=step9,
                        hidden=rnn_hidden,
                        horizon=2,
                        num_rollouts=128,
                        iterations=2,
                        pool_size=20,
                        temperature=0.7,
                        seed=1000 + step,
                    )
                    chosen_local = plan["first_candidate"]
                    plan_mode = "NEAR-GOAL RMPPI YAW FALLBACK; EXECUTE FIRST ONLY"
                    predicted_terminal_local = np.asarray(plan["predicted_trajectory"][-1], dtype=np.float32)
                    predicted_families = list(plan["best_families"])

            chosen_world = candidate_push_to_world(chosen_local)

            if orientation_phase:
                # DO NOT retarget a yaw-correction push toward the XY goal.
                # Its tangential/off-center direction is exactly what creates the
                # signed torque about the whole-object COM.
                chosen_world["yaw_tracking_active"] = True
                chosen_world["push_length"] = float(np.clip(
                    chosen_world.get("push_length", 0.045), 0.024, 0.070
                ))
                chosen_world["push_speed"] = float(np.clip(
                    chosen_world.get("push_speed", 0.045), 0.030, 0.060
                ))
            else:
                # FINAL GOAL-TRACKING LAYER for translation only.
                chosen_world = retarget_candidate_to_goal(
                    candidate=chosen_world,
                    target_body_id=target_body_id,
                    goal_pose_world=goal_world,
                    primitive_center_world=chosen_record_world.get("center_world"),
                    activate_above_m=POSITION_TOLERANCE_M * 1.35,
                )

                chosen_world = tune_execution_candidate(
                    chosen_world,
                    step,
                    current_err_before,
                    curr_pos[:2],
                    goal_world,
                    stagnation_count=stagnation_count,
                    execution_gain_estimate=execution_gain_estimate,
                )
            predicted_terminal_world = np.asarray(predicted_terminal_local, dtype=np.float32).copy()
            predicted_terminal_world[:2] += PUSH_FRAME_ORIGIN_XY

            # ------------------------------------------------------------------
            # EXECUTION FEASIBILITY GATE
            # ------------------------------------------------------------------
            # The network decides the useful push, but it does NOT solve Franka IK
            # or exact simulator contact geometry. Before the robot moves, verify
            # that the selected side really intersects the chosen object and that
            # approach/contact/end poses are reachable. If RMPPI's first action is
            # not executable, use the next best Step-9 action that is.
            execution_preview = prepare_executable_push(
                target_body_id=target_body_id,
                candidate=chosen_world,
                franka_robot=franka_robot,
                support_z=TABLE_TOP_Z,
                radius=PUSHER_RADIUS,
                pusher_height=0.10,
            )

            if not execution_preview.get("feasible", False):
                print("\nPlanner's first action is not physically executable:")
                print(" ", execution_preview.get("reason", "unknown reason"))
                print("Searching Step-9 top candidates for the best reachable real contact...")
                replacement_found = False
                for _idx in step9["top_indices"]:
                    _idx = int(_idx)
                    _alt_local = step9["candidates"][_idx]
                    _alt_world = candidate_push_to_world(_alt_local)
                    _alt_world = retarget_candidate_to_goal(
                        candidate=_alt_world,
                        target_body_id=target_body_id,
                        goal_pose_world=goal_world,
                        primitive_center_world=chosen_record_world.get("center_world"),
                        activate_above_m=POSITION_TOLERANCE_M * 1.35,
                    )
                    _alt_world = tune_execution_candidate(
                        _alt_world, step, current_err_before, curr_pos[:2], goal_world,
                        stagnation_count=stagnation_count,
                        execution_gain_estimate=execution_gain_estimate,
                    )
                    _alt_preview = prepare_executable_push(
                        target_body_id=target_body_id,
                        candidate=_alt_world,
                        franka_robot=franka_robot,
                        support_z=TABLE_TOP_Z,
                        radius=PUSHER_RADIUS,
                        pusher_height=0.10,
                    )
                    if _alt_preview.get("feasible", False):
                        chosen_local = _alt_local
                        chosen_world = _alt_preview["candidate"]
                        execution_preview = _alt_preview
                        predicted_terminal_local = np.asarray(step9["pred_next"][_idx], dtype=np.float32)
                        predicted_terminal_world = predicted_terminal_local.copy()
                        predicted_terminal_world[:2] += PUSH_FRAME_ORIGIN_XY
                        predicted_families = [chosen_local["family"]]
                        plan_mode += " -> EXECUTION-FEASIBLE TOP-K FALLBACK"
                        replacement_found = True
                        print(f"Reachable replacement: Step-9 candidate {_idx}, family={chosen_local['family']}")
                        break
                if not replacement_found:
                    consecutive_execution_failures += 1
                    temporarily_avoid_types.add(chosen_record_world["primitive_type"])
                    print("No standard preview was reachable for this primitive on this cycle.")
                    print("IMPORTANT: the final executor now has relaxed/position-only Panda IK and a")
                    print("Cartesian-pusher fallback, so we will still attempt the selected physical push")
                    print("instead of exiting before any action.")
                    # Keep the already goal-retargeted candidate.  The low-level executor
                    # will localize the true collision surface again and attempt relaxed IK.
                    # If the Panda is marginally outside its nominal workspace, the red Cartesian
                    # pusher still performs the planned physical interaction while the Panda follows
                    # the closest feasible pose.  This prevents zero-action termination.
                    execution_preview = {"feasible": True, "candidate": chosen_world, "reason": "relaxed-execution fallback"}
                    consecutive_execution_failures = min(consecutive_execution_failures, MAX_CONSECUTIVE_EXECUTION_FAILURES - 1)
            else:
                chosen_world = execution_preview["candidate"]

            print("\nFINAL PLANNER DECISION:", plan_mode)
            print("predicted families:", predicted_families)
            print("execute:", chosen_local["family"])
            print("contact WORLD:", np.round(chosen_world["contact_world"], 4).tolist())
            print("direction WORLD:", np.round(chosen_world["direction_world"], 4).tolist())
            print("length/speed:", chosen_world["push_length"], "m /", chosen_world["push_speed"], "m/s")
            print("predicted terminal primitive WORLD [x,y,theta_deg]:", [float(predicted_terminal_world[0]), float(predicted_terminal_world[1]), float(math.degrees(predicted_terminal_world[2]))])

            # Show the selected push BEFORE execution, then update the XY window
            # continuously from the real PyBullet object state during approach,
            # contact, push, retract and settling.
            live_xy.set_plan(chosen_world)
            live_xy.update("PLANNED PUSH")

            def _live_progress(stage_name, pusher_xyz=None):
                live_xy.update(stage_name, pusher_xyz)

            # Update recurrent history using the action that is ACTUALLY executed
            # after goal-direction/contact/length corrections, not the pre-gate
            # nominal network candidate.
            executed_model_candidate = candidate_world_to_push_frame(chosen_world)
            proposed_hidden = advance_hidden(
                push_runtime, planner_record, executed_model_candidate, rnn_hidden
            )
            execution = _execute_push_compat(
                target_body_id=target_body_id,
                candidate=chosen_world,
                dt=DT,
                radius=PUSHER_RADIUS,
                support_z=TABLE_TOP_Z,
                franka_robot=franka_robot,
                progress_callback=_live_progress,
                realtime_visualization=True,
                display_every_steps=12,
                return_home=False,
                fast_robot_motion=True,
                goal_pose_world=goal_world,
                goal_position_tolerance_m=POSITION_TOLERANCE_M,
                abort_if_goal_error_worsens=True,
                force_feedback_enabled=True,
                force_config=FORCE_FEEDBACK,
                object_com_world=np.array([curr_com_world_xy[0], curr_com_world_xy[1], gemini_com0[2]], dtype=np.float64),
                object_inertia_zz_kgm2=selected["estimated_Izz_kgm2"],
                yaw_tracking_active=bool(orientation_phase),
                goal_yaw_world=float(goal_world[2]),
                goal_yaw_tolerance_deg=YAW_TOLERANCE_DEG,
                max_xy_drift_m=YAW_CORRECTION_MAX_XY_DRIFT_M,
                fine_yaw_band_deg=YAW_FINE_BAND_DEG,
                fine_force_scale=YAW_FINE_FORCE_SCALE,
            )
            live_xy.update("AFTER PUSH")
            print("\nEXECUTION:", execution)
            if execution["contact_established"]:
                displacement = float(execution.get("object_displacement_m", 0.0))
                travel = max(float(execution.get("actual_travel_m", 0.0)), 1e-6)
                yaw_motion_deg = abs(math.degrees(float(execution.get("actual_object_delta", [0.0, 0.0, 0.0])[2])))
                effective_motion = (
                    displacement >= 0.0025
                    or (orientation_phase and yaw_motion_deg >= YAW_CORRECTION_EFFECTIVE_MIN_DEG)
                )
                if effective_motion:
                    rnn_hidden = proposed_hidden
                    consecutive_execution_failures = 0
                    temporarily_avoid_types.clear()
                    effective_pushes += 1
                    if not orientation_phase:
                        measured_gain = float(np.clip(displacement / travel, 0.15, 1.25))
                        if execution_gain_estimate is None:
                            execution_gain_estimate = measured_gain
                        else:
                            execution_gain_estimate = 0.65 * execution_gain_estimate + 0.35 * measured_gain
                        print(
                            f"Effective translation push -> LSTM history updated; "
                            f"measured object/pusher gain={measured_gain:.3f}, filtered gain={execution_gain_estimate:.3f}."
                        )
                    else:
                        print(
                            f"Effective YAW push -> LSTM history updated; "
                            f"actual yaw change={yaw_motion_deg:.2f} deg, "
                            f"stop={execution.get('goal_stop_reason')}."
                        )
                    last_effective_direction = _candidate_direction_xy(chosen_world)
                    stagnation_count = 0
                else:
                    consecutive_execution_failures += 1
                    stagnation_count += 1
                    temporarily_avoid_types.add(chosen_record_world["primitive_type"])
                    if orientation_phase:
                        print(
                            f"NO-OP YAW PUSH: contact occurred but yaw changed only {yaw_motion_deg:.2f} deg "
                            f"and XY moved {displacement*1000:.1f} mm."
                        )
                    else:
                        print("NO-OP PUSH: contact occurred but selected object moved < 2.5 mm.")
                    print("This push is NOT counted as progress and its primitive is avoided next cycle.")
            else:
                consecutive_execution_failures += 1
                temporarily_avoid_types.add(chosen_record_world["primitive_type"])
                print("No contact -> recurrent history NOT updated.")
                print("The next loop will re-perceive the moved/current object and try the next ranked primitive.")

            err = actual_object_error(target_body_id, goal_world)
            improvement = float(previous_position_error - err["position_error_m"])
            previous_position_error = float(err["position_error_m"])
            if execution.get("contact_established", False):
                if orientation_phase:
                    yaw_before_abs = abs(float(execution.get("yaw_error_before_deg", phase_err["yaw_error_deg"])))
                    yaw_after_abs = abs(float(err["yaw_error_deg"]))
                    yaw_improvement_deg = yaw_before_abs - yaw_after_abs
                    if yaw_improvement_deg < 1.0 and yaw_after_abs > YAW_TOLERANCE_DEG:
                        stagnation_count += 1
                        print(
                            f"Small yaw improvement ({yaw_improvement_deg:+.2f} deg) -> "
                            f"next cycle will choose a stronger/different signed-torque contact."
                        )
                elif improvement < 0.0025 and err["position_error_m"] > POSITION_TOLERANCE_M:
                    stagnation_count += 1
                    print(f"Small goal-error improvement ({improvement*1000:.1f} mm) -> strengthen next correction slightly.")
            print("\nDESIRED VS ACTUAL ENTIRE OBJECT")
            print("desired [x,y,theta_deg]:", [float(goal_world[0]), float(goal_world[1]), float(math.degrees(goal_world[2]))])
            print("actual  [x,y,theta_deg]:", [err["actual_pose_world"][0], err["actual_pose_world"][1], float(math.degrees(err["actual_pose_world"][2]))])
            print(
                f"errors: dx={err['dx_m']*1000:+.1f} mm, dy={err['dy_m']*1000:+.1f} mm, "
                f"position={err['position_error_m']*1000:.1f} mm, theta={err['yaw_error_deg']:+.2f} deg"
            )

            run_log.append({
                "step": step + 1,
                "object_number": int(selected["object_number"]),
                "chosen_primitive_type": chosen_record_world["primitive_type"],
                "chosen_primitive_id_current": int(chosen_record_world["primitive_id"]),
                "plan_mode": plan_mode,
                "push_family": chosen_local["family"],
                "execution": execution,
                "desired_object_world": [float(x) for x in goal_world],
                "actual_error": err,
            })
            with open(OUTPUT_DIR / "selected_object_push_log.json", "w", encoding="utf-8") as f:
                json.dump(run_log, f, indent=2)

            if not execution.get("contact_established", False) and consecutive_execution_failures >= MAX_CONSECUTIVE_EXECUTION_FAILURES:
                print("Maximum consecutive contact failures reached. Stopping safely.")
                break

        # Return to the original Franka home only ONCE after the complete task,
        # instead of wasting time returning home between every correction push.
        try:
            franka_robot.home_fast(dt=DT, realtime=True, progress_callback=lambda stage, xyz=None: live_xy.update(stage, xyz))
        except Exception as exc:
            print("Final Panda home warning:", exc)

        final_err = actual_object_error(target_body_id, goal_world)
        print("\n" + "=" * 92)
        print("FINAL RESULT")
        print("=" * 92)
        print("Object number:", selected["object_number"], selected["object_label"], "/", selected["body_name"])
        print("Initial WORLD [x,y,theta_deg]:", [float(initial_pose[0]), float(initial_pose[1]), float(math.degrees(initial_pose[2]))])
        print("Desired WORLD [x,y,theta_deg]:", [float(goal_world[0]), float(goal_world[1]), float(math.degrees(goal_world[2]))])
        print("Final actual WORLD [x,y,theta_deg]:", [final_err["actual_pose_world"][0], final_err["actual_pose_world"][1], float(math.degrees(final_err["actual_pose_world"][2]))])
        print(f"FINAL ERROR: position={final_err['position_error_m']*1000:.1f} mm, theta={final_err['yaw_error_deg']:+.2f} deg")
        if not _yaw_irrelevant and abs(float(final_err["yaw_error_deg"])) > YAW_TOLERANCE_DEG:
            print(
                f"NOTE: yaw is still outside the final +/-{YAW_TOLERANCE_DEG:.1f} deg band; "
                "the run ended because the closed-loop action budget was exhausted, not because the angle was accepted."
            )
        print("Number of executed closed-loop attempts:", len(run_log))
        print("Number of EFFECTIVE object-moving pushes:", effective_pushes)
        print("Outputs:", OUTPUT_DIR)
        live_xy.set_plan(None)
        live_xy.update("FINAL")
        print("Live selected-object XY image:", live_xy.out_path)

        wait_before_close()

    except Exception:
        print("\nPIPELINE ERROR:")
        traceback.print_exc()
        wait_before_close("A pipeline error was printed. Close this dialog to close PyBullet.")
    finally:
        cv2.destroyAllWindows()
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()
