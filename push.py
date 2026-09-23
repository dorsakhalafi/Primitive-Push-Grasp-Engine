from __future__ import annotations
PUSH_ENGINE_BUILD_ID = "2026-09-22_PUSH_GRASP_ENGINE_V1"

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pybullet as p
import torch
import torch.nn as nn

from force_feedback import (
    ForceFeedbackConfig,
    CartesianPushForceController,
    measure_push_force_pybullet,
)


ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"
OUTPUT_DIR = ROOT / "outputs"
PLANNING_DIR = OUTPUT_DIR / "planning_logs"
PLANNING_DIR.mkdir(parents=True, exist_ok=True)

def _resolve_checkpoint_path():
    preferred = WEIGHTS_DIR / "primitive_recurrent_forward_latest.pt"
    if preferred.exists():
        return preferred
    best = sorted(WEIGHTS_DIR.glob("primitive_recurrent_forward_best_*.pt"), key=lambda x: x.stat().st_mtime)
    if best:
        return best[-1]
    any_pt = sorted(WEIGHTS_DIR.glob("*.pt"), key=lambda x: x.stat().st_mtime)
    if any_pt:
        return any_pt[-1]
    return preferred

CHECKPOINT_PATH = _resolve_checkpoint_path()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WORKSPACE = {"xmin": -0.30, "xmax": 0.30, "ymin": -0.22, "ymax": 0.22}

MODEL_PRIMITIVE_ID = {
    "cuboid": 0,
    "sphere": 1,
    "hemisphere": 2,
    "cylinder": 3,
    "ring": 4,
    "stick": 5,
    "cone": 6,
}

PUSH_LENGTHS = (0.025, 0.040, 0.060, 0.080)
PUSH_SPEEDS = (0.020, 0.040, 0.060)


# =============================================================================
# RECURRENT FORWARD MODEL - EXACTLY THE STEP-8 ARCHITECTURE
# =============================================================================

class PrimitiveRecurrentForwardModel(nn.Module):
    def __init__(
        self,
        num_primitives=7,
        continuous_dim=26,
        embedding_dim=12,
        pre_hidden_dim=128,
        lstm_hidden_dim=128,
        lstm_layers=2,
        dropout=0.10,
    ):
        super().__init__()
        self.primitive_embedding = nn.Embedding(num_primitives, embedding_dim)
        self.pre = nn.Sequential(
            nn.Linear(continuous_dim + embedding_dim, pre_hidden_dim),
            nn.LayerNorm(pre_hidden_dim),
            nn.GELU(),
            nn.Linear(pre_hidden_dim, pre_hidden_dim),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            input_size=pre_hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=(dropout if lstm_layers > 1 else 0.0),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 64), nn.GELU(), nn.Linear(64, 3)
        )
        self.com_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 64), nn.GELU(), nn.Linear(64, 3)
        )

    def forward(self, primitive_ids, continuous_features, hidden=None):
        emb = self.primitive_embedding(primitive_ids)
        x = torch.cat([emb, continuous_features], dim=-1)
        x = self.pre(x)
        h, hidden_out = self.lstm(x, hidden)
        return self.delta_head(h), self.com_head(h), hidden_out


def _safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class PushModelRuntime:
    def __init__(self, checkpoint_path: Path = CHECKPOINT_PATH):
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Push checkpoint missing:\n{checkpoint_path}\n"
                "Put the FULL Step-8 .pt checkpoint in weights/. Accepted names include "
                "primitive_recurrent_forward_latest.pt or primitive_recurrent_forward_best_*.pt."
            )
        ckpt = _safe_torch_load(checkpoint_path, DEVICE)
        cfg = ckpt["model_config"]
        self.model = PrimitiveRecurrentForwardModel(
            num_primitives=cfg["num_primitives"],
            continuous_dim=cfg["continuous_dim"],
            embedding_dim=cfg["embedding_dim"],
            pre_hidden_dim=cfg["pre_hidden_dim"],
            lstm_hidden_dim=cfg["lstm_hidden_dim"],
            lstm_layers=cfg["lstm_layers"],
            dropout=cfg["dropout"],
        ).to(DEVICE)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.cont_mean = torch.as_tensor(ckpt["cont_mean"], dtype=torch.float32, device=DEVICE)
        self.cont_std = torch.as_tensor(ckpt["cont_std"], dtype=torch.float32, device=DEVICE)
        self.target_mean = torch.as_tensor(ckpt["target_mean"], dtype=torch.float32, device=DEVICE)
        self.target_std = torch.as_tensor(ckpt["target_std"], dtype=torch.float32, device=DEVICE)
        self.checkpoint_path = str(checkpoint_path)


# =============================================================================
# GEOMETRY / ACTION HELPERS
# =============================================================================

def _norm(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def _rot2(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _world_from_local_xy(center_xy, yaw, local_xy):
    return np.asarray(center_xy, float) + _rot2(yaw) @ np.asarray(local_xy, float)


def _direction_from_local_xy(yaw, local_dir):
    d = _rot2(yaw) @ np.asarray(local_dir, float)
    return d / (np.linalg.norm(d) + 1e-12)


def orientation_mode(record: dict):
    name = record["primitive_type"]
    axis = np.asarray(record["axis_world"], dtype=float)
    if name in ("sphere", "cuboid"):
        return "default"
    zscore = abs(float(axis[2])) if np.linalg.norm(axis) > 1e-8 else 1.0
    if name == "hemisphere":
        return "flat_face_supported" if axis[2] >= 0.7 else "rolling_or_tilted"
    if zscore > 0.70:
        return "upright"
    if zscore < 0.35:
        return "lying"
    return "tilted"


def orientation_rule(record: dict):
    name = record["primitive_type"]
    mode = orientation_mode(record)
    if name == "sphere":
        return {"yaw_relevant": False, "yaw_period": None}
    if name == "ring":
        return {"yaw_relevant": False, "yaw_period": None}
    if name == "hemisphere" and mode == "flat_face_supported":
        return {"yaw_relevant": False, "yaw_period": None}
    if name in ("cylinder", "cone") and mode == "upright":
        return {"yaw_relevant": False, "yaw_period": None}
    if name in ("cuboid", "stick") or (name == "cylinder" and mode == "lying"):
        return {"yaw_relevant": True, "yaw_period": math.pi}
    return {"yaw_relevant": True, "yaw_period": 2.0 * math.pi}


def make_candidate(record, family, contact_xyz, direction_xy, length, speed, risk="normal"):
    d = _norm([direction_xy[0], direction_xy[1], 0.0])
    if np.linalg.norm(d[:2]) < 1e-8:
        return None
    return {
        "primitive_id": MODEL_PRIMITIVE_ID[record["primitive_type"]],
        "instance_id": int(record["primitive_id"]),
        "primitive_type": record["primitive_type"],
        "family": family,
        "risk": risk,
        "contact_world": np.asarray(contact_xyz, dtype=np.float32),
        "direction_world": d.astype(np.float32),
        "theta_push": float(math.atan2(d[1], d[0])),
        "push_length": float(length),
        "push_speed": float(speed),
    }


def _expand(out, record, family, contact_xyz, direction_xy, risk="normal"):
    for length in PUSH_LENGTHS:
        for speed in PUSH_SPEEDS:
            c = make_candidate(record, family, contact_xyz, direction_xy, length, speed, risk)
            if c is not None:
                out.append(c)


def generate_push_candidates(record: dict) -> List[dict]:
    """Primitive-family generator using only perceived geometry/state."""
    name = record["primitive_type"]
    g = np.asarray(record["geometry_vector"], float)
    state = np.asarray(record["state_vector"], float)
    center = np.array([state[0], state[1], float(record["center_world"][2])], dtype=float)
    yaw = float(state[2])
    axis = np.asarray(record["axis_world"], float)
    mode = orientation_mode(record)
    L, W, H, Rout, Rin = [float(x) for x in g[:5]]
    out = []

    if name == "cuboid":
        z = center[2]
        for sign in (-1.0, 1.0):
            # +/- local X faces
            face = np.array([sign * L / 2.0, 0.0])
            direction = _direction_from_local_xy(yaw, [-sign, 0.0])
            cxy = _world_from_local_xy(center[:2], yaw, face)
            _expand(out, record, "face_normal_translation", [cxy[0], cxy[1], z], direction)
            for osign in (-1.0, 1.0):
                cxy = _world_from_local_xy(center[:2], yaw, [sign * L / 2.0, osign * 0.30 * W])
                _expand(out, record, "offset_face_rotation", [cxy[0], cxy[1], z], direction)
        for sign in (-1.0, 1.0):
            direction = _direction_from_local_xy(yaw, [0.0, -sign])
            cxy = _world_from_local_xy(center[:2], yaw, [0.0, sign * W / 2.0])
            _expand(out, record, "face_normal_translation", [cxy[0], cxy[1], z], direction)
            for osign in (-1.0, 1.0):
                cxy = _world_from_local_xy(center[:2], yaw, [osign * 0.30 * L, sign * W / 2.0])
                _expand(out, record, "offset_face_rotation", [cxy[0], cxy[1], z], direction)

    elif name == "sphere":
        R = max(Rout, 0.01)
        for phi in np.linspace(0, 2 * math.pi, 8, endpoint=False):
            radial = np.array([math.cos(phi), math.sin(phi)])
            cxy = center[:2] + R * radial
            _expand(out, record, "radial_rolling_push", [cxy[0], cxy[1], center[2]], -radial, "rolling")

    elif name == "hemisphere":
        R = max(Rout, 0.01)
        family = "side_translation" if mode == "flat_face_supported" else "rolling_mode"
        risk = "normal" if mode == "flat_face_supported" else "rolling_or_tipping"
        for phi in np.linspace(0, 2 * math.pi, 8, endpoint=False):
            radial = np.array([math.cos(phi), math.sin(phi)])
            cxy = center[:2] + 0.85 * R * radial
            _expand(out, record, family, [cxy[0], cxy[1], center[2]], -radial, risk)

    elif name == "cylinder":
        R = max(Rout, 0.01)
        if mode == "upright":
            for phi in np.linspace(0, 2 * math.pi, 8, endpoint=False):
                radial = np.array([math.cos(phi), math.sin(phi)])
                cxy = center[:2] + R * radial
                _expand(out, record, "radial_circumference", [cxy[0], cxy[1], center[2]], -radial)
        else:
            ah = _norm([axis[0], axis[1], 0.0])[:2]
            if np.linalg.norm(ah) < 1e-8:
                ah = np.array([math.cos(yaw), math.sin(yaw)])
            side = np.array([-ah[1], ah[0]])
            for sign in (-1.0, 1.0):
                cxy = center[:2] + sign * R * side
                _expand(out, record, "lateral_rolling_push", [cxy[0], cxy[1], center[2]], -sign * side, "rolling")
                cxy2 = center[:2] + sign * max(H, 0.05) / 2.0 * ah
                _expand(out, record, "axial_push", [cxy2[0], cxy2[1], center[2]], -sign * ah)

    elif name == "ring":
        R = max(Rout, 0.01)
        for phi in np.linspace(0, 2 * math.pi, 12, endpoint=False):
            radial = np.array([math.cos(phi), math.sin(phi)])
            cxy = center[:2] + R * radial
            _expand(out, record, "outer_rim_push", [cxy[0], cxy[1], center[2]], -radial)

    elif name == "stick":
        R = max(Rout, 0.005)
        ah = _norm([axis[0], axis[1], 0.0])[:2]
        if np.linalg.norm(ah) < 1e-8:
            ah = np.array([math.cos(yaw), math.sin(yaw)])
        side = np.array([-ah[1], ah[0]])
        if mode == "upright":
            for phi in np.linspace(0, 2 * math.pi, 8, endpoint=False):
                radial = np.array([math.cos(phi), math.sin(phi)])
                cxy = center[:2] + R * radial
                _expand(out, record, "radial_push", [cxy[0], cxy[1], center[2]], -radial)
        else:
            for sgn in (-1.0, 1.0):
                cxy = center[:2] + sgn * R * side
                _expand(out, record, "center_lateral_push", [cxy[0], cxy[1], center[2]], -sgn * side)
                cxy2 = center[:2] + sgn * max(L, 0.08) / 2.0 * ah
                _expand(out, record, "axial_push", [cxy2[0], cxy2[1], center[2]], -sgn * ah)
            for asgn in (-1.0, 1.0):
                for ssgn in (-1.0, 1.0):
                    cxy = center[:2] + asgn * 0.35 * max(L, 0.08) * ah + ssgn * R * side
                    _expand(out, record, "offset_end_rotation", [cxy[0], cxy[1], center[2]], -ssgn * side)

    elif name == "cone":
        R = max(Rout, 0.01)
        if mode == "upright":
            for phi in np.linspace(0, 2 * math.pi, 8, endpoint=False):
                radial = np.array([math.cos(phi), math.sin(phi)])
                cxy = center[:2] + 0.8 * R * radial
                _expand(out, record, "base_low_side_push", [cxy[0], cxy[1], center[2]], -radial)
        else:
            ah = _norm([axis[0], axis[1], 0.0])[:2]
            if np.linalg.norm(ah) < 1e-8:
                ah = np.array([math.cos(yaw), math.sin(yaw)])
            side = np.array([-ah[1], ah[0]])
            for sign in (-1.0, 1.0):
                cxy = center[:2] + sign * 0.8 * R * side
                _expand(out, record, "rolling_side_push", [cxy[0], cxy[1], center[2]], -sign * side, "rolling_or_tipping")

    return out


def candidate_action_vector(candidate: dict):
    c = candidate["contact_world"]
    return np.array(
        [
            candidate["primitive_id"],
            c[0], c[1], c[2],
            candidate["theta_push"],
            candidate["push_length"],
            candidate["push_speed"],
        ],
        dtype=np.float32,
    )


def environment_vector(state):
    x, y = float(state[0]), float(state[1])
    return np.array(
        [x - WORKSPACE["xmin"], WORKSPACE["xmax"] - x,
         y - WORKSPACE["ymin"], WORKSPACE["ymax"] - y],
        dtype=np.float32,
    )


def build_features_np(geometry, state, env, action):
    geometry = np.asarray(geometry, np.float32)
    state = np.asarray(state, np.float32)
    env = np.asarray(env, np.float32)
    action = np.asarray(action, np.float32)
    sf = np.array(
        [state[0], state[1], math.sin(state[2]), math.cos(state[2]), state[3], state[4], state[5]],
        np.float32,
    )
    af = np.array(
        [action[1], action[2], action[3], math.sin(action[4]), math.cos(action[4]), action[5], action[6]],
        np.float32,
    )
    return np.concatenate([geometry, sf, env, af]).astype(np.float32)


def _expand_hidden(hidden, batch):
    if hidden is None:
        return None
    h, c = hidden
    if h.shape[1] == batch:
        return h, c
    if h.shape[1] != 1:
        raise ValueError("Base recurrent hidden state must have batch dimension 1.")
    return h.expand(-1, batch, -1).contiguous(), c.expand(-1, batch, -1).contiguous()


# =============================================================================
# STEP 9 - CANDIDATE SCORING
# =============================================================================

RISK_VALUE = {"normal": 0.0, "rolling": 0.15, "moderate_tipping": 0.5, "rolling_or_tipping": 0.8}


def _orientation_error(theta, goal_theta, rule):
    theta = np.asarray(theta, float)
    if not rule["yaw_relevant"]:
        return np.zeros_like(theta)
    d = theta - float(goal_theta)
    if rule["yaw_period"] == math.pi:
        return 0.5 * np.arctan2(np.sin(2.0 * d), np.cos(2.0 * d))
    return np.arctan2(np.sin(d), np.cos(d))


def predict_candidates(runtime: PushModelRuntime, record: dict, candidates: List[dict], hidden=None):
    state = np.asarray(record["state_vector"], np.float32)
    geometry = np.asarray(record["geometry_vector"], np.float32)
    env = environment_vector(state)
    actions = np.stack([candidate_action_vector(c) for c in candidates], axis=0)
    feats = np.stack([build_features_np(geometry, state, env, a) for a in actions], axis=0)
    feats_t = torch.from_numpy(feats).to(DEVICE)
    feats_t = (feats_t - runtime.cont_mean) / runtime.cont_std
    feats_t = feats_t.unsqueeze(1)
    pid = MODEL_PRIMITIVE_ID[record["primitive_type"]]
    pid_t = torch.full((len(candidates), 1), pid, dtype=torch.long, device=DEVICE)
    hidden_b = _expand_hidden(hidden, len(candidates))
    with torch.no_grad():
        pred_n, _, _ = runtime.model(pid_t, feats_t, hidden_b)
        pred_delta = pred_n[:, 0, :] * runtime.target_std + runtime.target_mean
    pred_delta = pred_delta.cpu().numpy().astype(np.float32)
    next_state = np.repeat(state[None, :], len(candidates), axis=0)
    next_state[:, :3] += pred_delta
    next_state[:, 2] = np.arctan2(np.sin(next_state[:, 2]), np.cos(next_state[:, 2]))
    next_state[:, 3:] = 0.0
    return actions, pred_delta, next_state


def _target_planar_radius(record: dict):
    g = np.asarray(record["geometry_vector"], dtype=float)
    L, W, _, Rout, _ = [float(x) for x in g[:5]]
    return max(L / 2.0, W / 2.0, Rout, 0.025)


def _obstacle_cost(states_xy, record: dict):
    """Approximate collision/non-target-object protection cost.

    Obstacles come from current perceived primitive/object centroids. This does not
    pretend to predict motion of other objects; it only penalizes trajectories that
    move the target too close to them.
    """
    states_xy = np.asarray(states_xy, dtype=float)
    if states_xy.ndim == 1:
        states_xy = states_xy[None, :]
    obstacles = record.get("obstacles", []) or []
    if not obstacles:
        return np.zeros(states_xy.shape[0], dtype=np.float64)
    target_r = _target_planar_radius(record)
    cost = np.zeros(states_xy.shape[0], dtype=np.float64)
    for obs in obstacles:
        center = np.asarray(obs["center"], dtype=float)
        radius = float(obs.get("radius", 0.03))
        safe = target_r + radius + 0.020
        d = np.linalg.norm(states_xy[:, :2] - center[None, :2], axis=1)
        penetration = np.maximum(safe - d, 0.0)
        cost += 20.0 * (penetration / max(safe, 1e-6)) ** 2
    return cost


def score_candidates(runtime, record, goal_state, hidden=None, top_k=20):
    candidates = generate_push_candidates(record)
    if not candidates:
        raise RuntimeError("No primitive-specific push candidates were generated.")
    actions, pred_delta, next_state = predict_candidates(runtime, record, candidates, hidden)
    goal = np.asarray(goal_state, np.float32)
    rule = orientation_rule(record)

    pos_err = np.linalg.norm(next_state[:, :2] - goal[None, :2], axis=1)
    ori_err = np.abs(_orientation_error(next_state[:, 2], goal[2], rule))
    pos_cost = (pos_err / 0.10) ** 2
    ori_cost = 0.5 * (ori_err / math.radians(30.0)) ** 2
    effort_cost = 0.05 * ((actions[:, 5] / max(PUSH_LENGTHS)) ** 2 + 0.25 * (actions[:, 6] / max(PUSH_SPEEDS)) ** 2)
    risk_cost = 0.20 * np.array([RISK_VALUE.get(c.get("risk", "normal"), 0.25) for c in candidates])

    x, y = next_state[:, 0], next_state[:, 1]
    dx_out = np.maximum(WORKSPACE["xmin"] - x, 0) + np.maximum(x - WORKSPACE["xmax"], 0)
    dy_out = np.maximum(WORKSPACE["ymin"] - y, 0) + np.maximum(y - WORKSPACE["ymax"], 0)
    workspace_cost = 50.0 * ((dx_out / 0.02) ** 2 + (dy_out / 0.02) ** 2)
    obstacle_cost = _obstacle_cost(next_state[:, :2], record)
    total = pos_cost + ori_cost + effort_cost + risk_cost + workspace_cost + obstacle_cost
    order = np.argsort(total)
    top = order[: min(top_k, len(order))]

    result = {
        "record": record,
        "goal_state": goal,
        "rule": rule,
        "candidates": candidates,
        "actions": actions,
        "pred_delta": pred_delta,
        "pred_next": next_state,
        "cost": total.astype(np.float32),
        "obstacle_cost": obstacle_cost.astype(np.float32),
        "top_indices": top.astype(np.int64),
    }

    np.savez_compressed(
        PLANNING_DIR / "step9_latest.npz",
        actions=actions,
        pred_delta=pred_delta,
        pred_next=next_state,
        cost=total,
        obstacle_cost=obstacle_cost,
        top_indices=top,
        goal_state=goal,
    )
    return result


# =============================================================================
# STEP 10 - PRIMITIVE-CONSTRAINED DISCRETE RMPPI
# =============================================================================

def _candidate_template(candidate, reference_state):
    c = np.asarray(candidate["contact_world"], float)
    s = np.asarray(reference_state, float)
    yaw = float(s[2])
    offset_world = c[:2] - s[:2]
    offset_local = _rot2(-yaw) @ offset_world
    theta_rel = math.atan2(math.sin(candidate["theta_push"] - yaw), math.cos(candidate["theta_push"] - yaw))
    return {
        "candidate": candidate,
        "offset_local": offset_local.astype(np.float32),
        "contact_z": float(c[2]),
        "theta_rel": float(theta_rel),
    }


def _instantiate_template(template, state):
    s = np.asarray(state, float)
    yaw = float(s[2])
    cxy = s[:2] + _rot2(yaw) @ template["offset_local"]
    base = template["candidate"]
    c = dict(base)
    c["contact_world"] = np.array([cxy[0], cxy[1], template["contact_z"]], np.float32)
    c["theta_push"] = float(yaw + template["theta_rel"])
    c["direction_world"] = np.array([math.cos(c["theta_push"]), math.sin(c["theta_push"]), 0.0], np.float32)
    return c


def _single_model_step(runtime, record, state, candidate, hidden):
    geometry = np.asarray(record["geometry_vector"], np.float32).copy()
    # Rotate world axis XY by the predicted yaw change from the current perceived pose.
    base_yaw = float(record["state_vector"][2])
    dyaw = float(state[2] - base_yaw)
    axis = geometry[5:8].copy()
    if np.linalg.norm(axis) > 1e-8:
        axis_xy = _rot2(dyaw) @ axis[:2]
        geometry[5:8] = np.array([axis_xy[0], axis_xy[1], axis[2]], np.float32)

    env = environment_vector(state)
    action = candidate_action_vector(candidate)
    feat = build_features_np(geometry, state, env, action)
    feat_t = torch.from_numpy(feat).to(DEVICE)
    feat_t = ((feat_t - runtime.cont_mean) / runtime.cont_std).view(1, 1, -1)
    pid = MODEL_PRIMITIVE_ID[record["primitive_type"]]
    pid_t = torch.tensor([[pid]], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        pred_n, _, hidden_out = runtime.model(pid_t, feat_t, hidden)
        delta = pred_n[0, 0] * runtime.target_std + runtime.target_mean
    delta = delta.cpu().numpy()
    next_state = np.asarray(state, np.float32).copy()
    next_state[:3] += delta
    next_state[2] = math.atan2(math.sin(next_state[2]), math.cos(next_state[2]))
    next_state[3:] = 0.0
    return next_state, hidden_out


def _goal_cost(state, goal, rule):
    pos = np.linalg.norm(np.asarray(state[:2]) - np.asarray(goal[:2]))
    ori = abs(float(_orientation_error(np.array([state[2]]), goal[2], rule)[0]))
    return (pos / 0.10) ** 2 + 0.5 * (ori / math.radians(30.0)) ** 2


def rmppi_plan(
    runtime: PushModelRuntime,
    step9: dict,
    hidden=None,
    horizon: int = 4,
    num_rollouts: int = 256,
    iterations: int = 4,
    pool_size: int = 40,
    temperature: float = 1.0,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    order = np.argsort(step9["cost"])

    # Keep family diversity first, then fill by Step-9 score.
    chosen = []
    seen = set()
    for idx in order:
        fam = step9["candidates"][int(idx)]["family"]
        if fam not in seen:
            chosen.append(int(idx)); seen.add(fam)
    for idx in order:
        if int(idx) not in chosen:
            chosen.append(int(idx))
        if len(chosen) >= min(pool_size, len(order)):
            break
    pool = np.array(chosen[: min(pool_size, len(chosen))], dtype=np.int64)
    templates = [_candidate_template(step9["candidates"][int(i)], step9["record"]["state_vector"]) for i in pool]
    one_step = step9["cost"][pool]
    logits = -(one_step - one_step.min()) / max(temperature, 1e-6)
    probs0 = np.exp(logits - logits.max())
    probs0 = probs0 / probs0.sum()
    probs = np.repeat(probs0[None, :], horizon, axis=0)

    best_cost = float("inf")
    best_seq = None
    best_states = None

    for it in range(iterations):
        sequences = np.stack(
            [rng.choice(len(pool), size=num_rollouts, replace=True, p=probs[t]) for t in range(horizon)],
            axis=1,
        )
        costs = np.zeros(num_rollouts, dtype=np.float64)
        all_states = []

        for r in range(num_rollouts):
            state = np.asarray(step9["record"]["state_vector"], np.float32).copy()
            hstate = None if hidden is None else (hidden[0].clone(), hidden[1].clone())
            traj = [state.copy()]
            prev_theta = None
            for t in range(horizon):
                cand = _instantiate_template(templates[int(sequences[r, t])], state)
                state, hstate = _single_model_step(runtime, step9["record"], state, cand, hstate)
                costs[r] += 0.30 * _goal_cost(state, step9["goal_state"], step9["rule"])
                costs[r] += 0.05 * (cand["push_length"] / max(PUSH_LENGTHS)) ** 2
                costs[r] += 0.20 * RISK_VALUE.get(cand.get("risk", "normal"), 0.25)
                if not (WORKSPACE["xmin"] <= state[0] <= WORKSPACE["xmax"] and WORKSPACE["ymin"] <= state[1] <= WORKSPACE["ymax"]):
                    costs[r] += 50.0
                costs[r] += float(_obstacle_cost(np.asarray(state[:2])[None, :], step9["record"])[0])
                if prev_theta is not None:
                    d = math.atan2(math.sin(cand["theta_push"] - prev_theta), math.cos(cand["theta_push"] - prev_theta))
                    costs[r] += 0.02 * d * d
                prev_theta = cand["theta_push"]
                traj.append(state.copy())
            costs[r] += 2.0 * _goal_cost(state, step9["goal_state"], step9["rule"])
            all_states.append(np.stack(traj))

        imin = int(np.argmin(costs))
        if costs[imin] < best_cost:
            best_cost = float(costs[imin])
            best_seq = sequences[imin].copy()
            best_states = all_states[imin].copy()

        # Path-integral weighting and categorical update.
        w = np.exp(-(costs - costs.min()) / max(temperature, 1e-6))
        w /= w.sum() + 1e-12
        for t in range(horizon):
            freq = np.zeros(len(pool), dtype=np.float64)
            np.add.at(freq, sequences[:, t], w)
            freq /= freq.sum() + 1e-12
            probs[t] = 0.30 * probs[t] + 0.70 * freq
            probs[t] = np.maximum(probs[t], 1e-4)
            probs[t] /= probs[t].sum()
        print(f"RMPPI iteration {it+1}/{iterations}: best={costs[imin]:.4f}, global={best_cost:.4f}")

    global_indices = pool[best_seq]
    first_global = int(global_indices[0])
    result = {
        "best_cost": best_cost,
        "best_step9_indices": global_indices,
        "best_families": [step9["candidates"][int(i)]["family"] for i in global_indices],
        "predicted_trajectory": best_states,
        "first_candidate_index": first_global,
        "first_candidate": step9["candidates"][first_global],
    }

    np.savez_compressed(
        PLANNING_DIR / "step10_rmppi_latest.npz",
        best_cost=np.array([best_cost], np.float32),
        best_step9_indices=global_indices,
        predicted_trajectory=best_states,
    )
    with open(PLANNING_DIR / "step10_rmppi_latest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_cost": best_cost,
                "best_step9_indices": [int(i) for i in global_indices],
                "best_families": result["best_families"],
                "first_candidate_index": first_global,
            },
            f,
            indent=2,
        )
    return result


# =============================================================================
# PHYSICAL PYBULLET PUSH EXECUTION -- NO FORCE FEEDBACK
# =============================================================================


def _require_physics_connection():
    if not p.isConnected():
        raise RuntimeError(
            "PyBullet physics server is not connected. Do not close the PyBullet GUI "
            "while the planner/executor is running."
        )


class FrankaPandaRobot:
    """Smooth GUI-visible serial Franka Panda that follows the real pusher path.

    The learned dynamics model was trained with a Cartesian cylindrical pusher.
    Therefore the red cylindrical pusher remains the ONLY collision body that
    touches the scene object.  The Panda is collision-disabled and follows the
    same pusher path with stable IK + POSITION_CONTROL, so the action is visually
    clear without changing the contact physics used by the trained model.

    Restored first-simulation configuration:
        base = [-0.65, 0.00, 0.525]
        q    = [0.0, -0.5, 0.0, -2.5, 0.0, 2.0, 0.8]
        fingers = 0.04 m
    """

    DEFAULT_BASE = (-0.65, 0.00, 0.525)
    DEFAULT_ARM_Q = (0.0, -0.5, 0.0, -2.5, 0.0, 2.0, 0.8)

    def __init__(
        self,
        base_position=DEFAULT_BASE,
        initial_arm_q=DEFAULT_ARM_Q,
        finger_opening=0.04,
        use_fixed_base=True,
        hand_above_pusher_m=0.035,
    ):
        _require_physics_connection()
        self.base_position = np.asarray(base_position, dtype=np.float64)
        self.initial_arm_q = np.asarray(initial_arm_q, dtype=np.float64)
        self.finger_opening = float(finger_opening)
        self.hand_above_pusher_m = float(hand_above_pusher_m)
        self._visible = True

        self.body_id = int(
            p.loadURDF(
                "franka_panda/panda.urdf",
                basePosition=self.base_position.tolist(),
                baseOrientation=[0, 0, 0, 1],
                useFixedBase=bool(use_fixed_base),
            )
        )

        self.arm_joint_indices = []
        self.finger_joint_indices = []
        self.link_name_to_index = {}
        self.lower_limits = []
        self.upper_limits = []
        self.joint_ranges = []
        self.max_forces = []

        for j in range(p.getNumJoints(self.body_id)):
            info = p.getJointInfo(self.body_id, j)
            joint_name = info[1].decode("utf-8", errors="ignore")
            joint_type = int(info[2])
            link_name = info[12].decode("utf-8", errors="ignore")
            self.link_name_to_index[link_name] = int(j)

            if joint_type == p.JOINT_REVOLUTE and len(self.arm_joint_indices) < 7:
                self.arm_joint_indices.append(int(j))
                lo = float(info[8])
                hi = float(info[9])
                if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
                    lo, hi = -2.9, 2.9
                self.lower_limits.append(lo)
                self.upper_limits.append(hi)
                self.joint_ranges.append(hi - lo)
                self.max_forces.append(float(info[10]) if float(info[10]) > 0 else 87.0)

            if "finger_joint" in joint_name:
                self.finger_joint_indices.append(int(j))

        if len(self.arm_joint_indices) != 7:
            raise RuntimeError(
                f"Unexpected Panda URDF: expected 7 revolute arm joints, got {self.arm_joint_indices}"
            )

        for name in ("panda_grasptarget", "panda_hand", "panda_link8"):
            if name in self.link_name_to_index:
                self.ee_link_index = int(self.link_name_to_index[name])
                break
        else:
            self.ee_link_index = int(max(self.link_name_to_index.values()))

        # Stable downward-facing hand.  This orientation is used in standard
        # PyBullet Panda Cartesian examples.
        self.ee_orientation = p.getQuaternionFromEuler([0.0, -math.pi, 0.0])

        # Store visual colors so main.py can hide Panda only during RGB-D capture.
        self._visual_rgba = {}
        for item in p.getVisualShapeData(self.body_id) or []:
            link_index = int(item[1])
            rgba = list(item[7]) if len(item) > 7 else [1, 1, 1, 1]
            self._visual_rgba[link_index] = rgba

        # The cylindrical tool proxy is the only contact body.  Disable all Panda
        # collision groups to avoid double-pushing while preserving true 7-DoF IK.
        for link in range(-1, p.getNumJoints(self.body_id)):
            p.setCollisionFilterGroupMask(self.body_id, link, 0, 0)

        # Move to original home exactly once at initialization.
        for i, joint in enumerate(self.arm_joint_indices):
            p.resetJointState(self.body_id, joint, float(self.initial_arm_q[i]))
        self._set_fingers(self.finger_opening)
        self.command_home()

        print("Franka Panda serial arm loaded with stable IK tracking.")
        print("  base WORLD =", np.round(self.base_position, 4).tolist())
        print("  EE link index =", self.ee_link_index)

    def _set_fingers(self, opening=None):
        value = self.finger_opening if opening is None else float(opening)
        for joint in self.finger_joint_indices:
            p.setJointMotorControl2(
                self.body_id,
                joint,
                p.POSITION_CONTROL,
                targetPosition=value,
                force=30.0,
                positionGain=0.25,
                velocityGain=1.0,
            )

    def current_arm_q(self):
        return np.asarray(
            [p.getJointState(self.body_id, j)[0] for j in self.arm_joint_indices],
            dtype=np.float64,
        )

    def current_tool_center(self):
        """Approximate current red-pusher center associated with the Panda hand."""
        state = p.getLinkState(self.body_id, self.ee_link_index, computeForwardKinematics=True)
        hand = np.asarray(state[4], dtype=np.float64)
        return hand - np.array([0.0, 0.0, self.hand_above_pusher_m], dtype=np.float64)

    def is_tool_center_reachable(self, pusher_center_world):
        return self._ik_for_tool_center(pusher_center_world) is not None

    def command_home(self):
        if not p.isConnected():
            return
        p.setJointMotorControlArray(
            self.body_id,
            self.arm_joint_indices,
            p.POSITION_CONTROL,
            targetPositions=self.initial_arm_q.tolist(),
            forces=self.max_forces,
            positionGains=[0.18] * 7,
            velocityGains=[1.0] * 7,
        )
        self._set_fingers(self.finger_opening)

    def _ik_for_tool_center(self, pusher_center_world):
        """Robust Panda IK for the Cartesian pusher center.

        The previous version rejected marginally reachable points because it used a
        strict 0.90 m spherical gate and one fixed wrist orientation.  The tabletop
        objects near +X are close to the real Panda workspace boundary, especially
        when the required contact is on the far side of an object.

        This version:
          1) uses a smaller hand/tool vertical offset;
          2) allows the actual Panda workspace to be solved by Bullet instead of a
             brittle pre-gate;
          3) tries the preferred downward orientation first;
          4) falls back to position-only IK when orientation is what makes a pose fail.
        """
        center = np.asarray(pusher_center_world, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(center)):
            return None

        hand_target = center + np.array([0.0, 0.0, self.hand_above_pusher_m], dtype=np.float64)
        shoulder_distance = float(np.linalg.norm(hand_target - self.base_position))
        # Panda nominal max reach is below 1 m, but keep a small numerical margin.
        # This is only a coarse impossible-pose filter; it is intentionally much
        # less restrictive than the old 0.90 m gate.
        if shoulder_distance > 1.03 or hand_target[2] < self.base_position[2] - 0.01:
            return None

        rest = self.current_arm_q()
        attempts = []
        # Preferred: stable downward hand.
        attempts.append((True, self.ee_orientation))
        # Two relaxed but still downward-ish orientations improve far-side reach.
        attempts.append((True, p.getQuaternionFromEuler([math.pi, 0.0, 0.0])))
        attempts.append((True, p.getQuaternionFromEuler([math.pi, 0.35, 0.0])))
        # Final fallback: position-only IK.  Panda collisions are disabled and the
        # red cylindrical pusher is the actual contact body, so exact wrist yaw is
        # not part of the learned pushing physics.
        attempts.append((False, None))

        for use_orientation, orientation in attempts:
            try:
                kwargs = dict(
                    bodyUniqueId=self.body_id,
                    endEffectorLinkIndex=self.ee_link_index,
                    targetPosition=hand_target.tolist(),
                    lowerLimits=self.lower_limits,
                    upperLimits=self.upper_limits,
                    jointRanges=self.joint_ranges,
                    restPoses=rest.tolist(),
                    maxNumIterations=220,
                    residualThreshold=3e-4,
                )
                if use_orientation:
                    kwargs["targetOrientation"] = orientation
                ik = p.calculateInverseKinematics(**kwargs)
            except TypeError:
                try:
                    if use_orientation:
                        ik = p.calculateInverseKinematics(
                            self.body_id, self.ee_link_index, hand_target.tolist(), orientation
                        )
                    else:
                        ik = p.calculateInverseKinematics(
                            self.body_id, self.ee_link_index, hand_target.tolist()
                        )
                except Exception:
                    ik = None
            except Exception:
                ik = None

            if ik is None or len(ik) < 7:
                continue
            q = np.asarray(ik[:7], dtype=np.float64)
            if not np.all(np.isfinite(q)):
                continue
            q = np.clip(q, np.asarray(self.lower_limits), np.asarray(self.upper_limits))
            return q

        return None

    def command_tool_center(self, pusher_center_world):
        """Continuously command the Panda to the pusher path without joint jumps."""
        if not p.isConnected():
            return False
        q = self._ik_for_tool_center(pusher_center_world)
        if q is None:
            return False
        p.setJointMotorControlArray(
            self.body_id,
            self.arm_joint_indices,
            p.POSITION_CONTROL,
            targetPositions=q.tolist(),
            forces=self.max_forces,
            positionGains=[0.25] * 7,
            velocityGains=[1.0] * 7,
        )
        self._set_fingers(self.finger_opening)
        return True

    def set_visible(self, visible=True):
        if not p.isConnected():
            return
        self._visible = bool(visible)
        for link_index, rgba in self._visual_rgba.items():
            color = list(rgba)
            if len(color) < 4:
                color = [1.0, 1.0, 1.0, 1.0]
            color[3] = float(rgba[3] if visible else 0.0)
            try:
                p.changeVisualShape(self.body_id, int(link_index), rgbaColor=color)
            except Exception:
                pass

    def clear(self):
        if p.isConnected():
            try:
                p.removeBody(self.body_id)
            except Exception:
                pass


FrankaPandaExecutor = FrankaPandaRobot


def _union_aabb(body_id: int):
    lows = []
    highs = []
    for link in [-1] + list(range(p.getNumJoints(int(body_id)))):
        try:
            lo, hi = p.getAABB(int(body_id), int(link))
            lows.append(np.asarray(lo, dtype=np.float64))
            highs.append(np.asarray(hi, dtype=np.float64))
        except Exception:
            pass
    if not lows:
        raise RuntimeError(f"Could not obtain AABB for body {body_id}")
    return np.min(np.stack(lows), axis=0), np.max(np.stack(highs), axis=0)


def create_cartesian_pusher(
    start_xy,
    pusher_z,
    radius=0.008,
    height=0.10,
    friction=0.5,
    motor_force=250.0,
    visible=True,
):
    """Create the exact Cartesian pusher used by the trained push model.

    During execution it is intentionally rendered RED, so the user can see the
    actual contact point and push direction.  It does not exist during perception.
    """
    _require_physics_connection()
    collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=float(radius), height=float(height))
    visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=float(radius),
        length=float(height),
        rgbaColor=[0.95, 0.08, 0.05, 0.95] if visible else [0, 0, 0, 0],
    )
    gantry = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=-1,
        basePosition=[0, 0, float(pusher_z)],
        baseOrientation=[0, 0, 0, 1],
        linkMasses=[0.05, 0.20],
        linkCollisionShapeIndices=[-1, collision],
        linkVisualShapeIndices=[-1, visual],
        linkPositions=[[0, 0, 0], [0, 0, 0]],
        linkOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
        linkInertialFramePositions=[[0, 0, 0], [0, 0, 0]],
        linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
        linkParentIndices=[0, 1],
        linkJointTypes=[p.JOINT_PRISMATIC, p.JOINT_PRISMATIC],
        linkJointAxis=[[1, 0, 0], [0, 1, 0]],
    )
    p.resetJointState(gantry, 0, float(start_xy[0]))
    p.resetJointState(gantry, 1, float(start_xy[1]))
    p.changeDynamics(gantry, 1, lateralFriction=float(friction), restitution=0.0)
    p.setJointMotorControlArray(
        gantry,
        [0, 1],
        p.VELOCITY_CONTROL,
        targetVelocities=[0.0, 0.0],
        forces=[float(motor_force), float(motor_force)],
    )
    return gantry


def _set_pusher_velocity(gantry, vx, vy, force=250.0):
    _require_physics_connection()
    p.setJointMotorControlArray(
        gantry,
        [0, 1],
        p.VELOCITY_CONTROL,
        targetVelocities=[float(vx), float(vy)],
        forces=[float(force), float(force)],
    )


def _pusher_xyz(gantry):
    _require_physics_connection()
    pos = p.getLinkState(gantry, 1, computeForwardKinematics=True)[4]
    return np.asarray(pos, dtype=np.float64)


def _pusher_xy(gantry):
    return _pusher_xyz(gantry)[:2].copy()


def _snap_contact_to_target_aabb(target_body_id: int, candidate: dict, radius: float):
    """Project perceived contact to the whole selected rigid body's real surface."""
    _require_physics_connection()
    d3 = _norm(candidate["direction_world"])
    d = np.asarray(d3[:2], dtype=np.float64)
    d /= np.linalg.norm(d) + 1e-12

    lo, hi = _union_aabb(int(target_body_id))
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)

    support_radius = abs(d[0]) * half[0] + abs(d[1]) * half[1]
    nominal = np.asarray(candidate["contact_world"], dtype=np.float64)
    tangent = np.array([-d[1], d[0]], dtype=np.float64)
    nominal_offset = nominal[:2] - center[:2]
    tangential_offset = float(np.dot(nominal_offset, tangent))
    tangent_limit = abs(tangent[0]) * half[0] + abs(tangent[1]) * half[1]
    tangential_offset = float(np.clip(tangential_offset, -0.75 * tangent_limit, 0.75 * tangent_limit))

    snapped_xy = center[:2] - d * support_radius + tangent * tangential_offset
    vertical_span = max(float(hi[2] - lo[2]), 1e-3)
    snapped_z = float(np.clip(nominal[2], lo[2] + 0.25 * vertical_span, hi[2] - 0.25 * vertical_span))
    touch_center_xy = snapped_xy - float(radius) * d
    return snapped_xy, snapped_z, touch_center_xy, d


def _remove_debug_items(ids):
    for uid in ids:
        try:
            p.removeUserDebugItem(uid)
        except Exception:
            pass


def execute_push_pybullet(
    target_body_id: int,
    candidate: dict,
    dt=1.0 / 240.0,
    radius=0.008,
    support_z=0.0,
    pusher_height=0.10,
    franka_robot: Optional[FrankaPandaRobot] = None,
    progress_callback=None,
    realtime_visualization=True,
    display_every_steps=6,
):
    """Execute exactly ONE planned push, visibly and in real time.

    Key behavior:
      1) Panda first moves ABOVE the push start.
      2) Panda lowers to the tool/pusher height.
      3) Red pusher approaches the exact selected object surface.
      4) Horizontal push is performed at the network-commanded speed/length.
      5) Tool retracts, Panda lifts and returns home.
      6) progress_callback receives the actual object pose throughout execution,
         allowing main.py to refresh the XY view while PyBullet updates live.

    NO force feedback is used.
    """
    _require_physics_connection()

    target_length = float(candidate["push_length"])
    commanded_speed = max(float(candidate["push_speed"]), 0.005)
    if target_length <= 0.0:
        raise ValueError(f"push_length must be > 0, got {target_length}")

    snapped_xy, snapped_z, touch_center_xy, d = _snap_contact_to_target_aabb(
        int(target_body_id), candidate, float(radius)
    )

    approach_clearance = 0.045
    start_xy = touch_center_xy - approach_clearance * d
    minimum_center_z = float(support_z) + float(pusher_height) / 2.0 + 0.004
    pusher_z = max(minimum_center_z, float(snapped_z))

    before_pos, before_q = p.getBasePositionAndOrientation(int(target_body_id))
    before_yaw = p.getEulerFromQuaternion(before_q)[2]

    # Draw the planned path directly in the real PyBullet scene.
    debug_ids = []
    z_dbg = float(pusher_z)
    push_end_xy = touch_center_xy + target_length * d
    debug_ids.append(p.addUserDebugLine(
        [float(start_xy[0]), float(start_xy[1]), z_dbg],
        [float(touch_center_xy[0]), float(touch_center_xy[1]), z_dbg],
        [0.1, 0.8, 0.1], 3.0, lifeTime=0,
    ))
    debug_ids.append(p.addUserDebugLine(
        [float(touch_center_xy[0]), float(touch_center_xy[1]), z_dbg],
        [float(push_end_xy[0]), float(push_end_xy[1]), z_dbg],
        [0.95, 0.05, 0.05], 5.0, lifeTime=0,
    ))
    debug_ids.append(p.addUserDebugText(
        f"PUSH: {candidate['family']}",
        [float(touch_center_xy[0]), float(touch_center_xy[1]), z_dbg + 0.11],
        [0.9, 0.05, 0.05], 1.1, lifeTime=0,
    ))

    gantry = create_cartesian_pusher(
        start_xy=start_xy,
        pusher_z=pusher_z,
        radius=float(radius),
        height=float(pusher_height),
        friction=0.60,
        motor_force=320.0,
        visible=True,
    )

    callback_counter = 0

    def _notify(stage):
        nonlocal callback_counter
        callback_counter += 1
        if progress_callback is not None and callback_counter % max(int(display_every_steps), 1) == 0:
            try:
                progress_callback(stage, _pusher_xyz(gantry))
            except Exception:
                pass

    def _sim_step(stage, track_pusher=True):
        _require_physics_connection()
        if franka_robot is not None and track_pusher:
            franka_robot.command_tool_center(_pusher_xyz(gantry))
        p.stepSimulation()
        _notify(stage)
        if realtime_visualization:
            import time as _time
            _time.sleep(float(dt))

    # Smoothly bring Panda from its ACTUAL current EE pose to a safe hover,
    # move horizontally above the selected object, then lower.  This avoids the
    # previous visual behavior where IK appeared to make unrelated/random jumps.
    if franka_robot is not None:
        target_start_center = np.array([float(start_xy[0]), float(start_xy[1]), float(pusher_z)])
        if not franka_robot.is_tool_center_reachable(target_start_center):
            p.removeBody(gantry)
            _remove_debug_items(debug_ids)
            return {
                "contact_established": False,
                "actual_travel_m": 0.0,
                "commanded_length_m": target_length,
                "commanded_speed_mps": commanded_speed,
                "snapped_contact_world": [float(snapped_xy[0]), float(snapped_xy[1]), float(snapped_z)],
                "before_pose_world": [float(before_pos[0]), float(before_pos[1]), float(before_yaw)],
                "failure_reason": "Selected contact is outside the Franka Panda IK workspace.",
            }

        def _robot_segment(a, b, steps, stage):
            a = np.asarray(a, dtype=np.float64)
            b = np.asarray(b, dtype=np.float64)
            for k, alpha in enumerate(np.linspace(0.0, 1.0, int(steps))):
                desired = (1.0 - alpha) * a + alpha * b
                franka_robot.command_tool_center(desired)
                p.stepSimulation()
                if progress_callback is not None and k % 10 == 0:
                    try:
                        progress_callback(stage, desired)
                    except Exception:
                        pass
                if realtime_visualization:
                    import time as _time
                    _time.sleep(float(dt))

        current_center = franka_robot.current_tool_center()
        safe_z = max(float(current_center[2]), float(pusher_z) + 0.18)
        lifted_current = np.array([current_center[0], current_center[1], safe_z])
        above_start = np.array([float(start_xy[0]), float(start_xy[1]), safe_z])
        lower_end = target_start_center

        _robot_segment(current_center, lifted_current, 70, "ROBOT LIFT TO SAFE HEIGHT")
        _robot_segment(lifted_current, above_start, 140, "ROBOT MOVE ABOVE OBJECT")
        _robot_segment(above_start, lower_end, 100, "ROBOT LOWER TO PUSH HEIGHT")

    actual_travel = 0.0
    contact_established = False

    try:
        for _ in range(12):
            _sim_step("READY AT START")

        def target_contacts():
            return [
                cp for cp in p.getContactPoints(bodyA=gantry, bodyB=int(target_body_id))
                if cp[3] == 1
            ]

        # 1) Visible horizontal approach to contact.
        approach_speed = 0.040
        _set_pusher_velocity(gantry, approach_speed * d[0], approach_speed * d[1])
        max_approach_steps = int((approach_clearance + 0.025) / (approach_speed * dt)) + 120
        for _ in range(max_approach_steps):
            _sim_step("APPROACH OBJECT")
            if target_contacts():
                contact_established = True
                break
        _set_pusher_velocity(gantry, 0.0, 0.0)

        if not contact_established:
            # Visibly retract to the start and return the Panda home instead of
            # leaving the arm at a strange low pose after a failed contact.
            current_xy = _pusher_xy(gantry)
            back_vec = np.asarray(start_xy, dtype=np.float64) - current_xy
            back_dist = float(np.linalg.norm(back_vec))
            if back_dist > 1e-5:
                back_dir = back_vec / back_dist
                back_speed = 0.055
                _set_pusher_velocity(gantry, back_speed * back_dir[0], back_speed * back_dir[1])
                for _ in range(int(back_dist / (back_speed * dt)) + 80):
                    _sim_step("NO CONTACT - RETRACT")
                    if np.linalg.norm(_pusher_xy(gantry) - np.asarray(start_xy)) < 0.003:
                        break
                _set_pusher_velocity(gantry, 0.0, 0.0)

            if franka_robot is not None:
                q_start = franka_robot.current_arm_q()
                for k, alpha in enumerate(np.linspace(0.0, 1.0, 150)):
                    q = (1.0 - alpha) * q_start + alpha * franka_robot.initial_arm_q
                    p.setJointMotorControlArray(
                        franka_robot.body_id,
                        franka_robot.arm_joint_indices,
                        p.POSITION_CONTROL,
                        targetPositions=q.tolist(),
                        forces=franka_robot.max_forces,
                        positionGains=[0.22] * 7,
                        velocityGains=[1.0] * 7,
                    )
                    p.stepSimulation()
                    if progress_callback is not None and k % 10 == 0:
                        try:
                            progress_callback("NO CONTACT - ROBOT HOME", None)
                        except Exception:
                            pass
                    if realtime_visualization:
                        import time as _time
                        _time.sleep(float(dt))

            return {
                "contact_established": False,
                "actual_travel_m": 0.0,
                "commanded_length_m": target_length,
                "commanded_speed_mps": commanded_speed,
                "snapped_contact_world": [float(snapped_xy[0]), float(snapped_xy[1]), float(snapped_z)],
                "before_pose_world": [float(before_pos[0]), float(before_pos[1]), float(before_yaw)],
                "failure_reason": "Pusher reached the planned approach side but no target contact was detected.",
            }

        # 2) The actual planned push, slowly enough to WATCH in the GUI.
        push_start = _pusher_xy(gantry)
        max_steps = int(target_length / (commanded_speed * dt) * 1.35) + 120
        _set_pusher_velocity(gantry, commanded_speed * d[0], commanded_speed * d[1])
        for _ in range(max_steps):
            _sim_step("PUSHING SELECTED OBJECT")
            actual_travel = float(np.dot(_pusher_xy(gantry) - push_start, d))
            if actual_travel >= target_length:
                break
        _set_pusher_velocity(gantry, 0.0, 0.0)

        # 3) Visible horizontal retract.
        retract_start = _pusher_xy(gantry)
        retract_distance = 0.045
        retract_speed = 0.055
        _set_pusher_velocity(gantry, -retract_speed * d[0], -retract_speed * d[1])
        for _ in range(int(retract_distance / (retract_speed * dt)) + 100):
            _sim_step("RETRACT TOOL")
            if np.linalg.norm(_pusher_xy(gantry) - retract_start) >= retract_distance:
                break
        _set_pusher_velocity(gantry, 0.0, 0.0)

        # Lift Panda away before returning home.
        if franka_robot is not None:
            tool_now = _pusher_xyz(gantry)
            lifted = tool_now + np.array([0.0, 0.0, 0.16])
            for alpha in np.linspace(0.0, 1.0, 80):
                desired = (1.0 - alpha) * tool_now + alpha * lifted
                franka_robot.command_tool_center(desired)
                p.stepSimulation()
                if progress_callback is not None and int(alpha * 80) % 8 == 0:
                    try:
                        progress_callback("LIFT ROBOT", desired)
                    except Exception:
                        pass
                if realtime_visualization:
                    import time as _time
                    _time.sleep(float(dt))

            # Smooth return toward original joint-space home.
            q_start = franka_robot.current_arm_q()
            for alpha in np.linspace(0.0, 1.0, 140):
                q = (1.0 - alpha) * q_start + alpha * franka_robot.initial_arm_q
                p.setJointMotorControlArray(
                    franka_robot.body_id,
                    franka_robot.arm_joint_indices,
                    p.POSITION_CONTROL,
                    targetPositions=q.tolist(),
                    forces=franka_robot.max_forces,
                    positionGains=[0.22] * 7,
                    velocityGains=[1.0] * 7,
                )
                p.stepSimulation()
                if progress_callback is not None and int(alpha * 140) % 10 == 0:
                    try:
                        progress_callback("ROBOT RETURN HOME", lifted)
                    except Exception:
                        pass
                if realtime_visualization:
                    import time as _time
                    _time.sleep(float(dt))

    finally:
        if p.isConnected():
            try:
                p.removeBody(gantry)
            except Exception:
                pass
            _remove_debug_items(debug_ids)

    # 4) Let the selected object settle while the XY view continues updating.
    stable = 0
    for i in range(1200):
        p.stepSimulation()
        if progress_callback is not None and i % 8 == 0:
            try:
                progress_callback("OBJECT SETTLING", None)
            except Exception:
                pass
        if realtime_visualization and i % 2 == 0:
            import time as _time
            _time.sleep(float(dt))
        v, w = p.getBaseVelocity(int(target_body_id))
        if np.linalg.norm(v) < 1e-3 and np.linalg.norm(w) < 1e-3:
            stable += 1
            if stable >= 60:
                break
        else:
            stable = 0

    after_pos, after_q = p.getBasePositionAndOrientation(int(target_body_id))
    after_yaw = p.getEulerFromQuaternion(after_q)[2]

    return {
        "contact_established": bool(contact_established),
        "actual_travel_m": float(actual_travel),
        "commanded_length_m": float(target_length),
        "commanded_speed_mps": float(commanded_speed),
        "snapped_contact_world": [float(snapped_xy[0]), float(snapped_xy[1]), float(snapped_z)],
        "before_pose_world": [float(before_pos[0]), float(before_pos[1]), float(before_yaw)],
        "after_pose_world": [float(after_pos[0]), float(after_pos[1]), float(after_yaw)],
        "failure_reason": None,
    }

# =============================================================================
# REAL RECURRENT-HISTORY UPDATE AFTER EXECUTED ACTION
# =============================================================================

@torch.no_grad()
def advance_hidden(runtime: PushModelRuntime, record: dict, candidate: dict, hidden=None):
    state = np.asarray(record["state_vector"], np.float32)
    geometry = np.asarray(record["geometry_vector"], np.float32)
    env = environment_vector(state)
    action = candidate_action_vector(candidate)
    feat = build_features_np(geometry, state, env, action)
    feat_t = torch.from_numpy(feat).to(DEVICE)
    feat_t = ((feat_t - runtime.cont_mean) / runtime.cont_std).view(1, 1, -1)
    pid = torch.tensor([[MODEL_PRIMITIVE_ID[record["primitive_type"]]]], dtype=torch.long, device=DEVICE)
    _, _, hidden_out = runtime.model(pid, feat_t, hidden)
    return tuple(x.detach() for x in hidden_out)

# =============================================================================
# ENHANCED GOAL/COM-AWARE PRIMITIVE SELECTION + HOLLOW PUSHES
# These definitions intentionally override the earlier generic versions above.
# =============================================================================

def generate_push_candidates(record: dict, pusher_radius: float = 0.008) -> List[dict]:
    """Generate only physically meaningful candidates for one perceived primitive.

    Hollow inner-rim pushes are generated only for ring/tape-like primitives when
    Gemini/perception says the interior is accessible and the opening is wider
    than the pusher plus clearance.
    """
    name = record["primitive_type"]
    g = np.asarray(record["geometry_vector"], float)
    state = np.asarray(record["state_vector"], float)
    center = np.array([state[0], state[1], float(record["center_world"][2])], dtype=float)
    yaw = float(state[2])
    axis = np.asarray(record["axis_world"], float)
    mode = orientation_mode(record)
    L, W, H, Rout, Rin = [float(x) for x in g[:5]]
    out: List[dict] = []

    if name == "cuboid":
        z = center[2]
        for sign in (-1.0, 1.0):
            direction = _direction_from_local_xy(yaw, [-sign, 0.0])
            cxy = _world_from_local_xy(center[:2], yaw, [sign * max(L, 0.04) / 2.0, 0.0])
            _expand(out, record, "face_normal_translation", [cxy[0], cxy[1], z], direction)
            for osign in (-1.0, 1.0):
                cxy = _world_from_local_xy(center[:2], yaw, [sign * max(L, 0.04) / 2.0, osign * 0.30 * max(W, 0.03)])
                _expand(out, record, "offset_face_rotation", [cxy[0], cxy[1], z], direction)
        for sign in (-1.0, 1.0):
            direction = _direction_from_local_xy(yaw, [0.0, -sign])
            cxy = _world_from_local_xy(center[:2], yaw, [0.0, sign * max(W, 0.03) / 2.0])
            _expand(out, record, "face_normal_translation", [cxy[0], cxy[1], z], direction)
            for osign in (-1.0, 1.0):
                cxy = _world_from_local_xy(center[:2], yaw, [osign * 0.30 * max(L, 0.04), sign * max(W, 0.03) / 2.0])
                _expand(out, record, "offset_face_rotation", [cxy[0], cxy[1], z], direction)

    elif name == "sphere":
        R = max(Rout, 0.018)
        for phi in np.linspace(0, 2 * math.pi, 10, endpoint=False):
            radial = np.array([math.cos(phi), math.sin(phi)])
            cxy = center[:2] + R * radial
            _expand(out, record, "radial_rolling_push", [cxy[0], cxy[1], center[2]], -radial, "rolling")

    elif name == "hemisphere":
        R = max(Rout, 0.018)
        family = "side_translation" if mode == "flat_face_supported" else "rolling_mode"
        risk = "normal" if mode == "flat_face_supported" else "rolling_or_tipping"
        for phi in np.linspace(0, 2 * math.pi, 10, endpoint=False):
            radial = np.array([math.cos(phi), math.sin(phi)])
            cxy = center[:2] + 0.85 * R * radial
            _expand(out, record, family, [cxy[0], cxy[1], center[2]], -radial, risk)

    elif name == "cylinder":
        R = max(Rout, 0.015)
        if mode == "upright":
            for phi in np.linspace(0, 2 * math.pi, 10, endpoint=False):
                radial = np.array([math.cos(phi), math.sin(phi)])
                cxy = center[:2] + R * radial
                _expand(out, record, "radial_circumference", [cxy[0], cxy[1], center[2]], -radial)
        else:
            ah = _norm([axis[0], axis[1], 0.0])[:2]
            if np.linalg.norm(ah) < 1e-8:
                ah = np.array([math.cos(yaw), math.sin(yaw)])
            side = np.array([-ah[1], ah[0]])
            for sign in (-1.0, 1.0):
                cxy = center[:2] + sign * R * side
                _expand(out, record, "lateral_rolling_push", [cxy[0], cxy[1], center[2]], -sign * side, "rolling")
                cxy2 = center[:2] + sign * max(H, 0.05) / 2.0 * ah
                _expand(out, record, "axial_push", [cxy2[0], cxy2[1], center[2]], -sign * ah)

    elif name == "ring":
        Rout = max(Rout, 0.025)
        Rin = max(Rin, 0.45 * Rout)
        for phi in np.linspace(0, 2 * math.pi, 12, endpoint=False):
            radial = np.array([math.cos(phi), math.sin(phi)])
            cxy = center[:2] + Rout * radial
            _expand(out, record, "outer_rim_push", [cxy[0], cxy[1], center[2]], -radial)

        hollow = str(record.get("occupancy", "unknown")).lower() == "hollow"
        accessible = bool(record.get("inner_accessible", False))
        opening_ok = Rin > (float(pusher_radius) + 0.004)
        if hollow and accessible and opening_ok:
            for phi in np.linspace(0, 2 * math.pi, 12, endpoint=False):
                radial = np.array([math.cos(phi), math.sin(phi)])
                cxy = center[:2] + Rin * radial
                # From inside the opening, push outward on the inner rim.
                _expand(out, record, "inner_rim_push", [cxy[0], cxy[1], center[2]], radial)

    elif name == "stick":
        R = max(Rout, 0.006)
        ah = _norm([axis[0], axis[1], 0.0])[:2]
        if np.linalg.norm(ah) < 1e-8:
            ah = np.array([math.cos(yaw), math.sin(yaw)])
        side = np.array([-ah[1], ah[0]])
        if mode == "upright":
            for phi in np.linspace(0, 2 * math.pi, 8, endpoint=False):
                radial = np.array([math.cos(phi), math.sin(phi)])
                cxy = center[:2] + R * radial
                _expand(out, record, "radial_push", [cxy[0], cxy[1], center[2]], -radial)
        else:
            for sgn in (-1.0, 1.0):
                cxy = center[:2] + sgn * R * side
                _expand(out, record, "center_lateral_push", [cxy[0], cxy[1], center[2]], -sgn * side)
                cxy2 = center[:2] + sgn * max(L, 0.08) / 2.0 * ah
                _expand(out, record, "axial_push", [cxy2[0], cxy2[1], center[2]], -sgn * ah)
            for asgn in (-1.0, 1.0):
                for ssgn in (-1.0, 1.0):
                    cxy = center[:2] + asgn * 0.35 * max(L, 0.08) * ah + ssgn * R * side
                    _expand(out, record, "offset_end_rotation", [cxy[0], cxy[1], center[2]], -ssgn * side)

    elif name == "cone":
        R = max(Rout, 0.018)
        if mode == "upright":
            for phi in np.linspace(0, 2 * math.pi, 10, endpoint=False):
                radial = np.array([math.cos(phi), math.sin(phi)])
                cxy = center[:2] + 0.80 * R * radial
                _expand(out, record, "base_low_side_push", [cxy[0], cxy[1], center[2]], -radial)
                tangent = np.array([-radial[1], radial[0]])
                biased = _norm(np.r_[(-radial + 0.25 * tangent), 0.0])[:2]
                _expand(out, record, "controlled_offset_push", [cxy[0], cxy[1], center[2]], biased, "moderate_tipping")
        else:
            ah = _norm([axis[0], axis[1], 0.0])[:2]
            if np.linalg.norm(ah) < 1e-8:
                ah = np.array([math.cos(yaw), math.sin(yaw)])
            side = np.array([-ah[1], ah[0]])
            for sign in (-1.0, 1.0):
                cxy = center[:2] + sign * 0.80 * R * side
                _expand(out, record, "rolling_side_push", [cxy[0], cxy[1], center[2]], -sign * side, "rolling_or_tipping")
                # In-plane yaw correction on a lying cone is induced by offset side contact.
                cxy2 = center[:2] + 0.25 * max(H, 0.06) * ah + sign * 0.75 * R * side
                _expand(out, record, "lying_cone_offset_rotation", [cxy2[0], cxy2[1], center[2]], -sign * side, "moderate_tipping")

    return out


def _goal_motion_mix(object_state: np.ndarray, object_goal: np.ndarray):
    state = np.asarray(object_state, dtype=float)
    goal = np.asarray(object_goal, dtype=float)
    pos_error = float(np.linalg.norm(goal[:2] - state[:2]))
    yaw_error = float(math.atan2(math.sin(goal[2] - state[2]), math.cos(goal[2] - state[2])))
    # 5 cm translation and 25 deg rotation are treated as comparable magnitudes.
    t = min(pos_error / 0.05, 1.0)
    r = min(abs(yaw_error) / math.radians(25.0), 1.0)
    denom = t + r + 1e-9
    return t / denom, r / denom, pos_error, yaw_error


def rank_primitives_for_goal(
    records: List[dict],
    object_com_xy,
    object_state,
    object_goal,
    gemini_advice: Optional[dict] = None,
):
    """Choose which primitive should be contacted using math + Gemini soft prior.

    Mathematical principle:
      - contact region near the whole-object COM is favored for translation;
      - contact region far from COM is favored when significant yaw change is needed.
    Gemini is only a soft semantic prior; it never overrides geometry/RMPPI.
    """
    if not records:
        raise RuntimeError("No MR-Former primitives are available for the selected object.")
    com = np.asarray(object_com_xy, dtype=float)[:2]
    state = np.asarray(object_state, dtype=float)
    goal = np.asarray(object_goal, dtype=float)
    trans_mix, rot_mix, _, _ = _goal_motion_mix(state, goal)
    advice = gemini_advice or {}
    pref_t = set(int(x) for x in advice.get("preferred_translation_primitive_ids", []))
    pref_r = set(int(x) for x in advice.get("preferred_rotation_primitive_ids", []))
    avoid = set(int(x) for x in advice.get("avoid_primitive_ids", []))
    pref_t_types = set(str(x) for x in advice.get("preferred_translation_types", []))
    pref_r_types = set(str(x) for x in advice.get("preferred_rotation_types", []))
    avoid_types = set(str(x) for x in advice.get("avoid_types", []))

    shape_stability = {
        "cuboid": 1.00,
        "cylinder": 0.90,
        "stick": 0.75,
        "ring": 0.70,
        "cone": 0.60,
        "hemisphere": 0.45,
        "sphere": 0.35,
    }

    distances = []
    for r in records:
        c = np.asarray(r["center_world"], dtype=float)[:2]
        distances.append(float(np.linalg.norm(c - com)))
    dmax = max(max(distances), 0.03)

    ranking = []
    for r, d in zip(records, distances):
        pid = int(r["primitive_id"])
        near = 1.0 - min(d / dmax, 1.0)
        far = min(d / dmax, 1.0)
        score = 2.0 * trans_mix * near + 2.0 * rot_mix * far
        score += 0.7 * shape_stability.get(r["primitive_type"], 0.5)
        score += 0.25 * min(float(r.get("pixel_count", 0)) / 1200.0, 1.0)
        if pid in pref_t or r["primitive_type"] in pref_t_types:
            score += 0.65 * trans_mix
        if pid in pref_r or r["primitive_type"] in pref_r_types:
            score += 0.65 * rot_mix
        if pid in avoid or r["primitive_type"] in avoid_types:
            score -= 1.5
        if str(r.get("occupancy", "unknown")).lower() == "hollow" and r["primitive_type"] == "ring":
            score += 0.10
        ranking.append({
            "record": r,
            "primitive_id": pid,
            "primitive_type": r["primitive_type"],
            "center_to_com_m": float(d),
            "translation_near_com_score": float(near),
            "rotation_lever_score": float(far),
            "combined_score": float(score),
        })

    ranking.sort(key=lambda x: x["combined_score"], reverse=True)
    return ranking[0]["record"], ranking


def _analytic_candidate_prior(candidate: dict, object_com_xy, object_state, object_goal, gemini_advice=None):
    com = np.asarray(object_com_xy, dtype=float)[:2]
    state = np.asarray(object_state, dtype=float)
    goal = np.asarray(object_goal, dtype=float)
    c = np.asarray(candidate["contact_world"], dtype=float)[:2]
    d = _norm(candidate["direction_world"])[:2]
    r = c - com
    lever = float(np.linalg.norm(r))
    torque_signed = float(r[0] * d[1] - r[1] * d[0])
    torque_mag = abs(torque_signed)

    trans_mix, rot_mix, _, yaw_error = _goal_motion_mix(state, goal)
    goal_vec = goal[:2] - state[:2]
    if np.linalg.norm(goal_vec) > 1e-8:
        goal_dir = goal_vec / np.linalg.norm(goal_vec)
        alignment = float(np.clip(np.dot(d, goal_dir), -1.0, 1.0))
    else:
        alignment = 0.0

    lever_norm = min(lever / 0.12, 1.0)
    torque_norm = min(torque_mag / 0.10, 1.0)
    # Lower is better. Translation likes aligned, low-torque pushes; rotation likes
    # correct-sign high torque.
    translation_cost = trans_mix * (0.70 * (1.0 - max(alignment, 0.0)) + 0.45 * torque_norm)
    desired_sign = 0.0 if abs(yaw_error) < math.radians(2.0) else math.copysign(1.0, yaw_error)
    torque_sign_good = (desired_sign == 0.0) or (torque_signed * desired_sign > 0.0)
    rotation_cost = rot_mix * (0.70 * (1.0 - lever_norm) + (0.0 if torque_sign_good else 0.80))

    advice = gemini_advice or {}
    hollow_pref = advice.get("hollow_contact_preference", {}) or {}
    key = f"P{int(candidate.get('instance_id', -1))}"
    pref = str(hollow_pref.get(key, "not_applicable")).lower()
    hollow_cost = 0.0
    if candidate["family"] == "inner_rim_push" and pref == "outer":
        hollow_cost += 0.5
    if candidate["family"] == "outer_rim_push" and pref == "inner":
        hollow_cost += 0.5

    return float(translation_cost + rotation_cost + hollow_cost), {
        "lever_m": lever,
        "torque_signed": torque_signed,
        "alignment": alignment,
    }


def score_candidates(
    runtime,
    record,
    goal_state,
    hidden=None,
    top_k=20,
    object_com_xy=None,
    object_state=None,
    object_goal=None,
    gemini_advice=None,
    pusher_radius=0.008,
):
    candidates = generate_push_candidates(record, pusher_radius=pusher_radius)
    if not candidates:
        raise RuntimeError("No primitive-specific push candidates were generated.")
    actions, pred_delta, next_state = predict_candidates(runtime, record, candidates, hidden)
    goal = np.asarray(goal_state, np.float32)
    rule = orientation_rule(record)

    pos_err = np.linalg.norm(next_state[:, :2] - goal[None, :2], axis=1)
    ori_err = np.abs(_orientation_error(next_state[:, 2], goal[2], rule))
    pos_cost = (pos_err / 0.10) ** 2
    ori_cost = 0.5 * (ori_err / math.radians(30.0)) ** 2
    effort_cost = 0.05 * ((actions[:, 5] / max(PUSH_LENGTHS)) ** 2 + 0.25 * (actions[:, 6] / max(PUSH_SPEEDS)) ** 2)
    risk_cost = 0.20 * np.array([RISK_VALUE.get(c.get("risk", "normal"), 0.25) for c in candidates])

    x, y = next_state[:, 0], next_state[:, 1]
    dx_out = np.maximum(WORKSPACE["xmin"] - x, 0) + np.maximum(x - WORKSPACE["xmax"], 0)
    dy_out = np.maximum(WORKSPACE["ymin"] - y, 0) + np.maximum(y - WORKSPACE["ymax"], 0)
    workspace_cost = 50.0 * ((dx_out / 0.02) ** 2 + (dy_out / 0.02) ** 2)
    obstacle_cost = _obstacle_cost(next_state[:, :2], record)

    analytic_cost = np.zeros(len(candidates), dtype=np.float64)
    analytic_meta = []
    if object_com_xy is not None and object_state is not None and object_goal is not None:
        for i, c in enumerate(candidates):
            analytic_cost[i], meta = _analytic_candidate_prior(c, object_com_xy, object_state, object_goal, gemini_advice)
            analytic_meta.append(meta)
    else:
        analytic_meta = [{} for _ in candidates]

    total = pos_cost + ori_cost + effort_cost + risk_cost + workspace_cost + obstacle_cost + 0.40 * analytic_cost

    # Strong bonus for a predicted one-shot solution: this makes one push ideal,
    # while RMPPI remains available for corrections if one push is insufficient.
    one_shot = (pos_err <= 0.020) & (ori_err <= math.radians(7.0))
    total = total - one_shot.astype(np.float64) * 2.0

    order = np.argsort(total)
    top = order[: min(top_k, len(order))]
    result = {
        "record": record,
        "goal_state": goal,
        "rule": rule,
        "candidates": candidates,
        "actions": actions,
        "pred_delta": pred_delta,
        "pred_next": next_state,
        "cost": total.astype(np.float32),
        "obstacle_cost": obstacle_cost.astype(np.float32),
        "analytic_cost": analytic_cost.astype(np.float32),
        "analytic_meta": analytic_meta,
        "one_shot_mask": one_shot,
        "top_indices": top.astype(np.int64),
    }
    np.savez_compressed(
        PLANNING_DIR / "step9_latest.npz",
        actions=actions,
        pred_delta=pred_delta,
        pred_next=next_state,
        cost=total,
        analytic_cost=analytic_cost,
        one_shot_mask=one_shot,
        top_indices=top,
        goal_state=goal,
    )
    return result


def choose_one_shot_if_available(step9: dict):
    feasible = np.where(np.asarray(step9.get("one_shot_mask", []), dtype=bool))[0]
    if len(feasible) == 0:
        return None
    costs = np.asarray(step9["cost"], dtype=float)
    idx = int(feasible[np.argmin(costs[feasible])])
    return {
        "candidate_index": idx,
        "candidate": step9["candidates"][idx],
        "predicted_next": np.asarray(step9["pred_next"][idx], dtype=np.float32),
        "cost": float(costs[idx]),
    }

# =============================================================================
# MULTIBODY-AWARE CONTACT SNAP (overrides base-link-only version)
# =============================================================================

def _union_body_aabb(body_id: int):
    _require_physics_connection()
    lows = []
    highs = []
    for link in [-1] + list(range(p.getNumJoints(int(body_id)))):
        try:
            lo, hi = p.getAABB(int(body_id), int(link))
        except Exception:
            continue
        lows.append(np.asarray(lo, dtype=np.float64))
        highs.append(np.asarray(hi, dtype=np.float64))
    if not lows:
        raise RuntimeError(f"No AABB available for body {body_id}")
    return np.min(np.stack(lows), axis=0), np.max(np.stack(highs), axis=0)


def _snap_contact_to_target_aabb(target_body_id: int, candidate: dict, radius: float):
    """Project perceived contact onto the UNION AABB of the whole rigid multibody."""
    _require_physics_connection()
    d3 = _norm(candidate["direction_world"])
    d = np.asarray(d3[:2], dtype=np.float64)
    d /= np.linalg.norm(d) + 1e-12

    lo, hi = _union_body_aabb(int(target_body_id))
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    support_radius = abs(d[0]) * half[0] + abs(d[1]) * half[1]
    nominal = np.asarray(candidate["contact_world"], dtype=np.float64)
    tangent = np.array([-d[1], d[0]], dtype=np.float64)
    nominal_offset = nominal[:2] - center[:2]
    tangential_offset = float(np.dot(nominal_offset, tangent))
    tangent_limit = abs(tangent[0]) * half[0] + abs(tangent[1]) * half[1]
    tangential_offset = float(np.clip(tangential_offset, -0.85 * tangent_limit, 0.85 * tangent_limit))
    snapped_xy = center[:2] - d * support_radius + tangent * tangential_offset
    zspan = max(float(hi[2] - lo[2]), 1e-3)
    snapped_z = float(np.clip(nominal[2], lo[2] + 0.12 * zspan, hi[2] - 0.12 * zspan))
    touch_center_xy = snapped_xy - float(radius) * d
    return snapped_xy, snapped_z, touch_center_xy, d


# =============================================================================
# FINAL EXECUTION FIX -- ROBUST CONTACT + SYNCHRONIZED FRANKA MOTION
# =============================================================================
# This section intentionally overrides the earlier low-level execution helpers.
# The learned network still chooses the primitive family, push direction, length,
# and speed.  The simulator-specific execution layer only does what a real robot
# controller must do anyway: verify reachability, localize the actual surface,
# move to it, establish contact, then execute the commanded push.


def _candidate_xy_direction(candidate: dict):
    d = np.asarray(candidate["direction_world"], dtype=np.float64).reshape(-1)
    if d.size < 2:
        raise ValueError("candidate direction_world must contain x,y")
    dxy = d[:2]
    n = float(np.linalg.norm(dxy))
    if n < 1e-9:
        raise ValueError("candidate push direction has zero XY norm")
    return dxy / n


def _ray_contact_on_selected_body(
    target_body_id: int,
    candidate: dict,
    radius: float,
    support_z: float,
    pusher_height: float,
):
    """Find a real collision-surface contact on the selected PyBullet body.

    The network/perception contact is treated as a *semantic/geometric proposal*.
    For execution we cast horizontal rays through the selected body while
    preserving the planned push direction and, as much as possible, the planned
    tangential offset.  This avoids the old failure mode where the union AABB
    placed the red pusher in empty space next to cups, rings, and compound bodies.
    """
    _require_physics_connection()
    target_body_id = int(target_body_id)
    d = _candidate_xy_direction(candidate)
    tangent = np.array([-d[1], d[0]], dtype=np.float64)

    lo, hi = _union_body_aabb(target_body_id)
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    span_z = max(float(hi[2] - lo[2]), 1e-3)

    nominal = np.asarray(candidate.get("contact_world", center), dtype=np.float64)
    nominal_t = float(np.dot(nominal[:2] - center[:2], tangent))
    tangent_limit = abs(tangent[0]) * half[0] + abs(tangent[1]) * half[1]
    nominal_t = float(np.clip(nominal_t, -0.82 * tangent_limit, 0.82 * tangent_limit))

    # Ray height is guaranteed to cut through the object's vertical AABB.
    ray_z = float(np.clip(
        nominal[2] if nominal.size >= 3 and np.isfinite(nominal[2]) else center[2],
        lo[2] + 0.18 * span_z,
        hi[2] - 0.18 * span_z,
    ))

    # The vertical pusher itself must stay above the table, but it may straddle a
    # short object such as tape because its collision cylinder extends +/- H/2.
    pusher_z = max(
        float(support_z) + float(pusher_height) / 2.0 + 0.003,
        ray_z,
    )

    ray_extent = float(np.linalg.norm(half[:2]) + 0.18)
    offsets = [
        nominal_t,
        0.75 * nominal_t,
        0.50 * nominal_t,
        0.25 * nominal_t,
        0.0,
        0.30 * tangent_limit,
        -0.30 * tangent_limit,
    ]

    seen = set()
    for t in offsets:
        key = round(float(t), 5)
        if key in seen:
            continue
        seen.add(key)
        line_xy = center[:2] + tangent * float(t)
        ray_from = np.array([*(line_xy - d * ray_extent), ray_z], dtype=np.float64)
        ray_to = np.array([*(line_xy + d * ray_extent), ray_z], dtype=np.float64)
        hit = p.rayTest(ray_from.tolist(), ray_to.tolist())[0]
        hit_body = int(hit[0])
        if hit_body != target_body_id:
            continue
        surface = np.asarray(hit[3], dtype=np.float64)
        if not np.all(np.isfinite(surface)):
            continue

        # Pusher center sits immediately outside the surface on the approach side.
        tool_contact = surface.copy()
        tool_contact[:2] -= d * (float(radius) + 0.0015)
        tool_contact[2] = pusher_z
        return {
            "success": True,
            "surface_world": surface.astype(np.float32),
            "tool_contact_world": tool_contact.astype(np.float32),
            "direction_xy": d.astype(np.float32),
            "ray_height": ray_z,
            "pusher_z": float(pusher_z),
            "tangential_offset_m": float(t),
        }

    return {
        "success": False,
        "reason": (
            "No unobstructed horizontal ray reached the selected body along this "
            "planned push side. The side is empty or blocked by another object."
        ),
    }


def _franka_simple_reach(franka_robot: Optional[FrankaPandaRobot], point_world):
    if franka_robot is None:
        return True
    point = np.asarray(point_world, dtype=np.float64)
    hand = point + np.array([0.0, 0.0, float(franka_robot.hand_above_pusher_m)])
    horizontal = float(np.linalg.norm(hand[:2] - franka_robot.base_position[:2]))
    spatial = float(np.linalg.norm(hand - franka_robot.base_position))
    # Conservative bounds for the standard Panda. The exact IK check follows.
    if horizontal > 0.84 or spatial > 0.90:
        return False
    return bool(franka_robot.is_tool_center_reachable(point))


def prepare_executable_push(
    target_body_id: int,
    candidate: dict,
    franka_robot: Optional[FrankaPandaRobot] = None,
    support_z: float = 0.0,
    radius: float = 0.008,
    pusher_height: float = 0.10,
    approach_clearance: float = 0.055,
):
    """Validate and convert a planned push into a physically executable push.

    Returns a dictionary with `feasible`, a corrected execution candidate, and
    the approach/contact/end tool-center poses. No robot motion is performed here.
    """
    _require_physics_connection()
    c = dict(candidate)
    contact = _ray_contact_on_selected_body(
        int(target_body_id), c, float(radius), float(support_z), float(pusher_height)
    )
    if not contact.get("success", False):
        return {"feasible": False, "reason": contact.get("reason", "surface contact failed")}

    d = np.asarray(contact["direction_xy"], dtype=np.float64)
    tool_contact = np.asarray(contact["tool_contact_world"], dtype=np.float64)
    start = tool_contact.copy()
    start[:2] -= float(approach_clearance) * d
    end = tool_contact.copy()
    end[:2] += float(c["push_length"]) * d

    # A safe hover is checked too because the arm has to reach it before lowering.
    safe = start.copy()
    safe[2] += 0.18
    for label, point in (("safe hover", safe), ("approach start", start), ("contact", tool_contact), ("push end", end)):
        if not _franka_simple_reach(franka_robot, point):
            return {
                "feasible": False,
                "reason": f"Franka cannot reach the {label} pose for this candidate.",
                "failed_pose_world": np.asarray(point, dtype=np.float32).tolist(),
            }

    corrected = dict(c)
    corrected["contact_world"] = np.asarray(contact["surface_world"], dtype=np.float32)
    corrected["execution_tool_contact_world"] = tool_contact.astype(np.float32)
    corrected["execution_start_world"] = start.astype(np.float32)
    corrected["execution_end_world"] = end.astype(np.float32)
    corrected["execution_pusher_z"] = float(contact["pusher_z"])
    corrected["direction_world"] = np.array([d[0], d[1], 0.0], dtype=np.float32)
    corrected["theta_push"] = float(math.atan2(d[1], d[0]))

    return {
        "feasible": True,
        "candidate": corrected,
        "surface_world": np.asarray(contact["surface_world"], dtype=np.float32),
        "tool_contact_world": tool_contact.astype(np.float32),
        "start_world": start.astype(np.float32),
        "end_world": end.astype(np.float32),
    }


def _final_command_tool_center(self, pusher_center_world):
    """Higher-gain continuous command used by the final synchronized executor."""
    if not p.isConnected():
        return False
    q = self._ik_for_tool_center(pusher_center_world)
    if q is None:
        return False
    p.setJointMotorControlArray(
        self.body_id,
        self.arm_joint_indices,
        p.POSITION_CONTROL,
        targetPositions=q.tolist(),
        forces=self.max_forces,
        positionGains=[0.45] * 7,
        velocityGains=[1.0] * 7,
    )
    self._set_fingers(self.finger_opening)
    return True


def _move_tool_center_blocking(
    self,
    target_world,
    dt=1.0 / 240.0,
    realtime=True,
    tolerance=0.010,
    cartesian_step=0.025,
    max_steps_per_waypoint=120,
    progress_callback=None,
    stage="ROBOT MOVE",
):
    """Move through short Cartesian waypoints and WAIT for actual convergence.

    The old executor merely issued IK targets for a fixed number of simulation
    steps and then moved the independent pusher even when the Panda was still far
    away. This method does not advance to the next phase until the actual Panda
    tool center reaches each waypoint (or reports failure).
    """
    target = np.asarray(target_world, dtype=np.float64).reshape(3)
    start = self.current_tool_center()
    distance = float(np.linalg.norm(target - start))
    n_waypoints = max(1, int(math.ceil(distance / max(float(cartesian_step), 1e-3))))

    import time as _time
    for wi in range(1, n_waypoints + 1):
        alpha = wi / n_waypoints
        waypoint = (1.0 - alpha) * start + alpha * target
        if self._ik_for_tool_center(waypoint) is None:
            return False

        reached = False
        for k in range(int(max_steps_per_waypoint)):
            if not self.command_tool_center(waypoint):
                return False
            p.stepSimulation()
            actual = self.current_tool_center()
            err = float(np.linalg.norm(actual - waypoint))
            if progress_callback is not None and k % 8 == 0:
                try:
                    progress_callback(stage, actual)
                except Exception:
                    pass
            if realtime:
                _time.sleep(float(dt))
            if err <= float(tolerance):
                reached = True
                break
        if not reached:
            return False
    return True


# Monkey-patch the existing class so main.py keeps the same public API.
FrankaPandaRobot.command_tool_center = _final_command_tool_center
FrankaPandaRobot.move_tool_center_blocking = _move_tool_center_blocking
FrankaPandaExecutor = FrankaPandaRobot


def _set_pusher_position_target(gantry, xy, force=420.0):
    xy = np.asarray(xy, dtype=np.float64)
    p.setJointMotorControlArray(
        int(gantry),
        [0, 1],
        p.POSITION_CONTROL,
        targetPositions=[float(xy[0]), float(xy[1])],
        targetVelocities=[0.0, 0.0],
        forces=[float(force), float(force)],
        positionGains=[0.65, 0.65],
        velocityGains=[1.0, 1.0],
    )


def _target_contact_exists(gantry: int, target_body_id: int):
    for cp in p.getContactPoints(bodyA=int(gantry), bodyB=int(target_body_id)):
        if int(cp[1]) == int(gantry) and int(cp[2]) == int(target_body_id):
            return True
    return False


def _remove_body_safe(body_id):
    try:
        if body_id is not None and p.isConnected():
            p.removeBody(int(body_id))
    except Exception:
        pass


def _candidate_for_log(candidate: dict):
    out = {}
    for key, value in dict(candidate).items():
        if isinstance(value, np.ndarray):
            out[key] = value.astype(float).tolist()
        elif isinstance(value, (np.floating, np.integer)):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def execute_push_pybullet(
    target_body_id: int,
    candidate: dict,
    dt=1.0 / 240.0,
    radius=0.008,
    support_z=0.0,
    pusher_height=0.10,
    franka_robot: Optional[FrankaPandaRobot] = None,
    progress_callback=None,
    realtime_visualization=True,
    display_every_steps=6,
):
    """Final synchronized executor: reach -> contact -> push -> retract -> settle.

    NO force feedback is used. The recurrent model is not responsible for robot
    IK/contact. The network chooses the push; this controller makes that push
    physically executable and refuses to perform unrelated/random motions.
    """
    _require_physics_connection()
    target_body_id = int(target_body_id)
    preview = prepare_executable_push(
        target_body_id=target_body_id,
        candidate=candidate,
        franka_robot=franka_robot,
        support_z=float(support_z),
        radius=float(radius),
        pusher_height=float(pusher_height),
    )
    if not preview.get("feasible", False):
        return {
            "contact_established": False,
            "actual_travel_m": 0.0,
            "object_displacement_m": 0.0,
            "failure_reason": preview.get("reason", "candidate is not executable"),
        }

    c = preview["candidate"]
    d = _candidate_xy_direction(c)
    start = np.asarray(c["execution_start_world"], dtype=np.float64)
    tool_contact = np.asarray(c["execution_tool_contact_world"], dtype=np.float64)
    push_end = np.asarray(c["execution_end_world"], dtype=np.float64)
    pusher_z = float(c["execution_pusher_z"])
    target_length = float(c["push_length"])
    commanded_speed = max(float(c["push_speed"]), 0.010)

    before_pos, before_q = p.getBasePositionAndOrientation(target_body_id)
    before_pos = np.asarray(before_pos, dtype=np.float64)
    before_yaw = float(p.getEulerFromQuaternion(before_q)[2])

    debug_ids = []
    debug_ids.append(p.addUserDebugLine(start.tolist(), tool_contact.tolist(), [0.1, 0.8, 0.1], 3.0, lifeTime=0))
    debug_ids.append(p.addUserDebugLine(tool_contact.tolist(), push_end.tolist(), [0.95, 0.05, 0.05], 5.0, lifeTime=0))
    debug_ids.append(p.addUserDebugText(
        f"EXECUTE {c['family']}",
        (tool_contact + np.array([0, 0, 0.12])).tolist(),
        [0.9, 0.05, 0.05], 1.1, lifeTime=0,
    ))

    gantry = None
    import time as _time

    def notify(stage):
        if progress_callback is not None:
            try:
                xyz = None if gantry is None else _pusher_xyz(gantry)
                progress_callback(stage, xyz)
            except Exception:
                pass

    try:
        # ------------------------------------------------------------------
        # 1) Move the ACTUAL Panda to a safe hover and then to the start pose.
        #    Nothing physical is pushed until the arm has really converged.
        # ------------------------------------------------------------------
        if franka_robot is not None:
            current = franka_robot.current_tool_center()
            safe_z = max(float(current[2]), float(start[2]) + 0.20)
            lift = np.array([current[0], current[1], safe_z], dtype=np.float64)
            above_start = np.array([start[0], start[1], safe_z], dtype=np.float64)

            for target, stage in (
                (lift, "ROBOT LIFT"),
                (above_start, "ROBOT MOVE ABOVE SELECTED OBJECT"),
                (start, "ROBOT LOWER TO EXECUTION START"),
            ):
                ok = franka_robot.move_tool_center_blocking(
                    target,
                    dt=float(dt),
                    realtime=bool(realtime_visualization),
                    tolerance=0.011,
                    cartesian_step=0.025,
                    max_steps_per_waypoint=140,
                    progress_callback=progress_callback,
                    stage=stage,
                )
                if not ok:
                    return {
                        "contact_established": False,
                        "actual_travel_m": 0.0,
                        "object_displacement_m": 0.0,
                        "failure_reason": f"Panda failed to converge during: {stage}",
                    }

        # Create the red collision pusher exactly at the Panda's reached start.
        gantry = create_cartesian_pusher(
            start_xy=start[:2],
            pusher_z=pusher_z,
            radius=float(radius),
            height=float(pusher_height),
            friction=0.65,
            motor_force=420.0,
            visible=True,
        )
        _set_pusher_position_target(gantry, start[:2])

        for _ in range(20):
            if franka_robot is not None:
                franka_robot.command_tool_center(_pusher_xyz(gantry))
            p.stepSimulation()
            notify("READY AT EXECUTION START")
            if realtime_visualization:
                _time.sleep(float(dt))

        # ------------------------------------------------------------------
        # Shared path follower. The pusher never outruns the Panda visually.
        # ------------------------------------------------------------------
        def move_linear(target_xy, speed, stage, stop_when_contact=False):
            target_xy = np.asarray(target_xy, dtype=np.float64)
            begin = _pusher_xy(gantry)
            vec = target_xy - begin
            dist = float(np.linalg.norm(vec))
            if dist < 1e-8:
                return True, _target_contact_exists(gantry, target_body_id)
            u = vec / dist
            steps = max(1, int(math.ceil(dist / max(float(speed) * float(dt), 1e-5))))
            contact_seen = _target_contact_exists(gantry, target_body_id)

            for k in range(1, steps + 1):
                desired_xy = begin + u * min(dist, k * float(speed) * float(dt))
                _set_pusher_position_target(gantry, desired_xy, force=420.0)

                # If Panda lags, freeze the pusher target and let the arm catch up.
                if franka_robot is not None:
                    for _catch in range(80):
                        actual_pusher = _pusher_xyz(gantry)
                        franka_robot.command_tool_center(actual_pusher)
                        p.stepSimulation()
                        robot_err = float(np.linalg.norm(franka_robot.current_tool_center() - actual_pusher))
                        notify(stage)
                        if realtime_visualization:
                            _time.sleep(float(dt))
                        if robot_err <= 0.018:
                            break
                else:
                    p.stepSimulation()
                    notify(stage)
                    if realtime_visualization:
                        _time.sleep(float(dt))

                if _target_contact_exists(gantry, target_body_id):
                    contact_seen = True
                    if stop_when_contact:
                        _set_pusher_position_target(gantry, _pusher_xy(gantry), force=420.0)
                        return True, True

            return True, contact_seen

        # ------------------------------------------------------------------
        # 2) Horizontal approach. A small controlled extension is allowed only
        #    if ray contact was slightly conservative.
        # ------------------------------------------------------------------
        _, contact_established = move_linear(
            tool_contact[:2], 0.035, "APPROACH SELECTED OBJECT", stop_when_contact=True
        )
        if not contact_established:
            extension = tool_contact[:2] + 0.012 * d
            _, contact_established = move_linear(
                extension, 0.020, "FINAL CONTACT APPROACH", stop_when_contact=True
            )

        if not contact_established:
            # Return home once. Do NOT perform repeated/random attempts.
            if franka_robot is not None:
                _now = franka_robot.current_tool_center()
                franka_robot.move_tool_center_blocking(
                    _now + np.array([0.0, 0.0, 0.16]),
                    dt=float(dt), realtime=bool(realtime_visualization),
                    progress_callback=progress_callback, stage="NO CONTACT - LIFT",
                )
                q0 = franka_robot.current_arm_q()
                for alpha in np.linspace(0.0, 1.0, 180):
                    q = (1.0 - alpha) * q0 + alpha * franka_robot.initial_arm_q
                    p.setJointMotorControlArray(
                        franka_robot.body_id, franka_robot.arm_joint_indices,
                        p.POSITION_CONTROL, targetPositions=q.tolist(),
                        forces=franka_robot.max_forces,
                        positionGains=[0.35] * 7, velocityGains=[1.0] * 7,
                    )
                    p.stepSimulation()
                    if realtime_visualization:
                        _time.sleep(float(dt))
            return {
                "contact_established": False,
                "actual_travel_m": 0.0,
                "object_displacement_m": 0.0,
                "failure_reason": "Surface was reachable but physical target contact was not detected; execution stopped safely.",
                "execution_candidate": _candidate_for_log(c),
            }

        # ------------------------------------------------------------------
        # 3) Execute the network/RMPPI push only AFTER confirmed contact.
        # ------------------------------------------------------------------
        push_start_xy = _pusher_xy(gantry).copy()
        commanded_end_xy = push_start_xy + target_length * d
        move_linear(commanded_end_xy, commanded_speed, "PUSHING SELECTED OBJECT", stop_when_contact=False)
        actual_travel = float(np.dot(_pusher_xy(gantry) - push_start_xy, d))

        # Hold briefly, then retract horizontally.
        _set_pusher_position_target(gantry, _pusher_xy(gantry), force=420.0)
        for _ in range(25):
            if franka_robot is not None:
                franka_robot.command_tool_center(_pusher_xyz(gantry))
            p.stepSimulation()
            notify("PUSH COMPLETE - HOLD")
            if realtime_visualization:
                _time.sleep(float(dt))

        retract_xy = _pusher_xy(gantry) - 0.055 * d
        move_linear(retract_xy, 0.045, "RETRACT TOOL", stop_when_contact=False)

        # Lift and home using actual convergence, not an arbitrary fixed-time jump.
        if franka_robot is not None:
            tool_now = franka_robot.current_tool_center()
            lift_target = tool_now + np.array([0.0, 0.0, 0.18])
            franka_robot.move_tool_center_blocking(
                lift_target, dt=float(dt), realtime=bool(realtime_visualization),
                progress_callback=progress_callback, stage="ROBOT LIFT AFTER PUSH",
            )
            q_start = franka_robot.current_arm_q()
            for alpha in np.linspace(0.0, 1.0, 200):
                q = (1.0 - alpha) * q_start + alpha * franka_robot.initial_arm_q
                p.setJointMotorControlArray(
                    franka_robot.body_id, franka_robot.arm_joint_indices,
                    p.POSITION_CONTROL, targetPositions=q.tolist(),
                    forces=franka_robot.max_forces,
                    positionGains=[0.35] * 7, velocityGains=[1.0] * 7,
                )
                p.stepSimulation()
                if progress_callback is not None and int(alpha * 200) % 10 == 0:
                    try:
                        progress_callback("ROBOT RETURN HOME", None)
                    except Exception:
                        pass
                if realtime_visualization:
                    _time.sleep(float(dt))

        # Let the selected object settle and keep the live XY view updating.
        for k in range(180):
            p.stepSimulation()
            if progress_callback is not None and k % 10 == 0:
                try:
                    progress_callback("OBJECT SETTLING", None)
                except Exception:
                    pass
            if realtime_visualization:
                _time.sleep(float(dt) * 0.30)

        after_pos, after_q = p.getBasePositionAndOrientation(target_body_id)
        after_pos = np.asarray(after_pos, dtype=np.float64)
        after_yaw = float(p.getEulerFromQuaternion(after_q)[2])
        object_displacement = float(np.linalg.norm(after_pos[:2] - before_pos[:2]))

        return {
            "contact_established": True,
            "actual_travel_m": float(actual_travel),
            "commanded_length_m": float(target_length),
            "commanded_speed_mps": float(commanded_speed),
            "object_displacement_m": object_displacement,
            "before_pose_world": [float(before_pos[0]), float(before_pos[1]), float(before_yaw)],
            "after_pose_world": [float(after_pos[0]), float(after_pos[1]), float(after_yaw)],
            "actual_object_delta": [
                float(after_pos[0] - before_pos[0]),
                float(after_pos[1] - before_pos[1]),
                float(math.atan2(math.sin(after_yaw - before_yaw), math.cos(after_yaw - before_yaw))),
            ],
            "execution_candidate": _candidate_for_log(c),
        }

    finally:
        _remove_body_safe(gantry)
        _remove_debug_items(debug_ids)

# =============================================================================
# FINAL V2 EXECUTION OVERRIDES
# Strong first push support, robust re-contact after object motion, and faster
# Franka transit. These definitions intentionally override the earlier helpers.
# =============================================================================

def _selected_body_surface_from_nominal(
    target_body_id: int,
    candidate: dict,
    radius: float,
    support_z: float,
    pusher_height: float,
):
    """Find a real surface on ONLY the selected body using closest-point queries.

    Unlike a global ray cast, nearby clutter cannot hide the selected object from
    this geometric execution query.  The planned primitive contact and push
    direction are preserved as the nominal seed, then snapped to the selected
    body's true collision geometry.  This is recomputed every closed-loop cycle.
    """
    _require_physics_connection()
    target_body_id = int(target_body_id)
    d = _candidate_xy_direction(candidate)
    tangent = np.array([-d[1], d[0]], dtype=np.float64)
    lo, hi = _union_body_aabb(target_body_id)
    center = 0.5 * (lo + hi)
    zspan = max(float(hi[2] - lo[2]), 1e-3)
    nominal = np.asarray(candidate.get("contact_world", center), dtype=np.float64).reshape(-1)
    if nominal.size < 3:
        nominal = np.r_[nominal[:2], center[2]]
    nominal[2] = float(np.clip(nominal[2], lo[2] + 0.12 * zspan, hi[2] - 0.12 * zspan))
    pusher_z = max(float(support_z) + float(pusher_height) / 2.0 + 0.003, float(nominal[2]))

    probe_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.002)
    probe = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=probe_collision,
        baseVisualShapeIndex=-1,
        basePosition=nominal.tolist(),
    )
    try:
        family = str(candidate.get("family", ""))
        base_backoffs = (0.028, 0.045, 0.065, 0.085)
        if family == "inner_rim_push":
            base_backoffs = (0.012, 0.020, 0.030, 0.040)
        tangential_offsets = (0.0, 0.008, -0.008, 0.016, -0.016, 0.025, -0.025)
        z_offsets = (0.0, 0.010, -0.010, 0.020, -0.020)
        best = None
        best_score = float("inf")
        for backoff in base_backoffs:
            for toff in tangential_offsets:
                for zoff in z_offsets:
                    probe_pos = nominal.copy()
                    probe_pos[:2] = nominal[:2] - float(backoff) * d + float(toff) * tangent
                    probe_pos[2] = float(np.clip(nominal[2] + zoff, lo[2] + 0.08*zspan, hi[2] - 0.08*zspan))
                    p.resetBasePositionAndOrientation(probe, probe_pos.tolist(), [0,0,0,1])
                    pts = p.getClosestPoints(probe, target_body_id, distance=0.18)
                    for cp in pts:
                        # cp[6] = position on body B (the selected target)
                        surface = np.asarray(cp[6], dtype=np.float64)
                        v = surface[:2] - probe_pos[:2]
                        vn = float(np.linalg.norm(v))
                        if vn < 1e-9:
                            continue
                        alignment = float(np.dot(v / vn, d))
                        if alignment < 0.25:
                            continue
                        tangential_error = abs(float(np.dot(surface[:2] - nominal[:2], tangent)))
                        height_error = abs(float(surface[2] - nominal[2]))
                        score = vn + 0.8*tangential_error + 0.35*height_error - 0.03*alignment
                        if score < best_score:
                            best_score = score
                            best = surface
        if best is None:
            return {"success": False, "reason": "Could not snap planned contact to selected body's collision surface."}
        tool_contact = np.asarray(best, dtype=np.float64).copy()
        tool_contact[:2] -= d * (float(radius) + 0.001)
        tool_contact[2] = pusher_z
        return {
            "success": True,
            "surface_world": np.asarray(best, dtype=np.float32),
            "tool_contact_world": tool_contact.astype(np.float32),
            "direction_xy": d.astype(np.float32),
            "pusher_z": float(pusher_z),
        }
    finally:
        _remove_body_safe(probe)


def prepare_executable_push(
    target_body_id: int,
    candidate: dict,
    franka_robot: Optional[FrankaPandaRobot] = None,
    support_z: float = 0.0,
    radius: float = 0.008,
    pusher_height: float = 0.10,
    approach_clearance: float = 0.024,
):
    """Localize the TRUE target surface without prematurely rejecting the action.

    Reachability is now advisory.  Exact Panda reach is attempted in the executor
    with relaxed/position-only IK.  This prevents a marginal safe-hover test from
    terminating the entire task before the red pusher has executed anything.
    """
    _require_physics_connection()
    c = dict(candidate)
    contact = _selected_body_surface_from_nominal(
        target_body_id, c, radius, support_z, pusher_height
    )
    if not contact.get("success", False):
        return {"feasible": False, "reason": contact.get("reason", "surface localization failed")}

    d = np.asarray(contact["direction_xy"], dtype=np.float64)
    tool_contact = np.asarray(contact["tool_contact_world"], dtype=np.float64)
    start = tool_contact.copy()
    start[:2] -= float(approach_clearance) * d
    end = tool_contact.copy()
    end[:2] += float(c.get("push_length", 0.04)) * d

    # A low hover is enough because Panda collisions are disabled and the red
    # pusher is the physical contact body.  A high hover was the main source of
    # false "cannot reach safe hover" failures at the far side of the workspace.
    safe = start.copy()
    safe[2] += 0.050

    reach_report = {}
    if franka_robot is not None:
        for label, point in (("safe_hover", safe), ("approach_start", start),
                             ("contact", tool_contact), ("push_end", end)):
            reach_report[label] = bool(_franka_simple_reach(franka_robot, point))
    else:
        reach_report = {"safe_hover": True, "approach_start": True, "contact": True, "push_end": True}

    corrected = dict(c)
    corrected["contact_world"] = np.asarray(contact["surface_world"], dtype=np.float32)
    corrected["execution_tool_contact_world"] = tool_contact.astype(np.float32)
    corrected["execution_start_world"] = start.astype(np.float32)
    corrected["execution_safe_world"] = safe.astype(np.float32)
    corrected["execution_end_world"] = end.astype(np.float32)
    corrected["execution_pusher_z"] = float(contact["pusher_z"])
    corrected["direction_world"] = np.array([d[0], d[1], 0.0], dtype=np.float32)
    corrected["theta_push"] = float(math.atan2(d[1], d[0]))
    corrected["franka_reach_report"] = reach_report

    # Surface geometry is valid, so the physical push is executable.  If the Panda
    # cannot exactly follow one waypoint, execute_push_pybullet uses relaxed IK and
    # finally a clearly reported Cartesian-pusher fallback rather than returning 0
    # executed actions.
    return {
        "feasible": True,
        "candidate": corrected,
        "surface_world": corrected["contact_world"],
        "tool_contact_world": tool_contact.astype(np.float32),
        "start_world": start.astype(np.float32),
        "safe_world": safe.astype(np.float32),
        "end_world": end.astype(np.float32),
        "franka_reach_report": reach_report,
    }


def _fast_move_tool_center_blocking(
    self,
    target_world,
    dt=1.0/240.0,
    realtime=True,
    tolerance=0.015,
    cartesian_step=0.050,
    max_steps_per_waypoint=70,
    progress_callback=None,
    stage="ROBOT MOVE",
):
    target = np.asarray(target_world, dtype=np.float64).reshape(3)
    start = self.current_tool_center()
    distance = float(np.linalg.norm(target-start))
    n_waypoints = max(1, int(math.ceil(distance / max(float(cartesian_step), 1e-3))))
    import time as _time
    for wi in range(1, n_waypoints+1):
        a = wi / n_waypoints
        waypoint = (1-a)*start + a*target
        if self._ik_for_tool_center(waypoint) is None:
            return False
        reached=False
        for k in range(int(max_steps_per_waypoint)):
            if not self.command_tool_center(waypoint):
                return False
            p.stepSimulation()
            actual=self.current_tool_center()
            if progress_callback is not None and k % 6 == 0:
                try: progress_callback(stage, actual)
                except Exception: pass
            if realtime: _time.sleep(float(dt)*0.10)
            if float(np.linalg.norm(actual-waypoint)) <= float(tolerance):
                reached=True; break
        if not reached:
            return False
    return True


def _home_fast(self, dt=1.0/240.0, realtime=True, progress_callback=None):
    if not p.isConnected():
        return
    import time as _time
    q0=self.current_arm_q()
    for i, a in enumerate(np.linspace(0.0,1.0,100)):
        q=(1-a)*q0+a*self.initial_arm_q
        p.setJointMotorControlArray(self.body_id,self.arm_joint_indices,p.POSITION_CONTROL,
                                    targetPositions=q.tolist(),forces=self.max_forces,
                                    positionGains=[0.45]*7,velocityGains=[1.0]*7)
        self._set_fingers(self.finger_opening)
        p.stepSimulation()
        if progress_callback is not None and i%8==0:
            try: progress_callback("FINAL ROBOT HOME", self.current_tool_center())
            except Exception: pass
        if realtime: _time.sleep(float(dt)*0.08)

FrankaPandaRobot.move_tool_center_blocking = _fast_move_tool_center_blocking
FrankaPandaRobot.home_fast = _home_fast


def _disable_pusher_collisions_except_target(gantry: int, target_body_id: int):
    """Simulation-only execution isolation: the pusher acts only on selected object."""
    try:
        body_ids=[p.getBodyUniqueId(i) for i in range(p.getNumBodies())]
    except Exception:
        return
    for other in body_ids:
        if int(other) in (int(gantry), int(target_body_id)):
            continue
        links_b=[-1]+list(range(p.getNumJoints(int(other))))
        for lb in links_b:
            try: p.setCollisionFilterPair(int(gantry),int(other),1,int(lb),enableCollision=0)
            except Exception: pass


def execute_push_pybullet(
    target_body_id: int,
    candidate: dict,
    dt=1.0/240.0,
    radius=0.008,
    support_z=0.0,
    pusher_height=0.10,
    franka_robot: Optional[FrankaPandaRobot]=None,
    progress_callback=None,
    realtime_visualization=True,
    display_every_steps=6,
    return_home=False,
    fast_robot_motion=True,
):
    """Reach selected object, establish contact, execute one push, leave arm ready.

    First/macro push travel is supplied by main.py.  Later calls are short
    corrections.  No force feedback is used.  The pusher is collision-isolated to
    the selected target so unrelated clutter cannot steal the physical action.
    """
    _require_physics_connection()
    preview=prepare_executable_push(target_body_id,candidate,franka_robot,support_z,radius,pusher_height)
    if not preview.get("feasible",False):
        return {"contact_established":False,"actual_travel_m":0.0,"object_displacement_m":0.0,
                "failure_reason":preview.get("reason","candidate infeasible")}
    c=preview["candidate"]
    d=_candidate_xy_direction(c)
    start=np.asarray(c["execution_start_world"],dtype=np.float64)
    contact=np.asarray(c["execution_tool_contact_world"],dtype=np.float64)
    pusher_z=float(c["execution_pusher_z"])
    target_length=float(c["push_length"])
    commanded_speed=max(float(c.get("push_speed",0.05)),0.035)
    before_pos,before_q=p.getBasePositionAndOrientation(int(target_body_id))
    before_pos=np.asarray(before_pos,dtype=np.float64)
    before_yaw=float(p.getEulerFromQuaternion(before_q)[2])
    debug=[]; gantry=None
    import time as _time
    def notify(stage):
        if progress_callback is not None:
            try: progress_callback(stage,None if gantry is None else _pusher_xyz(gantry))
            except Exception: pass
    try:
        # Fast two-stage Panda transit: directly above start, then lower.
        if franka_robot is not None:
            current=franka_robot.current_tool_center()
            safe_z=max(float(current[2]),float(start[2])+0.11)
            above=np.array([start[0],start[1],safe_z],dtype=np.float64)
            ok=franka_robot.move_tool_center_blocking(above,dt=dt,realtime=realtime_visualization,
                    progress_callback=progress_callback,stage="FAST MOVE ABOVE OBJECT")
            if not ok:
                return {"contact_established":False,"actual_travel_m":0.0,"object_displacement_m":0.0,
                        "failure_reason":"Panda could not reach fast hover above selected object."}
            ok=franka_robot.move_tool_center_blocking(start,dt=dt,realtime=realtime_visualization,
                    progress_callback=progress_callback,stage="LOWER TO PUSH START")
            if not ok:
                return {"contact_established":False,"actual_travel_m":0.0,"object_displacement_m":0.0,
                        "failure_reason":"Panda could not reach push start."}

        gantry=create_cartesian_pusher(start[:2],pusher_z,radius,pusher_height,friction=1.20,motor_force=1400.0,visible=True)
        _disable_pusher_collisions_except_target(gantry,target_body_id)
        _set_pusher_position_target(gantry,start[:2],force=1400.0)
        for _ in range(8):
            if franka_robot is not None: franka_robot.command_tool_center(_pusher_xyz(gantry))
            p.stepSimulation(); notify("READY")
            if realtime_visualization: _time.sleep(float(dt)*0.10)

        def move_linear(target_xy,speed,stage,stop_on_contact=False,push_phase=False):
            target_xy=np.asarray(target_xy,dtype=np.float64)
            begin=_pusher_xy(gantry); vec=target_xy-begin; dist=float(np.linalg.norm(vec))
            if dist<1e-8: return _target_contact_exists(gantry,target_body_id)
            u=vec/dist
            # Coarser visible increments make transit much faster while retaining
            # dense enough simulation steps for contact dynamics.
            ds=max(float(speed)*float(dt)*3.0,0.0015)
            steps=max(1,int(math.ceil(dist/ds)))
            contact_seen=_target_contact_exists(gantry,target_body_id)
            for k in range(1,steps+1):
                desired=begin+u*min(dist,k*ds)
                _set_pusher_position_target(gantry,desired,force=1400.0)
                for _sim in range(3 if push_phase else 2):
                    if franka_robot is not None:
                        franka_robot.command_tool_center(np.array([desired[0],desired[1],pusher_z],dtype=np.float64))
                    p.stepSimulation(); notify(stage)
                if realtime_visualization:
                    _time.sleep(float(dt)*(0.30 if push_phase else 0.08))
                if _target_contact_exists(gantry,target_body_id):
                    contact_seen=True
                    if stop_on_contact:
                        _set_pusher_position_target(gantry,_pusher_xy(gantry),force=1400.0)
                        return True
            return contact_seen

        # Re-contact is recomputed from the CURRENT object pose each iteration.
        contact_ok=move_linear(contact[:2],0.080,"FAST APPROACH",stop_on_contact=True,push_phase=False)
        if not contact_ok:
            contact_ok=move_linear(contact[:2]+0.016*d,0.050,"CONTACT EXTENSION",stop_on_contact=True,push_phase=False)
        if not contact_ok:
            return {"contact_established":False,"actual_travel_m":0.0,"object_displacement_m":0.0,
                    "failure_reason":"Selected-object contact was not established after robust surface snap.",
                    "execution_candidate":_candidate_for_log(c)}

        push_start=_pusher_xy(gantry).copy()
        push_end=push_start+target_length*d
        move_linear(push_end,commanded_speed,"PUSHING SELECTED OBJECT",stop_on_contact=False,push_phase=True)
        actual_travel=float(np.dot(_pusher_xy(gantry)-push_start,d))

        # If contact existed but the object barely responded, use ONE small
        # continuation while the tool is already on the same surface.  This avoids
        # wasting an entire extra robot transit on a push that made no measurable
        # progress.  It is displacement-based execution robustness, not force control.
        mid_pos,_mid_q=p.getBasePositionAndOrientation(int(target_body_id))
        mid_pos=np.asarray(mid_pos,dtype=np.float64)
        mid_disp=float(np.linalg.norm(mid_pos[:2]-before_pos[:2]))
        if mid_disp < 0.0025:
            extension=float(np.clip(0.22*target_length,0.012,0.020))
            extension_end=_pusher_xy(gantry)+extension*d
            move_linear(extension_end,max(commanded_speed,0.055),"NO-OP BOOST EXTENSION",False,True)
            actual_travel=float(np.dot(_pusher_xy(gantry)-push_start,d))

        # Brief hold, short retract, then leave robot in a nearby safe hover so
        # the next correction starts quickly rather than from home.
        for _ in range(8):
            if franka_robot is not None: franka_robot.command_tool_center(_pusher_xyz(gantry))
            p.stepSimulation(); notify("PUSH HOLD")
            if realtime_visualization: _time.sleep(float(dt)*0.12)
        retract=_pusher_xy(gantry)-0.032*d
        move_linear(retract,0.090,"SHORT RETRACT",False,False)
        if franka_robot is not None:
            now=franka_robot.current_tool_center()
            hover=np.array([now[0],now[1],max(now[2]+0.10,float(start[2])+0.11)],dtype=np.float64)
            franka_robot.move_tool_center_blocking(hover,dt=dt,realtime=realtime_visualization,
                    progress_callback=progress_callback,stage="READY FOR NEXT CORRECTION")
            if return_home:
                franka_robot.home_fast(dt=dt,realtime=realtime_visualization,progress_callback=progress_callback)

        # Short settle: enough for pose measurement, much faster than old 180-step wait.
        for k in range(80):
            p.stepSimulation()
            if progress_callback is not None and k%8==0:
                try: progress_callback("OBJECT SETTLING",None)
                except Exception: pass
            if realtime_visualization: _time.sleep(float(dt)*0.06)
        after_pos,after_q=p.getBasePositionAndOrientation(int(target_body_id))
        after_pos=np.asarray(after_pos,dtype=np.float64)
        after_yaw=float(p.getEulerFromQuaternion(after_q)[2])
        displacement=float(np.linalg.norm(after_pos[:2]-before_pos[:2]))
        return {
            "contact_established":True,
            "actual_travel_m":float(actual_travel),
            "commanded_length_m":float(target_length),
            "commanded_speed_mps":float(commanded_speed),
            "object_displacement_m":displacement,
            "before_pose_world":[float(before_pos[0]),float(before_pos[1]),float(before_yaw)],
            "after_pose_world":[float(after_pos[0]),float(after_pos[1]),float(after_yaw)],
            "actual_object_delta":[float(after_pos[0]-before_pos[0]),float(after_pos[1]-before_pos[1]),
                float(math.atan2(math.sin(after_yaw-before_yaw),math.cos(after_yaw-before_yaw)))],
            "execution_candidate":_candidate_for_log(c),
        }
    finally:
        _remove_body_safe(gantry)
        _remove_debug_items(debug)


# =============================================================================
# FINAL MONOTONIC-PLANNING OVERRIDES
# =============================================================================

def _angle_abs_error(theta, goal_theta, rule):
    return float(abs(_orientation_error(np.asarray([theta], dtype=np.float64), float(goal_theta), rule)[0]))


def score_candidates(
    runtime,
    record,
    goal_state,
    hidden=None,
    top_k=24,
    object_com_xy=None,
    object_state=None,
    object_goal=None,
    gemini_advice=None,
    pusher_radius=0.008,
):
    """Step-9 scoring with a strong *monotonic progress* term.

    The learned recurrent model remains the forward model.  This override prevents
    the planner from preferring candidates that have a low generic score but move
    sideways, barely move, or increase distance to the requested goal.
    """
    candidates = generate_push_candidates(record, pusher_radius=pusher_radius)
    if not candidates:
        raise RuntimeError("No primitive-specific push candidates were generated.")

    actions, pred_delta, next_state = predict_candidates(runtime, record, candidates, hidden)
    goal = np.asarray(goal_state, np.float32)
    current = np.asarray(record["state_vector"], np.float32)
    rule = orientation_rule(record)

    current_pos_err = float(np.linalg.norm(current[:2] - goal[:2]))
    current_ori_err = _angle_abs_error(float(current[2]), float(goal[2]), rule)
    pos_err = np.linalg.norm(next_state[:, :2] - goal[None, :2], axis=1)
    ori_err = np.abs(_orientation_error(next_state[:, 2], goal[2], rule))
    improvement = current_pos_err - pos_err
    ori_improvement = current_ori_err - ori_err

    # Predicted motion magnitude: penalize candidates whose forward model says they
    # will barely affect the object.  This directly attacks the repeated no-op pushes.
    predicted_motion = np.linalg.norm(pred_delta[:, :2], axis=1)
    no_motion_penalty = 2.5 * np.maximum(0.004 - predicted_motion, 0.0) / 0.004

    # Strong penalty for moving away from goal.  Near the target, allow a candidate
    # that primarily fixes yaw even if position improvement is small.
    progress_penalty = np.zeros(len(candidates), dtype=np.float64)
    for i in range(len(candidates)):
        if current_pos_err > 0.030:
            if improvement[i] < 0.0:
                progress_penalty[i] += 12.0 + 80.0 * abs(float(improvement[i]))
            elif improvement[i] < min(0.004, 0.10 * current_pos_err):
                progress_penalty[i] += 1.8
        else:
            # Final correction: either position or yaw must improve.
            if improvement[i] < 0.001 and ori_improvement[i] < math.radians(1.5):
                progress_penalty[i] += 2.0

    # Goal-direction consistency.  This keeps the selected path smooth and avoids
    # the zig-zag behavior visible in the previous XY trace.
    goal_vec = goal[:2] - current[:2]
    goal_norm = float(np.linalg.norm(goal_vec))
    alignment_penalty = np.zeros(len(candidates), dtype=np.float64)
    if goal_norm > 1e-8:
        ug = goal_vec / goal_norm
        for i, cand in enumerate(candidates):
            d = np.asarray(cand.get("direction_world", [0, 0, 0]), dtype=np.float64)[:2]
            dn = float(np.linalg.norm(d))
            if dn > 1e-8:
                alignment = float(np.dot(d / dn, ug))
                if current_pos_err > 0.035:
                    alignment_penalty[i] = 1.4 * max(0.0, 0.55 - alignment) ** 2
                else:
                    alignment_penalty[i] = 0.35 * max(0.0, 0.20 - alignment) ** 2

    pos_cost = (pos_err / 0.10) ** 2
    ori_cost = 0.55 * (ori_err / math.radians(30.0)) ** 2
    effort_cost = 0.025 * (
        (actions[:, 5] / max(PUSH_LENGTHS)) ** 2
        + 0.20 * (actions[:, 6] / max(PUSH_SPEEDS)) ** 2
    )
    risk_cost = 0.20 * np.array([RISK_VALUE.get(c.get("risk", "normal"), 0.25) for c in candidates])

    x, y = next_state[:, 0], next_state[:, 1]
    dx_out = np.maximum(WORKSPACE["xmin"] - x, 0) + np.maximum(x - WORKSPACE["xmax"], 0)
    dy_out = np.maximum(WORKSPACE["ymin"] - y, 0) + np.maximum(y - WORKSPACE["ymax"], 0)
    workspace_cost = 50.0 * ((dx_out / 0.02) ** 2 + (dy_out / 0.02) ** 2)
    obstacle_cost = _obstacle_cost(next_state[:, :2], record)

    total = (
        pos_cost + ori_cost + effort_cost + risk_cost + workspace_cost + obstacle_cost
        + no_motion_penalty + progress_penalty + alignment_penalty
    )
    order = np.argsort(total)
    top = order[: min(int(top_k), len(order))]

    result = {
        "record": record,
        "goal_state": goal,
        "rule": rule,
        "candidates": candidates,
        "actions": actions,
        "pred_delta": pred_delta,
        "pred_next": next_state,
        "cost": total.astype(np.float32),
        "obstacle_cost": obstacle_cost.astype(np.float32),
        "predicted_motion_m": predicted_motion.astype(np.float32),
        "predicted_position_improvement_m": improvement.astype(np.float32),
        "predicted_orientation_improvement_rad": ori_improvement.astype(np.float32),
        "top_indices": top.astype(np.int64),
    }
    np.savez_compressed(
        PLANNING_DIR / "step9_latest.npz",
        actions=actions,
        pred_delta=pred_delta,
        pred_next=next_state,
        cost=total,
        obstacle_cost=obstacle_cost,
        predicted_motion_m=predicted_motion,
        predicted_position_improvement_m=improvement,
        top_indices=top,
        goal_state=goal,
    )
    return result


def choose_monotonic_correction(step9: dict, previous_direction=None):
    """Choose ONE correction predicted to reduce error monotonically.

    Returns None only when no useful one-step correction exists; main.py may then
    invoke a short-horizon RMPPI fallback.  This is intentionally preferred over
    repeated 4-step RMPPI execution near the goal.
    """
    record = step9["record"]
    goal = np.asarray(step9["goal_state"], dtype=np.float64)
    current = np.asarray(record["state_vector"], dtype=np.float64)
    current_pos_err = float(np.linalg.norm(current[:2] - goal[:2]))
    current_ori_err = _angle_abs_error(float(current[2]), float(goal[2]), step9["rule"])
    prev = None
    if previous_direction is not None:
        prev = np.asarray(previous_direction, dtype=np.float64)[:2]
        if np.linalg.norm(prev) > 1e-8:
            prev = prev / np.linalg.norm(prev)
        else:
            prev = None

    best = None
    best_score = float("inf")
    for idx in [int(i) for i in np.asarray(step9["top_indices"]).reshape(-1)]:
        cand = step9["candidates"][idx]
        nxt = np.asarray(step9["pred_next"][idx], dtype=np.float64)
        pos_err = float(np.linalg.norm(nxt[:2] - goal[:2]))
        ori_err = _angle_abs_error(float(nxt[2]), float(goal[2]), step9["rule"])
        pos_gain = current_pos_err - pos_err
        ori_gain = current_ori_err - ori_err
        pred_motion = float(np.linalg.norm(np.asarray(step9["pred_delta"][idx], dtype=np.float64)[:2]))

        # Require meaningful predicted progress unless we are specifically fixing yaw.
        if current_pos_err > 0.030:
            if pos_gain < max(0.0025, 0.04 * current_pos_err):
                continue
        else:
            if pos_gain < 0.0008 and ori_gain < math.radians(1.2):
                continue
        if pred_motion < 0.0025 and ori_gain < math.radians(1.5):
            continue

        smooth_pen = 0.0
        d = np.asarray(cand.get("direction_world", [0, 0, 0]), dtype=np.float64)[:2]
        if prev is not None and np.linalg.norm(d) > 1e-8:
            d = d / np.linalg.norm(d)
            smooth_pen = 0.45 * (1.0 - float(np.clip(np.dot(prev, d), -1.0, 1.0)))

        score = float(step9["cost"][idx]) + smooth_pen - 6.0 * pos_gain - 0.35 * ori_gain
        if score < best_score:
            best_score = score
            best = {
                "candidate_index": idx,
                "candidate": cand,
                "predicted_next": np.asarray(step9["pred_next"][idx], dtype=np.float32),
                "position_gain_m": float(pos_gain),
                "orientation_gain_rad": float(ori_gain),
                "predicted_motion_m": pred_motion,
            }
    return best


def rmppi_plan(
    runtime: PushModelRuntime,
    step9: dict,
    hidden=None,
    horizon: int = 2,
    num_rollouts: int = 192,
    iterations: int = 3,
    pool_size: int = 28,
    temperature: float = 0.8,
    seed: int = 42,
):
    """Short-horizon RMPPI fallback with explicit push-count/progress penalties.

    RMPPI is now a fallback for cases that genuinely need coupled corrections,
    rather than the default action chooser after the first push.
    """
    rng = np.random.default_rng(seed)
    order = np.argsort(step9["cost"])
    pool = np.asarray(order[: min(int(pool_size), len(order))], dtype=np.int64)
    templates = [_candidate_template(step9["candidates"][int(i)], step9["record"]["state_vector"]) for i in pool]
    one_step = np.asarray(step9["cost"][pool], dtype=np.float64)
    logits = -(one_step - one_step.min()) / max(float(temperature), 1e-6)
    probs0 = np.exp(logits - logits.max()); probs0 /= probs0.sum() + 1e-12
    probs = np.repeat(probs0[None, :], int(horizon), axis=0)

    start_state = np.asarray(step9["record"]["state_vector"], np.float32)
    start_goal_cost = _goal_cost(start_state, step9["goal_state"], step9["rule"])
    best_cost = float("inf"); best_seq = None; best_states = None

    for it in range(int(iterations)):
        sequences = np.stack([
            rng.choice(len(pool), size=int(num_rollouts), replace=True, p=probs[t])
            for t in range(int(horizon))
        ], axis=1)
        costs = np.zeros(int(num_rollouts), dtype=np.float64)
        all_states = []
        for r in range(int(num_rollouts)):
            state = start_state.copy()
            hstate = None if hidden is None else (hidden[0].clone(), hidden[1].clone())
            traj = [state.copy()]
            prev_theta = None
            last_goal_cost = start_goal_cost
            for t in range(int(horizon)):
                cand = _instantiate_template(templates[int(sequences[r, t])], state)
                state, hstate = _single_model_step(runtime, step9["record"], state, cand, hstate)
                gc = _goal_cost(state, step9["goal_state"], step9["rule"])
                # Penalize any predicted step that worsens the task error.
                if gc > last_goal_cost + 0.02:
                    costs[r] += 8.0 * (gc - last_goal_cost)
                costs[r] += 0.22 * gc
                costs[r] += 0.16  # explicit cost per extra push
                costs[r] += 0.03 * (cand["push_length"] / max(PUSH_LENGTHS)) ** 2
                costs[r] += 0.20 * RISK_VALUE.get(cand.get("risk", "normal"), 0.25)
                costs[r] += float(_obstacle_cost(np.asarray(state[:2])[None, :], step9["record"])[0])
                if prev_theta is not None:
                    da = math.atan2(math.sin(cand["theta_push"] - prev_theta), math.cos(cand["theta_push"] - prev_theta))
                    costs[r] += 0.10 * da * da
                prev_theta = cand["theta_push"]
                last_goal_cost = gc
                traj.append(state.copy())
            costs[r] += 2.5 * _goal_cost(state, step9["goal_state"], step9["rule"])
            all_states.append(np.stack(traj))

        imin = int(np.argmin(costs))
        if float(costs[imin]) < best_cost:
            best_cost = float(costs[imin]); best_seq = sequences[imin].copy(); best_states = all_states[imin].copy()
        w = np.exp(-(costs - costs.min()) / max(float(temperature), 1e-6)); w /= w.sum() + 1e-12
        for t in range(int(horizon)):
            freq = np.zeros(len(pool), dtype=np.float64)
            np.add.at(freq, sequences[:, t], w); freq /= freq.sum() + 1e-12
            probs[t] = 0.35 * probs[t] + 0.65 * freq
            probs[t] = np.maximum(probs[t], 1e-4); probs[t] /= probs[t].sum()
        print(f"RMPPI iteration {it+1}/{iterations}: best={costs[imin]:.4f}, global={best_cost:.4f}")

    global_indices = pool[best_seq]
    first_global = int(global_indices[0])
    return {
        "best_cost": best_cost,
        "best_step9_indices": global_indices,
        "best_families": [step9["candidates"][int(i)]["family"] for i in global_indices],
        "predicted_trajectory": best_states,
        "first_candidate_index": first_global,
        "first_candidate": step9["candidates"][first_global],
    }

# =============================================================================
# FINAL GOAL-TRACKED EXECUTION OVERRIDES
# =============================================================================

def retarget_candidate_to_goal(
    candidate: dict,
    target_body_id: int,
    goal_pose_world,
    primitive_center_world=None,
    activate_above_m: float = 0.020,
):
    """Constrain far-from-goal translation to point directly at the object goal.

    The recurrent network / Step-9 still decides the primitive and push family.
    This final execution layer only replaces the planar direction while the
    *whole rigid object* is still meaningfully far from its requested XY goal.
    Near the goal, the candidate is left untouched so orientation/offset pushes
    can be used for yaw correction.
    """
    _require_physics_connection()
    out = dict(candidate)
    goal = np.asarray(goal_pose_world, dtype=np.float64).reshape(-1)
    if goal.size < 2:
        return out

    body_pos, _body_q = p.getBasePositionAndOrientation(int(target_body_id))
    body_pos = np.asarray(body_pos, dtype=np.float64)
    delta = goal[:2] - body_pos[:2]
    dist = float(np.linalg.norm(delta))
    if dist <= float(activate_above_m):
        out["goal_tracking_active"] = False
        return out

    d = delta / max(dist, 1e-12)

    # Preserve the chosen primitive as the contact prior when possible.  Move the
    # nominal contact to the "back" side of that primitive relative to the goal
    # direction; prepare_executable_push() then snaps this hint to the *actual*
    # collision surface of the selected rigid body.
    old_contact = np.asarray(out.get("contact_world", body_pos), dtype=np.float64)
    if primitive_center_world is not None:
        pc = np.asarray(primitive_center_world, dtype=np.float64).reshape(-1)
        center = np.array([
            float(pc[0]), float(pc[1]),
            float(pc[2]) if pc.size >= 3 else float(old_contact[2])
        ], dtype=np.float64)
    else:
        lo, hi = p.getAABB(int(target_body_id), -1)
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        center = 0.5 * (lo + hi)

    radial = float(np.linalg.norm(old_contact[:2] - center[:2]))
    if not np.isfinite(radial) or radial < 0.010:
        # Conservative local contact scale; the surface-snap stage will refine it.
        radial = 0.030
    radial = float(np.clip(radial, 0.012, 0.070))

    new_contact = old_contact.copy()
    new_contact[:2] = center[:2] - radial * d
    new_contact[2] = float(old_contact[2])

    out["contact_world"] = new_contact.astype(np.float32)
    out["direction_world"] = np.array([d[0], d[1], 0.0], dtype=np.float32)
    out["theta_push"] = float(math.atan2(d[1], d[0]))
    out["goal_tracking_active"] = True
    out["goal_tracking_distance_before_m"] = dist
    return out


def execute_push_pybullet(
    target_body_id: int,
    candidate: dict,
    dt=1.0/240.0,
    radius=0.008,
    support_z=0.0,
    pusher_height=0.10,
    franka_robot: Optional[FrankaPandaRobot]=None,
    progress_callback=None,
    realtime_visualization=True,
    display_every_steps=6,
    return_home=False,
    fast_robot_motion=True,
    goal_pose_world=None,
    goal_position_tolerance_m: float = 0.015,
    abort_if_goal_error_worsens: bool = True,
    force_feedback_enabled: bool = True,
    force_config: Optional[ForceFeedbackConfig] = None,
    object_com_world=None,
    object_inertia_zz_kgm2: Optional[float] = None,
):
    """Goal-tracked Cartesian push with Franka visualization.

    Key behavior:
      * Network chooses primitive/family; far from goal, main.py constrains the
        push direction toward the requested whole-object XY goal.
      * The physical pusher monitors the *actual PyBullet object pose while the
        push is happening* and stops as soon as the goal neighborhood is reached.
      * If a push starts increasing goal distance after previously improving it,
        it stops early instead of drawing a long wandering path.
      * Cartesian force feedback is active after contact when enabled.
    """
    _require_physics_connection()
    preview = prepare_executable_push(
        target_body_id, candidate, franka_robot,
        support_z, radius, pusher_height
    )
    if not preview.get("feasible", False):
        return {
            "contact_established": False,
            "actual_travel_m": 0.0,
            "object_displacement_m": 0.0,
            "failure_reason": preview.get("reason", "candidate infeasible"),
            "goal_stop_reason": None,
        }

    c = preview["candidate"]
    d = _candidate_xy_direction(c)
    start = np.asarray(c["execution_start_world"], dtype=np.float64)
    contact = np.asarray(c["execution_tool_contact_world"], dtype=np.float64)
    pusher_z = float(c["execution_pusher_z"])
    target_length = max(float(c.get("push_length", 0.04)), 0.005)
    commanded_speed = max(float(c.get("push_speed", 0.05)), 0.035)

    # Inertia-aware execution diagnostics.  The force controller still regulates
    # contact force; inertia is used to interpret the same force as torque/
    # angular-acceleration authority about the WHOLE-object COM.
    com_for_inertia = None
    if object_com_world is not None:
        arr = np.asarray(object_com_world, dtype=np.float64).reshape(-1)
        if arr.size >= 2 and np.all(np.isfinite(arr[:2])):
            com_for_inertia = arr
    inertia_zz = None
    if object_inertia_zz_kgm2 is not None:
        try:
            val = float(object_inertia_zz_kgm2)
            if math.isfinite(val) and val > 1e-9:
                inertia_zz = val
        except Exception:
            inertia_zz = None

    signed_moment_arm = 0.0
    lever_arm = 0.0
    if com_for_inertia is not None:
        rr = contact[:2] - com_for_inertia[:2]
        signed_moment_arm = float(rr[0] * d[1] - rr[1] * d[0])
        lever_arm = float(np.linalg.norm(rr))
    torque_sign = 1.0 if signed_moment_arm >= 0.0 else -1.0

    before_pos, before_q = p.getBasePositionAndOrientation(int(target_body_id))
    before_pos = np.asarray(before_pos, dtype=np.float64)
    before_yaw = float(p.getEulerFromQuaternion(before_q)[2])

    goal = None
    if goal_pose_world is not None:
        arr = np.asarray(goal_pose_world, dtype=np.float64).reshape(-1)
        if arr.size >= 2 and np.all(np.isfinite(arr[:2])):
            goal = arr

    def object_goal_error_xy():
        if goal is None:
            return None
        pos, _q = p.getBasePositionAndOrientation(int(target_body_id))
        pos = np.asarray(pos, dtype=np.float64)
        return float(np.linalg.norm(pos[:2] - goal[:2]))

    initial_goal_error = object_goal_error_xy()
    best_goal_error = initial_goal_error
    best_goal_pose = before_pos.copy()
    goal_stop_reason = None

    debug = []
    gantry = None
    import time as _time

    def notify(stage):
        if progress_callback is not None:
            try:
                progress_callback(stage, None if gantry is None else _pusher_xyz(gantry))
            except Exception:
                pass

    def sim_sleep(mult=0.08):
        if realtime_visualization:
            _time.sleep(float(dt) * float(mult))

    try:
        # ------------------------------------------------------------------
        # 1) FAST FRANKA TRANSIT: above the *current* contact start, then down.
        # ------------------------------------------------------------------
        franka_follow_enabled = franka_robot is not None
        franka_fallback_reason = None
        if franka_robot is not None:
            current = franka_robot.current_tool_center()
            # Keep hover LOW.  High hover was outside the Panda sphere even when
            # the actual contact itself was reachable.
            safe_z = max(float(start[2]) + 0.045, min(float(current[2]), float(start[2]) + 0.075))
            above = np.array([start[0], start[1], safe_z], dtype=np.float64)
            ok = franka_robot.move_tool_center_blocking(
                above, dt=dt, realtime=realtime_visualization,
                tolerance=0.025, cartesian_step=0.080, max_steps_per_waypoint=70,
                progress_callback=progress_callback, stage="FAST LOW HOVER"
            )
            if not ok:
                # Try direct start; robust IK now includes position-only fallback.
                ok = franka_robot.move_tool_center_blocking(
                    start, dt=dt, realtime=realtime_visualization,
                    tolerance=0.028, cartesian_step=0.070, max_steps_per_waypoint=80,
                    progress_callback=progress_callback, stage="DIRECT REACH RECOVERY"
                )
            else:
                ok = franka_robot.move_tool_center_blocking(
                    start, dt=dt, realtime=realtime_visualization,
                    tolerance=0.022, cartesian_step=0.060, max_steps_per_waypoint=75,
                    progress_callback=progress_callback, stage="LOWER TO PUSH START"
                )

            if not ok:
                # Do NOT terminate the manipulation before any physical action.
                # The trained physics/planner is based on the red Cartesian pusher,
                # so that pusher remains a valid simulator execution fallback.
                franka_follow_enabled = False
                franka_fallback_reason = "PANDA_IK_WORKSPACE_FALLBACK_TO_CARTESIAN_PUSHER"
                print("WARNING: exact Panda IK is marginal/unreachable for this contact.")
                print("Executing the same planned red Cartesian pusher action instead of exiting with 0 pushes.")

        gantry = create_cartesian_pusher(
            start[:2], pusher_z, radius, pusher_height,
            friction=1.25, motor_force=1600.0, visible=True
        )
        _disable_pusher_collisions_except_target(gantry, target_body_id)
        _set_pusher_position_target(gantry, start[:2], force=1600.0)
        for _ in range(5):
            if franka_follow_enabled:
                franka_robot.command_tool_center(_pusher_xyz(gantry))
            p.stepSimulation(); notify("READY"); sim_sleep(0.05)

        # ------------------------------------------------------------------
        # 2) ROBUST APPROACH TO TRUE SELECTED-BODY COLLISION SURFACE.
        # ------------------------------------------------------------------
        def move_to_xy(target_xy, speed, stage, stop_on_contact=False):
            target_xy = np.asarray(target_xy, dtype=np.float64)
            begin = _pusher_xy(gantry)
            vec = target_xy - begin
            dist = float(np.linalg.norm(vec))
            if dist < 1e-8:
                return _target_contact_exists(gantry, target_body_id)
            u = vec / dist
            ds = max(float(speed) * float(dt) * 4.0, 0.0020)
            steps = max(1, int(math.ceil(dist / ds)))
            seen = _target_contact_exists(gantry, target_body_id)
            for k in range(1, steps + 1):
                desired = begin + u * min(dist, k * ds)
                _set_pusher_position_target(gantry, desired, force=1600.0)
                for _sim in range(2):
                    if franka_follow_enabled:
                        franka_robot.command_tool_center(
                            np.array([desired[0], desired[1], pusher_z], dtype=np.float64)
                        )
                    p.stepSimulation(); notify(stage)
                sim_sleep(0.045)
                if _target_contact_exists(gantry, target_body_id):
                    seen = True
                    if stop_on_contact:
                        _set_pusher_position_target(gantry, _pusher_xy(gantry), force=1600.0)
                        return True
            return seen

        contact_ok = move_to_xy(contact[:2], 0.11, "APPROACH TRUE SURFACE", True)
        if not contact_ok:
            contact_ok = move_to_xy(contact[:2] + 0.020 * d, 0.075, "CONTACT EXTENSION", True)
        if not contact_ok:
            return {
                "contact_established": False,
                "actual_travel_m": 0.0,
                "object_displacement_m": 0.0,
                "failure_reason": "Actual selected-object contact was not established.",
                "execution_candidate": _candidate_for_log(c),
                "goal_stop_reason": None,
            }

        # ------------------------------------------------------------------
        # 3) GOAL-TRACKED PUSH + CARTESIAN FORCE FEEDBACK.
        # ------------------------------------------------------------------
        push_start = _pusher_xy(gantry).copy()
        worsening_count = 0
        min_travel_before_abort = min(0.018, 0.25 * target_length)

        force_controller = None
        force_log_path = None
        max_force_seen = 0.0
        force_sum = 0.0
        force_samples = 0
        max_abs_torque_seen = 0.0
        max_abs_alpha_seen = 0.0
        if force_feedback_enabled:
            force_controller = CartesianPushForceController(
                config=(force_config or ForceFeedbackConfig()),
                log_directory=(PLANNING_DIR / "force_logs"),
            )

        # Dynamic path progress.  The learned/planned direction and maximum travel
        # are unchanged; force feedback only changes how quickly we advance along
        # that path after physical contact.
        control_dt = max(2.0 * float(dt), 1e-4)
        commanded_path_s = 0.0
        nominal_for_controller = min(commanded_speed, 0.075)
        estimated_runtime = target_length / max(0.015, 0.35 * nominal_for_controller)
        max_iterations = max(120, int(math.ceil(min(12.0, max(3.0, 2.2 * estimated_runtime)) / control_dt)))

        for k in range(max_iterations):
            if force_controller is not None:
                force_meas = measure_push_force_pybullet(gantry, target_body_id, d)
                measured_push_force = float(force_meas["push_axis_force_n"])
                ff_state = force_controller.update(
                    measured_push_force,
                    dt=control_dt,
                    nominal_speed_mps=nominal_for_controller,
                    lever_arm_m=abs(signed_moment_arm),
                    torque_sign=torque_sign,
                    inertia_zz_kgm2=inertia_zz,
                )
                active_speed = float(ff_state.commanded_speed_mps)
                max_force_seen = max(max_force_seen, ff_state.filtered_force_n)
                force_sum += ff_state.filtered_force_n
                force_samples += 1
                max_abs_torque_seen = max(max_abs_torque_seen, abs(float(ff_state.torque_z_nm)))
                max_abs_alpha_seen = max(max_abs_alpha_seen, abs(float(ff_state.angular_accel_est_rad_s2)))

                if ff_state.hard_stop:
                    goal_stop_reason = "FORCE_HARD_LIMIT"
                    _set_pusher_position_target(gantry, _pusher_xy(gantry), force=1600.0)
                    notify(
                        f"FORCE HARD STOP {ff_state.filtered_force_n:.1f} N "
                        f">= {ff_state.reference_force_n:.1f} N ref"
                    )
                    break
            else:
                active_speed = commanded_speed
                ff_state = None

            commanded_path_s = min(
                target_length,
                commanded_path_s + max(active_speed, 0.0) * control_dt,
            )
            desired_xy = push_start + d * commanded_path_s
            _set_pusher_position_target(gantry, desired_xy, force=1600.0)

            for _sim in range(2):
                if franka_follow_enabled:
                    franka_robot.command_tool_center(
                        np.array([desired_xy[0], desired_xy[1], pusher_z], dtype=np.float64)
                    )
                p.stepSimulation()

                if ff_state is not None:
                    notify(
                        f"FORCE PUSH {ff_state.filtered_force_n:.1f}/"
                        f"{ff_state.reference_force_n:.1f} N"
                    )
                else:
                    notify("GOAL-TRACKED PUSH")

            sim_sleep(0.12)

            current_travel = float(np.dot(_pusher_xy(gantry) - push_start, d))
            current_error = object_goal_error_xy()
            if current_error is not None:
                if best_goal_error is None or current_error < best_goal_error:
                    best_goal_error = current_error
                    pos_now, _q_now = p.getBasePositionAndOrientation(int(target_body_id))
                    best_goal_pose = np.asarray(pos_now, dtype=np.float64)
                    worsening_count = 0
                elif (
                    abort_if_goal_error_worsens
                    and current_travel >= min_travel_before_abort
                    and current_error > float(best_goal_error) + 0.004
                ):
                    worsening_count += 1
                else:
                    worsening_count = max(0, worsening_count - 1)

                if current_error <= float(goal_position_tolerance_m):
                    goal_stop_reason = "GOAL_XY_REACHED_DURING_PUSH"
                    break

                if worsening_count >= 4:
                    goal_stop_reason = "ABORTED_WHEN_GOAL_ERROR_STARTED_WORSENING"
                    break

            if commanded_path_s >= target_length - 1e-6:
                break

        actual_travel = float(np.dot(_pusher_xy(gantry) - push_start, d))
        _set_pusher_position_target(gantry, _pusher_xy(gantry), force=1600.0)

        # Paper-inspired end-of-push force smoothing: ramp the reference toward
        # zero over T_D while backing away only a few millimeters.  This reduces
        # the abrupt contact release before the normal retract phase.
        if force_controller is not None and force_controller.contact_triggered:
            cfg_ff = force_controller.config
            smooth_steps = max(1, int(math.ceil(float(cfg_ff.smoothing_time_s) / float(dt))))
            smooth_start = _pusher_xy(gantry).copy()
            for j in range(smooth_steps):
                elapsed = (j + 1) * float(dt)
                fref = force_controller.smoothing_reference(elapsed)
                meas = measure_push_force_pybullet(gantry, target_body_id, d)
                force_controller.update(
                    float(meas["push_axis_force_n"]),
                    dt=float(dt),
                    nominal_speed_mps=0.0,
                    reference_force_n=fref,
                    lever_arm_m=abs(signed_moment_arm),
                    torque_sign=torque_sign,
                    inertia_zz_kgm2=inertia_zz,
                )
                frac = float(j + 1) / float(smooth_steps)
                smooth_xy = smooth_start - d * float(cfg_ff.smoothing_backoff_m) * frac
                _set_pusher_position_target(gantry, smooth_xy, force=1600.0)
                if franka_follow_enabled:
                    franka_robot.command_tool_center(
                        np.array([smooth_xy[0], smooth_xy[1], pusher_z], dtype=np.float64)
                    )
                p.stepSimulation()
                if j % 4 == 0:
                    notify(f"FORCE SMOOTHING ref={fref:.1f} N")
                sim_sleep(0.03)

            force_log_path = force_controller.save_csv(
                prefix=f"body{int(target_body_id)}_force"
            )

        # If the push made virtually no movement, use one short continuation while
        # already in contact.  The continuation is still stopped by the same goal
        # logic on the next loop; no force feedback is involved.
        mid_pos, _mid_q = p.getBasePositionAndOrientation(int(target_body_id))
        mid_pos = np.asarray(mid_pos, dtype=np.float64)
        mid_disp = float(np.linalg.norm(mid_pos[:2] - before_pos[:2]))
        if mid_disp < 0.0025 and goal_stop_reason is None:
            extension = float(np.clip(0.18 * target_length, 0.010, 0.018))
            ext_end = _pusher_xy(gantry) + extension * d
            move_to_xy(ext_end, max(commanded_speed, 0.060), "ONE NO-OP EXTENSION", False)
            actual_travel = float(np.dot(_pusher_xy(gantry) - push_start, d))

        # ------------------------------------------------------------------
        # 4) SHORT RETRACT + NEARBY HOVER; do not waste time going home.
        # ------------------------------------------------------------------
        retract = _pusher_xy(gantry) - 0.030 * d
        move_to_xy(retract, 0.11, "SHORT RETRACT", False)

        if franka_follow_enabled:
            now = franka_robot.current_tool_center()
            hover = np.array([
                now[0], now[1],
                max(now[2] + 0.09, float(start[2]) + 0.10)
            ], dtype=np.float64)
            franka_robot.move_tool_center_blocking(
                hover, dt=dt, realtime=realtime_visualization,
                tolerance=0.020, cartesian_step=0.065, max_steps_per_waypoint=50,
                progress_callback=progress_callback, stage="READY FOR CORRECTION"
            )
            if return_home:
                franka_robot.home_fast(
                    dt=dt, realtime=realtime_visualization,
                    progress_callback=progress_callback
                )

        # Brief settle only.
        for k in range(70):
            p.stepSimulation()
            if progress_callback is not None and k % 10 == 0:
                try:
                    progress_callback("OBJECT SETTLING", None)
                except Exception:
                    pass
            sim_sleep(0.035)

        after_pos, after_q = p.getBasePositionAndOrientation(int(target_body_id))
        after_pos = np.asarray(after_pos, dtype=np.float64)
        after_yaw = float(p.getEulerFromQuaternion(after_q)[2])
        displacement = float(np.linalg.norm(after_pos[:2] - before_pos[:2]))
        final_goal_error = object_goal_error_xy()

        return {
            "contact_established": True,
            "actual_travel_m": float(actual_travel),
            "commanded_length_m": float(target_length),
            "commanded_speed_mps": float(commanded_speed),
            "object_displacement_m": displacement,
            "before_pose_world": [float(before_pos[0]), float(before_pos[1]), float(before_yaw)],
            "after_pose_world": [float(after_pos[0]), float(after_pos[1]), float(after_yaw)],
            "actual_object_delta": [
                float(after_pos[0] - before_pos[0]),
                float(after_pos[1] - before_pos[1]),
                float(math.atan2(math.sin(after_yaw - before_yaw), math.cos(after_yaw - before_yaw))),
            ],
            "execution_candidate": _candidate_for_log(c),
            "goal_tracking_active": bool(c.get("goal_tracking_active", False)),
            "goal_error_before_m": None if initial_goal_error is None else float(initial_goal_error),
            "best_goal_error_during_push_m": None if best_goal_error is None else float(best_goal_error),
            "goal_error_after_m": None if final_goal_error is None else float(final_goal_error),
            "goal_stop_reason": goal_stop_reason,
            "force_feedback_enabled": bool(force_feedback_enabled),
            "max_push_force_n": float(max_force_seen),
            "mean_push_force_n": float(force_sum / max(force_samples, 1)),
            "force_log_path": None if force_log_path is None else str(force_log_path),
            "whole_object_inertia_zz_kgm2": None if inertia_zz is None else float(inertia_zz),
            "contact_lever_arm_m": float(lever_arm),
            "signed_moment_arm_m": float(signed_moment_arm),
            "max_abs_torque_z_nm": float(max_abs_torque_seen),
            "max_abs_angular_accel_est_rad_s2": float(max_abs_alpha_seen),
            "franka_follow_enabled": bool(franka_follow_enabled),
            "franka_fallback_reason": franka_fallback_reason,
            "failure_reason": None,
        }
    finally:
        _remove_body_safe(gantry)
        _remove_debug_items(debug)

# =============================================================================
# FINAL INERTIA-AWARE PLANNING OVERRIDES
# =============================================================================

def _safe_inertia_zz(value, fallback=1e-4):
    try:
        v = float(value)
        if math.isfinite(v) and v > 1e-9:
            return v
    except Exception:
        pass
    return float(fallback)


def _safe_mass(value, fallback=0.25):
    try:
        v = float(value)
        if math.isfinite(v) and v > 1e-6:
            return v
    except Exception:
        pass
    return float(fallback)


def _inertia_candidate_prior(
    candidate: dict,
    object_com_xy,
    object_state,
    object_goal,
    object_inertia_zz_kgm2=None,
    object_mass_kg=None,
    planning_force_n: float = 6.0,
):
    """Analytic force/moment prior used *in addition* to the learned model.

    tau_z = (r x F)_z, alpha_z ~= tau_z / Izz.
    Translation authority is represented by a ~= F/m.  These are deliberately
    soft priors because planar friction/contact still comes from the recurrent
    model and the real closed-loop execution.
    """
    com = np.asarray(object_com_xy, dtype=np.float64)[:2]
    state = np.asarray(object_state, dtype=np.float64)
    goal = np.asarray(object_goal, dtype=np.float64)
    contact = np.asarray(candidate.get("contact_world", [0, 0, 0]), dtype=np.float64)[:2]
    d = np.asarray(candidate.get("direction_world", [1, 0, 0]), dtype=np.float64)[:2]
    dn = float(np.linalg.norm(d))
    d = d / max(dn, 1e-12)
    r = contact - com
    signed_arm = float(r[0] * d[1] - r[1] * d[0])
    lever = float(np.linalg.norm(r))
    F = max(float(planning_force_n), 0.1)
    Izz = _safe_inertia_zz(object_inertia_zz_kgm2)
    mass = _safe_mass(object_mass_kg)
    tau = F * signed_arm
    alpha = tau / Izz
    translational_accel = F / mass

    trans_mix, rot_mix, _dist, yaw_error = _goal_motion_mix(state, goal)
    goal_vec = goal[:2] - state[:2]
    alignment = 0.0
    if np.linalg.norm(goal_vec) > 1e-9:
        alignment = float(np.dot(d, goal_vec / np.linalg.norm(goal_vec)))

    # Required angular authority to remove the requested yaw in about 0.7 s.
    # This is a ranking prior only; it is not integrated as a rigid-body simulator.
    Trot = 0.70
    alpha_req = 2.0 * abs(float(yaw_error)) / max(Trot * Trot, 1e-6)
    authority_ratio = min(abs(alpha) / max(alpha_req, 0.5), 2.0)
    desired_sign = 0.0 if abs(yaw_error) < math.radians(2.0) else math.copysign(1.0, yaw_error)
    sign_good = desired_sign == 0.0 or tau * desired_sign > 0.0

    # Translation should align with the goal and avoid unnecessary moment;
    # rotation should create the correct moment with enough alpha authority.
    translation_cost = trans_mix * (
        0.70 * (1.0 - max(alignment, 0.0))
        + 0.35 * min(abs(tau) / max(F * 0.08, 1e-9), 1.0)
    )
    rotation_cost = rot_mix * (
        0.75 * max(0.0, 1.0 - authority_ratio)
        + (0.0 if sign_good else 1.25)
    )
    return float(translation_cost + rotation_cost), {
        "lever_arm_m": lever,
        "signed_moment_arm_m": signed_arm,
        "torque_z_nm": float(tau),
        "angular_accel_est_rad_s2": float(alpha),
        "translation_accel_est_m_s2": float(translational_accel),
        "rotation_authority_ratio": float(authority_ratio),
        "rotation_sign_good": bool(sign_good),
    }


def rank_primitives_for_goal(
    records: List[dict],
    object_com_xy,
    object_state,
    object_goal,
    gemini_advice: Optional[dict] = None,
    object_inertia_zz_kgm2=None,
    object_mass_kg=None,
    planning_force_n: float = 6.0,
):
    """Primitive choice using COM distance + whole-object inertia.

    High-Izz objects need a larger moment arm for the same desired yaw change;
    low-Izz objects can rotate from smaller offsets.  Translation still favors
    contacts near the whole-object COM.
    """
    if not records:
        raise RuntimeError("No MR-Former primitives are available for the selected object.")
    com = np.asarray(object_com_xy, dtype=float)[:2]
    state = np.asarray(object_state, dtype=float)
    goal = np.asarray(object_goal, dtype=float)
    trans_mix, rot_mix, _, yaw_error = _goal_motion_mix(state, goal)
    Izz = _safe_inertia_zz(object_inertia_zz_kgm2)
    F = max(float(planning_force_n), 0.1)

    # Required moment arm proxy for a 0.7-s rotational correction.
    alpha_req = 2.0 * abs(float(yaw_error)) / (0.70 ** 2)
    required_arm = Izz * alpha_req / F if abs(yaw_error) > math.radians(2.0) else 0.0
    required_arm = float(np.clip(required_arm, 0.008, 0.18)) if required_arm > 0 else 0.008

    advice = gemini_advice or {}
    pref_t = set(int(x) for x in advice.get("preferred_translation_primitive_ids", []))
    pref_r = set(int(x) for x in advice.get("preferred_rotation_primitive_ids", []))
    avoid = set(int(x) for x in advice.get("avoid_primitive_ids", []))
    pref_t_types = set(str(x) for x in advice.get("preferred_translation_types", []))
    pref_r_types = set(str(x) for x in advice.get("preferred_rotation_types", []))
    avoid_types = set(str(x) for x in advice.get("avoid_types", []))
    shape_stability = {
        "cuboid": 1.00, "cylinder": 0.90, "stick": 0.75, "ring": 0.70,
        "cone": 0.60, "hemisphere": 0.45, "sphere": 0.35,
    }

    distances = [float(np.linalg.norm(np.asarray(r["center_world"], float)[:2] - com)) for r in records]
    dmax = max(max(distances), 0.03)
    ranking = []
    for r, dist in zip(records, distances):
        pid = int(r["primitive_id"])
        near = 1.0 - min(dist / dmax, 1.0)
        far = min(dist / dmax, 1.0)
        inertia_rotation_authority = min(dist / max(required_arm, 1e-6), 1.0)
        score = 2.1 * trans_mix * near
        score += 1.25 * rot_mix * far + 1.15 * rot_mix * inertia_rotation_authority
        score += 0.65 * shape_stability.get(r["primitive_type"], 0.5)
        score += 0.22 * min(float(r.get("pixel_count", 0)) / 1200.0, 1.0)
        if pid in pref_t or r["primitive_type"] in pref_t_types:
            score += 0.60 * trans_mix
        if pid in pref_r or r["primitive_type"] in pref_r_types:
            score += 0.60 * rot_mix
        if pid in avoid or r["primitive_type"] in avoid_types:
            score -= 1.5
        if str(r.get("occupancy", "unknown")).lower() == "hollow" and r["primitive_type"] == "ring":
            score += 0.10
        ranking.append({
            "record": r,
            "primitive_id": pid,
            "primitive_type": r["primitive_type"],
            "center_to_com_m": float(dist),
            "translation_near_com_score": float(near),
            "rotation_lever_score": float(far),
            "inertia_rotation_authority": float(inertia_rotation_authority),
            "required_rotation_arm_m": float(required_arm),
            "object_Izz_kgm2": float(Izz),
            "combined_score": float(score),
        })
    ranking.sort(key=lambda x: x["combined_score"], reverse=True)
    return ranking[0]["record"], ranking


def score_candidates(
    runtime,
    record,
    goal_state,
    hidden=None,
    top_k=24,
    object_com_xy=None,
    object_state=None,
    object_goal=None,
    gemini_advice=None,
    pusher_radius=0.008,
    object_inertia_zz_kgm2=None,
    object_mass_kg=None,
    planning_force_n: float = 6.0,
):
    """Step-9 score = learned prediction + monotonic progress + force/inertia prior."""
    candidates = generate_push_candidates(record, pusher_radius=pusher_radius)
    if not candidates:
        raise RuntimeError("No primitive-specific push candidates were generated.")
    actions, pred_delta, next_state = predict_candidates(runtime, record, candidates, hidden)
    goal = np.asarray(goal_state, np.float32)
    current = np.asarray(record["state_vector"], np.float32)
    rule = orientation_rule(record)

    current_pos_err = float(np.linalg.norm(current[:2] - goal[:2]))
    current_ori_err = _angle_abs_error(float(current[2]), float(goal[2]), rule)
    pos_err = np.linalg.norm(next_state[:, :2] - goal[None, :2], axis=1)
    ori_err = np.abs(_orientation_error(next_state[:, 2], goal[2], rule))
    improvement = current_pos_err - pos_err
    ori_improvement = current_ori_err - ori_err
    predicted_motion = np.linalg.norm(pred_delta[:, :2], axis=1)
    no_motion_penalty = 2.5 * np.maximum(0.004 - predicted_motion, 0.0) / 0.004

    progress_penalty = np.zeros(len(candidates), dtype=np.float64)
    for i in range(len(candidates)):
        if current_pos_err > 0.030:
            if improvement[i] < 0.0:
                progress_penalty[i] += 12.0 + 80.0 * abs(float(improvement[i]))
            elif improvement[i] < min(0.004, 0.10 * current_pos_err):
                progress_penalty[i] += 1.8
        elif improvement[i] < 0.001 and ori_improvement[i] < math.radians(1.5):
            progress_penalty[i] += 2.0

    goal_vec = goal[:2] - current[:2]
    goal_norm = float(np.linalg.norm(goal_vec))
    alignment_penalty = np.zeros(len(candidates), dtype=np.float64)
    if goal_norm > 1e-8:
        ug = goal_vec / goal_norm
        for i, cand in enumerate(candidates):
            d = np.asarray(cand.get("direction_world", [0, 0, 0]), dtype=np.float64)[:2]
            dn = float(np.linalg.norm(d))
            if dn > 1e-8:
                alignment = float(np.dot(d / dn, ug))
                alignment_penalty[i] = (
                    1.4 * max(0.0, 0.55 - alignment) ** 2
                    if current_pos_err > 0.035 else
                    0.35 * max(0.0, 0.20 - alignment) ** 2
                )

    pos_cost = (pos_err / 0.10) ** 2
    ori_cost = 0.55 * (ori_err / math.radians(30.0)) ** 2
    effort_cost = 0.025 * ((actions[:, 5] / max(PUSH_LENGTHS)) ** 2 + 0.20 * (actions[:, 6] / max(PUSH_SPEEDS)) ** 2)
    risk_cost = 0.20 * np.array([RISK_VALUE.get(c.get("risk", "normal"), 0.25) for c in candidates])
    x, y = next_state[:, 0], next_state[:, 1]
    dx_out = np.maximum(WORKSPACE["xmin"] - x, 0) + np.maximum(x - WORKSPACE["xmax"], 0)
    dy_out = np.maximum(WORKSPACE["ymin"] - y, 0) + np.maximum(y - WORKSPACE["ymax"], 0)
    workspace_cost = 50.0 * ((dx_out / 0.02) ** 2 + (dy_out / 0.02) ** 2)
    obstacle_cost = _obstacle_cost(next_state[:, :2], record)

    inertia_cost = np.zeros(len(candidates), dtype=np.float64)
    inertia_meta = []
    if object_com_xy is not None and object_state is not None and object_goal is not None:
        for i, cand in enumerate(candidates):
            inertia_cost[i], meta = _inertia_candidate_prior(
                cand, object_com_xy, object_state, object_goal,
                object_inertia_zz_kgm2=object_inertia_zz_kgm2,
                object_mass_kg=object_mass_kg,
                planning_force_n=planning_force_n,
            )
            inertia_meta.append(meta)
    else:
        inertia_meta = [{} for _ in candidates]

    total = (
        pos_cost + ori_cost + effort_cost + risk_cost + workspace_cost + obstacle_cost
        + no_motion_penalty + progress_penalty + alignment_penalty
        + 0.65 * inertia_cost
    )
    one_shot = (pos_err <= 0.020) & (ori_err <= math.radians(7.0))
    total = total - one_shot.astype(np.float64) * 2.0
    order = np.argsort(total)
    top = order[: min(int(top_k), len(order))]

    torque_arr = np.asarray([float(m.get("torque_z_nm", 0.0)) for m in inertia_meta], dtype=np.float32)
    alpha_arr = np.asarray([float(m.get("angular_accel_est_rad_s2", 0.0)) for m in inertia_meta], dtype=np.float32)
    result = {
        "record": record, "goal_state": goal, "rule": rule,
        "candidates": candidates, "actions": actions, "pred_delta": pred_delta,
        "pred_next": next_state, "cost": total.astype(np.float32),
        "obstacle_cost": obstacle_cost.astype(np.float32),
        "predicted_motion_m": predicted_motion.astype(np.float32),
        "predicted_position_improvement_m": improvement.astype(np.float32),
        "predicted_orientation_improvement_rad": ori_improvement.astype(np.float32),
        "inertia_prior_cost": inertia_cost.astype(np.float32),
        "candidate_torque_z_nm": torque_arr,
        "candidate_angular_accel_est_rad_s2": alpha_arr,
        "object_inertia_zz_kgm2": float(_safe_inertia_zz(object_inertia_zz_kgm2)),
        "object_mass_kg": float(_safe_mass(object_mass_kg)),
        "planning_force_n": float(planning_force_n),
        "object_com_xy": None if object_com_xy is None else np.asarray(object_com_xy, dtype=np.float32),
        "top_indices": top.astype(np.int64),
    }
    np.savez_compressed(
        PLANNING_DIR / "step9_latest.npz",
        actions=actions, pred_delta=pred_delta, pred_next=next_state, cost=total,
        inertia_prior_cost=inertia_cost, candidate_torque_z_nm=torque_arr,
        candidate_angular_accel_est_rad_s2=alpha_arr, top_indices=top, goal_state=goal,
    )
    return result


def rmppi_plan(
    runtime: PushModelRuntime,
    step9: dict,
    hidden=None,
    horizon: int = 2,
    num_rollouts: int = 192,
    iterations: int = 3,
    pool_size: int = 28,
    temperature: float = 0.8,
    seed: int = 42,
):
    """Short-horizon RMPPI fallback with inertia-aware rotational authority."""
    rng = np.random.default_rng(seed)
    order = np.argsort(step9["cost"])
    pool = np.asarray(order[: min(int(pool_size), len(order))], dtype=np.int64)
    templates = [_candidate_template(step9["candidates"][int(i)], step9["record"]["state_vector"]) for i in pool]
    one_step = np.asarray(step9["cost"][pool], dtype=np.float64)
    logits = -(one_step - one_step.min()) / max(float(temperature), 1e-6)
    probs0 = np.exp(logits - logits.max()); probs0 /= probs0.sum() + 1e-12
    probs = np.repeat(probs0[None, :], int(horizon), axis=0)

    start_state = np.asarray(step9["record"]["state_vector"], np.float32)
    start_goal_cost = _goal_cost(start_state, step9["goal_state"], step9["rule"])
    start_com = step9.get("object_com_xy", None)
    if start_com is None:
        start_com = start_state[:2].copy()
    else:
        start_com = np.asarray(start_com, dtype=np.float64)[:2]
    Izz = float(step9.get("object_inertia_zz_kgm2", 1e-4))
    mass = float(step9.get("object_mass_kg", 0.25))
    force_n = float(step9.get("planning_force_n", 6.0))

    best_cost = float("inf"); best_seq = None; best_states = None
    for it in range(int(iterations)):
        sequences = np.stack([rng.choice(len(pool), size=int(num_rollouts), replace=True, p=probs[t]) for t in range(int(horizon))], axis=1)
        costs = np.zeros(int(num_rollouts), dtype=np.float64)
        all_states = []
        for r in range(int(num_rollouts)):
            state = start_state.copy()
            hstate = None if hidden is None else (hidden[0].clone(), hidden[1].clone())
            traj = [state.copy()]; prev_theta = None; last_goal_cost = start_goal_cost
            for t in range(int(horizon)):
                cand = _instantiate_template(templates[int(sequences[r, t])], state)
                # Approximate whole COM translation with the predicted primitive translation.
                com_now = start_com + (np.asarray(state[:2], dtype=np.float64) - np.asarray(start_state[:2], dtype=np.float64))
                inertia_c, _meta = _inertia_candidate_prior(
                    cand, com_now, state, step9["goal_state"], Izz, mass, force_n
                )
                state, hstate = _single_model_step(runtime, step9["record"], state, cand, hstate)
                gc = _goal_cost(state, step9["goal_state"], step9["rule"])
                if gc > last_goal_cost + 0.02:
                    costs[r] += 8.0 * (gc - last_goal_cost)
                costs[r] += 0.22 * gc + 0.16
                costs[r] += 0.03 * (cand["push_length"] / max(PUSH_LENGTHS)) ** 2
                costs[r] += 0.20 * RISK_VALUE.get(cand.get("risk", "normal"), 0.25)
                costs[r] += 0.45 * inertia_c
                costs[r] += float(_obstacle_cost(np.asarray(state[:2])[None, :], step9["record"])[0])
                if prev_theta is not None:
                    da = math.atan2(math.sin(cand["theta_push"] - prev_theta), math.cos(cand["theta_push"] - prev_theta))
                    costs[r] += 0.10 * da * da
                prev_theta = cand["theta_push"]; last_goal_cost = gc; traj.append(state.copy())
            costs[r] += 2.5 * _goal_cost(state, step9["goal_state"], step9["rule"])
            all_states.append(np.stack(traj))

        imin = int(np.argmin(costs))
        if float(costs[imin]) < best_cost:
            best_cost = float(costs[imin]); best_seq = sequences[imin].copy(); best_states = all_states[imin].copy()
        w = np.exp(-(costs - costs.min()) / max(float(temperature), 1e-6)); w /= w.sum() + 1e-12
        for t in range(int(horizon)):
            freq = np.zeros(len(pool), dtype=np.float64); np.add.at(freq, sequences[:, t], w); freq /= freq.sum() + 1e-12
            probs[t] = 0.35 * probs[t] + 0.65 * freq; probs[t] = np.maximum(probs[t], 1e-4); probs[t] /= probs[t].sum()
        print(f"RMPPI iteration {it+1}/{iterations}: best={costs[imin]:.4f}, global={best_cost:.4f}")

    global_indices = pool[best_seq]
    first_global = int(global_indices[0])
    return {
        "best_cost": best_cost,
        "best_step9_indices": global_indices,
        "best_families": [step9["candidates"][int(i)]["family"] for i in global_indices],
        "predicted_trajectory": best_states,
        "first_candidate_index": first_global,
        "first_candidate": step9["candidates"][first_global],
    }

# =============================================================================
# FINAL YAW-CORRECTION OVERRIDES
# =============================================================================
# These definitions are intentionally appended last so they override the earlier
# generic execution behavior without changing the trained network interface.

_TRANSLATION_EXECUTE_PUSH_PYBULLET = execute_push_pybullet


def _wrap_pi(a: float) -> float:
    return float(math.atan2(math.sin(float(a)), math.cos(float(a))))


def choose_yaw_correction_candidate(
    step9: dict,
    whole_object_com_xy,
    current_object_yaw: float,
    goal_object_yaw: float,
    object_inertia_zz_kgm2: Optional[float] = None,
    planning_force_n: float = 6.0,
    min_effective_arm_m: float = 0.010,
):
    """Choose a signed-torque primitive push for final tabletop-yaw correction.

    The generic learned model remains involved through ``step9``.  Near the final
    XY target, however, the physical requirement is different from translation:
    we need a contact whose moment about the *whole-object COM* has the same sign
    as the remaining yaw error.  This helper therefore combines:

      * Step-9 learned cost and predicted yaw improvement,
      * contact lever arm about whole-object COM,
      * desired torque sign,
      * whole-object inertia,
      * predicted translational drift.

    If no existing primitive-family direction has the correct moment sign, the
    best available contact is converted to a tangential push around the COM.
    """
    candidates = step9.get("candidates", [])
    if not candidates:
        return None

    com = np.asarray(whole_object_com_xy, dtype=np.float64).reshape(-1)[:2]
    current = np.asarray(step9["record"]["state_vector"], dtype=np.float64)
    goal = np.asarray(step9["goal_state"], dtype=np.float64)
    yaw_error = _wrap_pi(float(goal_object_yaw) - float(current_object_yaw))
    if abs(yaw_error) < math.radians(1.0):
        return None
    desired_sign = 1.0 if yaw_error > 0.0 else -1.0

    Izz = _safe_inertia_zz(object_inertia_zz_kgm2)
    F = max(float(planning_force_n), 0.1)
    current_ori_err = abs(yaw_error)
    current_xy = current[:2]

    raw_cost = np.asarray(step9.get("cost", np.zeros(len(candidates))), dtype=np.float64)
    cmin = float(np.min(raw_cost)) if len(raw_cost) else 0.0
    cspan = max(float(np.max(raw_cost) - cmin), 1e-6) if len(raw_cost) else 1.0
    pred_next = np.asarray(step9.get("pred_next", np.zeros((len(candidates), 6))), dtype=np.float64)

    best = None
    best_score = float("inf")
    all_meta = []
    for i, cand in enumerate(candidates):
        contact = np.asarray(cand.get("contact_world", [current_xy[0], current_xy[1], 0.0]), dtype=np.float64)[:2]
        d = np.asarray(cand.get("direction_world", [1.0, 0.0, 0.0]), dtype=np.float64)[:2]
        dn = float(np.linalg.norm(d))
        if dn < 1e-9:
            continue
        d = d / dn
        r = contact - com
        moment_arm_signed = float(r[0] * d[1] - r[1] * d[0])
        effective_arm = abs(moment_arm_signed)
        sign_good = moment_arm_signed * desired_sign > 0.0
        torque = F * moment_arm_signed
        alpha = torque / max(Izz, 1e-9)

        nxt = pred_next[i] if i < len(pred_next) else current
        pred_yaw_err = abs(_wrap_pi(float(goal[2]) - float(nxt[2])))
        yaw_gain = current_ori_err - pred_yaw_err
        pred_xy_drift = float(np.linalg.norm(nxt[:2] - current_xy))
        learned_norm = (float(raw_cost[i]) - cmin) / cspan if i < len(raw_cost) else 0.0

        family = str(cand.get("family", ""))
        family_bonus = 0.0
        if "rotation" in family or "offset" in family:
            family_bonus += 0.45
        if "rim" in family or "rolling" in family:
            family_bonus += 0.15

        # Required angular acceleration proxy for removing the yaw error in about
        # 0.65 s.  This is only an analytical prior; the actual stop is closed-loop.
        alpha_req = 2.0 * current_ori_err / (0.65 ** 2)
        authority = min(abs(alpha) / max(alpha_req, 1e-6), 2.0)

        score = 0.25 * learned_norm
        score += 6.0 * pred_xy_drift / 0.03
        score -= 2.0 * max(yaw_gain, 0.0) / max(current_ori_err, math.radians(2.0))
        score -= 1.2 * min(effective_arm / 0.06, 1.5)
        score -= 0.65 * min(authority, 1.5)
        score -= family_bonus
        if not sign_good:
            score += 7.0
        if effective_arm < float(min_effective_arm_m):
            score += 2.0 * (float(min_effective_arm_m) - effective_arm) / max(float(min_effective_arm_m), 1e-6)

        meta = {
            "candidate_index": int(i),
            "moment_arm_signed_m": float(moment_arm_signed),
            "effective_arm_m": float(effective_arm),
            "torque_z_nm": float(torque),
            "angular_accel_est_rad_s2": float(alpha),
            "predicted_yaw_gain_rad": float(yaw_gain),
            "predicted_xy_drift_m": float(pred_xy_drift),
            "sign_good": bool(sign_good),
            "score": float(score),
        }
        all_meta.append(meta)
        if sign_good and score < best_score:
            best_score = score
            best = (i, cand, meta)

    if best is not None:
        idx, cand, meta = best
        out = dict(cand)
        err_deg = abs(math.degrees(yaw_error))
        # Strong enough to make visible rotational progress, but still short enough
        # that the actual-yaw stop can prevent overshoot.
        if err_deg <= 8.0:
            # Fine-yaw pulse: short and slow so the final angle is not accepted
            # several degrees away or driven past the goal by residual momentum.
            out["push_length"] = float(np.clip(0.006 + 0.00070 * err_deg, 0.008, 0.014))
            out["push_speed"] = float(np.clip(float(out.get("push_speed", 0.020)), 0.010, 0.022))
            out["fine_yaw_mode"] = True
        else:
            out["push_length"] = float(np.clip(0.022 + 0.00070 * err_deg, 0.026, 0.066))
            out["push_speed"] = float(np.clip(float(out.get("push_speed", 0.045)), 0.035, 0.060))
            out["fine_yaw_mode"] = False
        out["yaw_tracking_active"] = True
        return {
            "candidate_index": int(idx),
            "candidate": out,
            "predicted_next": np.asarray(step9["pred_next"][idx], dtype=np.float32),
            "yaw_error_rad": float(yaw_error),
            "mode": "SIGNED-TORQUE STEP-9 YAW CORRECTION",
            **meta,
        }

    # ------------------------------------------------------------------
    # Analytical tangential fallback.
    # ------------------------------------------------------------------
    # Use the farthest valid network-generated contact and rotate its direction to
    # the tangent about whole-object COM.  This preserves primitive/contact
    # semantics while guaranteeing the requested torque sign.
    best_idx = None
    best_radius = -1.0
    for i, cand in enumerate(candidates):
        contact = np.asarray(cand.get("contact_world", [current_xy[0], current_xy[1], 0.0]), dtype=np.float64)[:2]
        rr = contact - com
        rrn = float(np.linalg.norm(rr))
        if rrn > best_radius:
            best_radius = rrn
            best_idx = i
    if best_idx is None or best_radius < 1e-6:
        return None

    base = dict(candidates[int(best_idx)])
    contact = np.asarray(base.get("contact_world"), dtype=np.float64)
    rr = contact[:2] - com
    rrn = max(float(np.linalg.norm(rr)), 1e-9)
    # perp(r) gives positive z moment; flip for negative desired yaw.
    tangent = desired_sign * np.array([-rr[1], rr[0]], dtype=np.float64) / rrn
    base["direction_world"] = np.array([tangent[0], tangent[1], 0.0], dtype=np.float32)
    base["theta_push"] = float(math.atan2(tangent[1], tangent[0]))
    base["family"] = str(base.get("family", "primitive_push")) + "__analytic_yaw_tangent"
    err_deg = abs(math.degrees(yaw_error))
    if err_deg <= 8.0:
        base["push_length"] = float(np.clip(0.006 + 0.00075 * err_deg, 0.008, 0.015))
        base["push_speed"] = 0.016
        base["fine_yaw_mode"] = True
    else:
        base["push_length"] = float(np.clip(0.026 + 0.00075 * err_deg, 0.030, 0.070))
        base["push_speed"] = 0.045
        base["fine_yaw_mode"] = False
    base["yaw_tracking_active"] = True
    moment = float(rr[0] * tangent[1] - rr[1] * tangent[0])
    torque = F * moment
    return {
        "candidate_index": int(best_idx),
        "candidate": base,
        "predicted_next": np.asarray(step9["pred_next"][best_idx], dtype=np.float32),
        "yaw_error_rad": float(yaw_error),
        "mode": "ANALYTIC TANGENTIAL YAW FALLBACK",
        "moment_arm_signed_m": float(moment),
        "effective_arm_m": abs(float(moment)),
        "torque_z_nm": float(torque),
        "angular_accel_est_rad_s2": float(torque / max(Izz, 1e-9)),
        "predicted_yaw_gain_rad": 0.0,
        "predicted_xy_drift_m": 0.0,
        "sign_good": True,
        "score": 0.0,
    }


def _execute_yaw_correction_pybullet(
    target_body_id: int,
    candidate: dict,
    dt=1.0/240.0,
    radius=0.008,
    support_z=0.0,
    pusher_height=0.10,
    franka_robot: Optional[FrankaPandaRobot]=None,
    progress_callback=None,
    realtime_visualization=True,
    return_home=False,
    force_feedback_enabled=True,
    force_config: Optional[ForceFeedbackConfig]=None,
    object_com_world=None,
    object_inertia_zz_kgm2: Optional[float]=None,
    goal_pose_world=None,
    goal_yaw_world: Optional[float]=None,
    goal_yaw_tolerance_deg: float=1.0,
    max_xy_drift_m: float=0.022,
    fine_yaw_band_deg: float=8.0,
    fine_force_scale: float=0.50,
):
    """Dedicated closed-loop yaw push.

    Unlike the translation executor, **XY tolerance does not stop the action**.
    The pusher continues until actual PyBullet yaw reaches the requested yaw,
    crosses it, starts getting worse, reaches a force limit, or exceeds the
    permitted XY drift.  This fixes the prior failure mode where the object was
    already within 15 mm in XY and therefore every yaw push stopped immediately.
    """
    _require_physics_connection()
    preview = prepare_executable_push(
        target_body_id, candidate, franka_robot,
        support_z, radius, pusher_height
    )
    if not preview.get("feasible", False):
        return {
            "contact_established": False,
            "actual_travel_m": 0.0,
            "object_displacement_m": 0.0,
            "failure_reason": preview.get("reason", "yaw candidate infeasible"),
            "goal_stop_reason": None,
            "yaw_tracking_active": True,
        }

    c = preview["candidate"]
    d = _candidate_xy_direction(c)
    start = np.asarray(c["execution_start_world"], dtype=np.float64)
    contact = np.asarray(c["execution_tool_contact_world"], dtype=np.float64)
    pusher_z = float(c["execution_pusher_z"])
    target_length = max(float(c.get("push_length", 0.045)), 0.010)
    commanded_speed = float(np.clip(float(c.get("push_speed", 0.045)), 0.025, 0.065))

    before_pos, before_q = p.getBasePositionAndOrientation(int(target_body_id))
    before_pos = np.asarray(before_pos, dtype=np.float64)
    before_yaw = float(p.getEulerFromQuaternion(before_q)[2])

    if goal_yaw_world is None:
        if goal_pose_world is None:
            raise ValueError("Yaw-tracking execution requires goal_yaw_world or goal_pose_world[2].")
        goal_yaw_world = float(np.asarray(goal_pose_world, dtype=np.float64).reshape(-1)[2])
    goal_yaw_world = float(goal_yaw_world)
    initial_yaw_error = _wrap_pi(goal_yaw_world - before_yaw)
    initial_yaw_sign = 0.0 if abs(initial_yaw_error) < 1e-9 else math.copysign(1.0, initial_yaw_error)
    yaw_tol = math.radians(float(goal_yaw_tolerance_deg))
    fine_yaw_mode = bool(c.get("fine_yaw_mode", False)) or abs(math.degrees(initial_yaw_error)) <= float(fine_yaw_band_deg)
    best_yaw_error = abs(initial_yaw_error)
    worsening_count = 0
    # Near the final angle, use a gentler controller.  Coarse yaw pushes keep the
    # normal force reference; fine pulses use roughly half the force and speed.
    if fine_yaw_mode:
        target_length = float(np.clip(target_length, 0.006, 0.016))
        commanded_speed = float(np.clip(commanded_speed, 0.008, 0.022))

    goal_xy = None
    if goal_pose_world is not None:
        arr = np.asarray(goal_pose_world, dtype=np.float64).reshape(-1)
        if arr.size >= 2:
            goal_xy = arr[:2].copy()
    initial_goal_xy_error = None if goal_xy is None else float(np.linalg.norm(before_pos[:2] - goal_xy))

    com_for_inertia = None
    if object_com_world is not None:
        arr = np.asarray(object_com_world, dtype=np.float64).reshape(-1)
        if arr.size >= 2 and np.all(np.isfinite(arr[:2])):
            com_for_inertia = arr
    inertia_zz = None
    try:
        if object_inertia_zz_kgm2 is not None and float(object_inertia_zz_kgm2) > 1e-9:
            inertia_zz = float(object_inertia_zz_kgm2)
    except Exception:
        inertia_zz = None

    signed_moment_arm = 0.0
    lever_arm = 0.0
    if com_for_inertia is not None:
        rr = contact[:2] - com_for_inertia[:2]
        signed_moment_arm = float(rr[0] * d[1] - rr[1] * d[0])
        lever_arm = float(np.linalg.norm(rr))
    torque_sign = 1.0 if signed_moment_arm >= 0.0 else -1.0

    gantry = None
    import time as _time

    def notify(stage):
        if progress_callback is not None:
            try:
                progress_callback(stage, None if gantry is None else _pusher_xyz(gantry))
            except Exception:
                pass

    def sim_sleep(mult=0.08):
        if realtime_visualization:
            _time.sleep(float(dt) * float(mult))

    try:
        # Faster low hover/direct reach; fall back to Cartesian pusher if Panda IK
        # cannot follow exactly.
        franka_follow_enabled = franka_robot is not None
        franka_fallback_reason = None
        if franka_robot is not None:
            current = franka_robot.current_tool_center()
            safe_z = max(float(start[2]) + 0.040, min(float(current[2]), float(start[2]) + 0.065))
            above = np.array([start[0], start[1], safe_z], dtype=np.float64)
            ok = franka_robot.move_tool_center_blocking(
                above, dt=dt, realtime=realtime_visualization,
                tolerance=0.027, cartesian_step=0.090, max_steps_per_waypoint=55,
                progress_callback=progress_callback, stage="YAW LOW HOVER"
            )
            if ok:
                ok = franka_robot.move_tool_center_blocking(
                    start, dt=dt, realtime=realtime_visualization,
                    tolerance=0.024, cartesian_step=0.075, max_steps_per_waypoint=60,
                    progress_callback=progress_callback, stage="YAW PUSH START"
                )
            else:
                ok = franka_robot.move_tool_center_blocking(
                    start, dt=dt, realtime=realtime_visualization,
                    tolerance=0.030, cartesian_step=0.080, max_steps_per_waypoint=65,
                    progress_callback=progress_callback, stage="YAW DIRECT REACH"
                )
            if not ok:
                franka_follow_enabled = False
                franka_fallback_reason = "PANDA_IK_WORKSPACE_FALLBACK_TO_CARTESIAN_PUSHER"
                print("WARNING: Panda cannot exactly follow yaw-correction contact; Cartesian pusher will execute it.")

        gantry = create_cartesian_pusher(
            start[:2], pusher_z, radius, pusher_height,
            friction=1.25, motor_force=1600.0, visible=True
        )
        _disable_pusher_collisions_except_target(gantry, target_body_id)
        _set_pusher_position_target(gantry, start[:2], force=1600.0)
        for _ in range(4):
            if franka_follow_enabled:
                franka_robot.command_tool_center(_pusher_xyz(gantry))
            p.stepSimulation(); notify("YAW READY"); sim_sleep(0.04)

        def move_to_xy(target_xy, speed, stage, stop_on_contact=False):
            target_xy = np.asarray(target_xy, dtype=np.float64)
            begin = _pusher_xy(gantry)
            vec = target_xy - begin
            dist = float(np.linalg.norm(vec))
            if dist < 1e-8:
                return _target_contact_exists(gantry, target_body_id)
            u = vec / dist
            ds = max(float(speed) * float(dt) * 4.0, 0.0020)
            steps = max(1, int(math.ceil(dist / ds)))
            seen = _target_contact_exists(gantry, target_body_id)
            for k in range(1, steps + 1):
                desired = begin + u * min(dist, k * ds)
                _set_pusher_position_target(gantry, desired, force=1600.0)
                for _sim in range(2):
                    if franka_follow_enabled:
                        franka_robot.command_tool_center(np.array([desired[0], desired[1], pusher_z], dtype=np.float64))
                    p.stepSimulation(); notify(stage)
                sim_sleep(0.04)
                if _target_contact_exists(gantry, target_body_id):
                    seen = True
                    if stop_on_contact:
                        _set_pusher_position_target(gantry, _pusher_xy(gantry), force=1600.0)
                        return True
            return seen

        contact_ok = move_to_xy(contact[:2], 0.11, "YAW APPROACH TRUE SURFACE", True)
        if not contact_ok:
            contact_ok = move_to_xy(contact[:2] + 0.018 * d, 0.075, "YAW CONTACT EXTENSION", True)
        if not contact_ok:
            return {
                "contact_established": False,
                "actual_travel_m": 0.0,
                "object_displacement_m": 0.0,
                "failure_reason": "Yaw correction could not establish selected-object contact.",
                "execution_candidate": _candidate_for_log(c),
                "goal_stop_reason": None,
                "yaw_tracking_active": True,
            }

        push_start = _pusher_xy(gantry).copy()
        force_controller = None
        force_log_path = None
        max_force_seen = 0.0
        force_sum = 0.0
        force_samples = 0
        max_abs_torque_seen = 0.0
        max_abs_alpha_seen = 0.0
        if force_feedback_enabled:
            _ff_cfg = force_config or ForceFeedbackConfig()
            if fine_yaw_mode:
                _ff_cfg = replace(
                    _ff_cfg,
                    desired_force_n=max(2.0, float(_ff_cfg.desired_force_n) * float(fine_force_scale)),
                    hard_force_limit_n=max(6.0, min(float(_ff_cfg.hard_force_limit_n), 9.0)),
                    min_push_speed_mps=min(float(_ff_cfg.min_push_speed_mps), 0.0025),
                    max_push_speed_mps=min(float(_ff_cfg.max_push_speed_mps), 0.030),
                    kp_speed=float(_ff_cfg.kp_speed) * 0.65,
                    ki_speed=float(_ff_cfg.ki_speed) * 0.50,
                    kd_speed=float(_ff_cfg.kd_speed) * 0.65,
                )
            force_controller = CartesianPushForceController(
                _ff_cfg,
                log_directory=PLANNING_DIR / "force_logs",
            )

        control_dt = max(float(dt) * (1.0 if fine_yaw_mode else 2.0), 1e-4)
        commanded_path_s = 0.0
        goal_stop_reason = None
        min_travel_before_worsen = min(0.004 if fine_yaw_mode else 0.012, 0.20 * target_length)
        while commanded_path_s < target_length - 1e-6:
            if force_controller is not None:
                meas = measure_push_force_pybullet(gantry, target_body_id, d)
                ff_state = force_controller.update(
                    float(meas["push_axis_force_n"]),
                    dt=control_dt,
                    nominal_speed_mps=commanded_speed,
                    lever_arm_m=abs(signed_moment_arm),
                    torque_sign=torque_sign,
                    inertia_zz_kgm2=inertia_zz,
                )
                active_speed = float(ff_state.commanded_speed_mps)
                max_force_seen = max(max_force_seen, float(ff_state.filtered_force_n))
                force_sum += float(ff_state.filtered_force_n)
                force_samples += 1
                max_abs_torque_seen = max(max_abs_torque_seen, abs(float(ff_state.torque_z_nm)))
                max_abs_alpha_seen = max(max_abs_alpha_seen, abs(float(ff_state.angular_accel_est_rad_s2)))
                if ff_state.hard_stop:
                    goal_stop_reason = "FORCE_HARD_LIMIT"
                    notify(f"YAW FORCE HARD STOP {ff_state.filtered_force_n:.1f} N")
                    break
            else:
                active_speed = commanded_speed
                ff_state = None

            commanded_path_s = min(target_length, commanded_path_s + max(active_speed, 0.0) * control_dt)
            desired_xy = push_start + d * commanded_path_s
            _set_pusher_position_target(gantry, desired_xy, force=1600.0)
            _sim_steps = 1 if fine_yaw_mode else 2
            for _sim in range(_sim_steps):
                if franka_follow_enabled:
                    franka_robot.command_tool_center(np.array([desired_xy[0], desired_xy[1], pusher_z], dtype=np.float64))
                p.stepSimulation()
                if ff_state is not None:
                    prefix = "FINE YAW" if fine_yaw_mode else "YAW"
                    notify(f"{prefix} FORCE PUSH {ff_state.filtered_force_n:.1f}/{ff_state.reference_force_n:.1f} N")
                else:
                    notify("FINE YAW PULSE" if fine_yaw_mode else "YAW-CORRECTION PUSH")
            sim_sleep(0.025 if fine_yaw_mode else 0.10)

            pos_now, q_now = p.getBasePositionAndOrientation(int(target_body_id))
            pos_now = np.asarray(pos_now, dtype=np.float64)
            yaw_now = float(p.getEulerFromQuaternion(q_now)[2])
            yaw_err = _wrap_pi(goal_yaw_world - yaw_now)
            yaw_abs = abs(yaw_err)
            actual_travel = float(np.dot(_pusher_xy(gantry) - push_start, d))

            if yaw_abs < best_yaw_error:
                best_yaw_error = yaw_abs
                worsening_count = 0
            elif actual_travel >= min_travel_before_worsen and yaw_abs > best_yaw_error + math.radians(2.5):
                worsening_count += 1
            else:
                worsening_count = max(0, worsening_count - 1)

            # Predict a small amount of residual rotation after force release.
            # This prevents a fine pulse from stopping at +0.8 deg but settling at
            # -2 or -3 deg because of remaining angular velocity.
            try:
                _lin_vel, _ang_vel = p.getBaseVelocity(int(target_body_id))
                omega_z = float(_ang_vel[2])
            except Exception:
                omega_z = 0.0
            settle_horizon_s = 0.045 if fine_yaw_mode else 0.020
            predicted_residual = abs(omega_z) * settle_horizon_s
            brake_band = yaw_tol + min(math.radians(1.25 if fine_yaw_mode else 0.50), predicted_residual)

            if yaw_abs <= brake_band:
                goal_stop_reason = "FINE_YAW_BRAKE" if fine_yaw_mode else "GOAL_YAW_REACHED_DURING_PUSH"
                break

            # Crossing is not accepted as final success.  It only ends this pulse;
            # after settling the outer closed loop measures the actual angle and
            # applies an opposite micro-pulse if more than +/-1 deg remains.
            if initial_yaw_sign != 0.0 and yaw_err * initial_yaw_sign <= 0.0:
                goal_stop_reason = "FINE_YAW_TARGET_CROSSED" if fine_yaw_mode else "YAW_TARGET_CROSSED"
                break

            if worsening_count >= 3:
                goal_stop_reason = "ABORTED_WHEN_YAW_ERROR_STARTED_WORSENING"
                break

            # Rotation is allowed to translate slightly, but not to destroy the
            # already-good XY placement.
            xy_drift = float(np.linalg.norm(pos_now[:2] - before_pos[:2]))
            if xy_drift > float(max_xy_drift_m):
                goal_stop_reason = "ABORTED_MAX_XY_DRIFT_DURING_YAW"
                break

        actual_travel = float(np.dot(_pusher_xy(gantry) - push_start, d))
        _set_pusher_position_target(gantry, _pusher_xy(gantry), force=1600.0)

        # Smoothly release force.  Fine-yaw uses a shorter release and smaller
        # backoff so the release itself does not introduce another few degrees.
        if force_controller is not None and force_controller.contact_triggered:
            cfg_ff = force_controller.config
            _smooth_time = min(float(cfg_ff.smoothing_time_s), 0.10) if fine_yaw_mode else float(cfg_ff.smoothing_time_s)
            smooth_steps = max(1, int(math.ceil(_smooth_time / float(dt))))
            smooth_start = _pusher_xy(gantry).copy()
            for j in range(smooth_steps):
                elapsed = (j + 1) * float(dt)
                fref = force_controller.smoothing_reference(elapsed)
                meas = measure_push_force_pybullet(gantry, target_body_id, d)
                force_controller.update(
                    float(meas["push_axis_force_n"]), dt=float(dt),
                    nominal_speed_mps=0.0, reference_force_n=fref,
                    lever_arm_m=abs(signed_moment_arm), torque_sign=torque_sign,
                    inertia_zz_kgm2=inertia_zz,
                )
                frac = float(j + 1) / float(smooth_steps)
                _backoff = min(float(cfg_ff.smoothing_backoff_m), 0.002) if fine_yaw_mode else float(cfg_ff.smoothing_backoff_m)
                smooth_xy = smooth_start - d * _backoff * frac
                _set_pusher_position_target(gantry, smooth_xy, force=1600.0)
                if franka_follow_enabled:
                    franka_robot.command_tool_center(np.array([smooth_xy[0], smooth_xy[1], pusher_z], dtype=np.float64))
                p.stepSimulation()
                if j % 4 == 0:
                    notify(f"YAW FORCE SMOOTHING ref={fref:.1f} N")
                sim_sleep(0.025)
            force_log_path = force_controller.save_csv(prefix=f"body{int(target_body_id)}_yaw_force")

        retract = _pusher_xy(gantry) - 0.028 * d
        move_to_xy(retract, 0.11, "YAW SHORT RETRACT", False)
        if franka_follow_enabled:
            now = franka_robot.current_tool_center()
            hover = np.array([now[0], now[1], max(now[2] + 0.075, float(start[2]) + 0.085)], dtype=np.float64)
            franka_robot.move_tool_center_blocking(
                hover, dt=dt, realtime=realtime_visualization,
                tolerance=0.022, cartesian_step=0.070, max_steps_per_waypoint=45,
                progress_callback=progress_callback, stage="YAW CORRECTION COMPLETE"
            )
            if return_home:
                franka_robot.home_fast(dt=dt, realtime=realtime_visualization, progress_callback=progress_callback)

        _settle_steps = 120 if fine_yaw_mode else 55
        for k in range(_settle_steps):
            p.stepSimulation()
            if k % 10 == 0:
                notify("FINE YAW SETTLING" if fine_yaw_mode else "YAW OBJECT SETTLING")
            sim_sleep(0.018 if fine_yaw_mode else 0.028)

        after_pos, after_q = p.getBasePositionAndOrientation(int(target_body_id))
        after_pos = np.asarray(after_pos, dtype=np.float64)
        after_yaw = float(p.getEulerFromQuaternion(after_q)[2])
        displacement = float(np.linalg.norm(after_pos[:2] - before_pos[:2]))
        final_yaw_error = _wrap_pi(goal_yaw_world - after_yaw)
        return {
            "contact_established": True,
            "actual_travel_m": float(actual_travel),
            "commanded_length_m": float(target_length),
            "commanded_speed_mps": float(commanded_speed),
            "object_displacement_m": displacement,
            "before_pose_world": [float(before_pos[0]), float(before_pos[1]), float(before_yaw)],
            "after_pose_world": [float(after_pos[0]), float(after_pos[1]), float(after_yaw)],
            "actual_object_delta": [
                float(after_pos[0] - before_pos[0]),
                float(after_pos[1] - before_pos[1]),
                float(_wrap_pi(after_yaw - before_yaw)),
            ],
            "execution_candidate": _candidate_for_log(c),
            "yaw_tracking_active": True,
            "fine_yaw_mode": bool(fine_yaw_mode),
            "yaw_error_before_deg": float(math.degrees(initial_yaw_error)),
            "best_abs_yaw_error_during_push_deg": float(math.degrees(best_yaw_error)),
            "yaw_error_after_deg": float(math.degrees(final_yaw_error)),
            "goal_stop_reason": goal_stop_reason,
            "force_feedback_enabled": bool(force_feedback_enabled),
            "max_push_force_n": float(max_force_seen),
            "mean_push_force_n": float(force_sum / max(force_samples, 1)),
            "force_log_path": None if force_log_path is None else str(force_log_path),
            "whole_object_inertia_zz_kgm2": None if inertia_zz is None else float(inertia_zz),
            "contact_lever_arm_m": float(lever_arm),
            "signed_moment_arm_m": float(signed_moment_arm),
            "max_abs_torque_z_nm": float(max_abs_torque_seen),
            "max_abs_angular_accel_est_rad_s2": float(max_abs_alpha_seen),
            "franka_follow_enabled": bool(franka_follow_enabled),
            "franka_fallback_reason": franka_fallback_reason,
            "failure_reason": None,
        }
    finally:
        _remove_body_safe(gantry)


def execute_push_pybullet(
    target_body_id: int,
    candidate: dict,
    dt=1.0/240.0,
    radius=0.008,
    support_z=0.0,
    pusher_height=0.10,
    franka_robot: Optional[FrankaPandaRobot]=None,
    progress_callback=None,
    realtime_visualization=True,
    display_every_steps=6,
    return_home=False,
    fast_robot_motion=True,
    goal_pose_world=None,
    goal_position_tolerance_m: float=0.015,
    abort_if_goal_error_worsens: bool=True,
    force_feedback_enabled: bool=True,
    force_config: Optional[ForceFeedbackConfig]=None,
    object_com_world=None,
    object_inertia_zz_kgm2: Optional[float]=None,
    yaw_tracking_active: bool=False,
    goal_yaw_world: Optional[float]=None,
    goal_yaw_tolerance_deg: float=1.0,
    max_xy_drift_m: float=0.022,
    fine_yaw_band_deg: float=8.0,
    fine_force_scale: float=0.50,
    **compat_kwargs,
):
    """Dispatch to translation or dedicated final-yaw execution.

    Translation keeps the proven goal-tracked executor.  Final orientation uses a
    separate stop condition based on actual yaw, not XY distance.
    """
    # Backward/forward compatibility: older main.py revisions may pass
    # optional fine-yaw keywords.  Never terminate the physical pipeline only
    # because one optional execution keyword is newer than push.py.
    if compat_kwargs:
        if "fine_yaw_band_deg" in compat_kwargs:
            fine_yaw_band_deg = float(compat_kwargs.pop("fine_yaw_band_deg"))
        if "fine_force_scale" in compat_kwargs:
            fine_force_scale = float(compat_kwargs.pop("fine_force_scale"))
        if "yaw_tracking_active" in compat_kwargs:
            yaw_tracking_active = bool(compat_kwargs.pop("yaw_tracking_active"))
        if "goal_yaw_world" in compat_kwargs:
            goal_yaw_world = compat_kwargs.pop("goal_yaw_world")
        if "goal_yaw_tolerance_deg" in compat_kwargs:
            goal_yaw_tolerance_deg = float(compat_kwargs.pop("goal_yaw_tolerance_deg"))
        if "max_xy_drift_m" in compat_kwargs:
            max_xy_drift_m = float(compat_kwargs.pop("max_xy_drift_m"))
        if compat_kwargs:
            print("WARNING: ignoring unsupported optional execute_push_pybullet kwargs:",
                  sorted(compat_kwargs.keys()))

    if bool(yaw_tracking_active):
        return _execute_yaw_correction_pybullet(
            target_body_id=target_body_id,
            candidate=candidate,
            dt=dt,
            radius=radius,
            support_z=support_z,
            pusher_height=pusher_height,
            franka_robot=franka_robot,
            progress_callback=progress_callback,
            realtime_visualization=realtime_visualization,
            return_home=return_home,
            force_feedback_enabled=force_feedback_enabled,
            force_config=force_config,
            object_com_world=object_com_world,
            object_inertia_zz_kgm2=object_inertia_zz_kgm2,
            goal_pose_world=goal_pose_world,
            goal_yaw_world=goal_yaw_world,
            goal_yaw_tolerance_deg=goal_yaw_tolerance_deg,
            max_xy_drift_m=max_xy_drift_m,
            fine_yaw_band_deg=fine_yaw_band_deg,
            fine_force_scale=fine_force_scale,
        )
    return _TRANSLATION_EXECUTE_PUSH_PYBULLET(
        target_body_id=target_body_id,
        candidate=candidate,
        dt=dt,
        radius=radius,
        support_z=support_z,
        pusher_height=pusher_height,
        franka_robot=franka_robot,
        progress_callback=progress_callback,
        realtime_visualization=realtime_visualization,
        display_every_steps=display_every_steps,
        return_home=return_home,
        fast_robot_motion=fast_robot_motion,
        goal_pose_world=goal_pose_world,
        goal_position_tolerance_m=goal_position_tolerance_m,
        abort_if_goal_error_worsens=abort_if_goal_error_worsens,
        force_feedback_enabled=force_feedback_enabled,
        force_config=force_config,
        object_com_world=object_com_world,
        object_inertia_zz_kgm2=object_inertia_zz_kgm2,
    )
