from __future__ import annotations
FORCE_FEEDBACK_BUILD_ID = "2026-09-22_PUSH_GRASP_ENGINE_V1"

"""Paper-inspired Cartesian push force feedback for the Primitive Push Engine.

This module is deliberately independent from perception and the learned forward model.
It is used ONLY by the low-level physical execution stage.

For the current PyBullet architecture the physical contact body is the red Cartesian
pusher. PyBullet directly exposes contact forces, so in simulation we use those forces
instead of reconstructing Cartesian force from Franka joint torques.

On a real Franka, replace ``measure_push_force_pybullet`` with the robot's measured
external Cartesian wrench (or J^{-T} tau_ext), while keeping the controller interface.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence
import csv
import math
import time

import numpy as np
import pybullet as p


@dataclass
class ForceFeedbackConfig:
    # Contact/force thresholds -------------------------------------------------
    # The paper used 13 N for its 260 g bushing and its own gripper/environment.
    # That value is NOT universal.  For the small tabletop PyBullet objects in
    # this project, start lower and tune if necessary.
    trigger_force_n: float = 0.8
    desired_force_n: float = 6.0
    hard_force_limit_n: float = 14.0

    # Low-pass filtering of the measured contact force.
    force_filter_alpha: float = 0.25

    # Force-error -> commanded push-speed correction.
    # v_cmd = v_nominal + Kp*e + Ki*int(e) + Kd*de/dt
    kp_speed: float = 0.0060       # (m/s)/N
    ki_speed: float = 0.0008       # (m/s)/(N*s)
    kd_speed: float = 0.00010      # (m/s)/(N/s)
    integral_limit_n_s: float = 8.0

    # Speed limits while force feedback is active.
    min_push_speed_mps: float = 0.004
    max_push_speed_mps: float = 0.095

    # If force rises well above the reference, slow strongly before the hard stop.
    high_force_ratio: float = 1.35
    high_force_speed_scale: float = 0.25

    # Paper-inspired force smoothing at the end of the push.
    smoothing_time_s: float = 0.20
    smoothing_backoff_m: float = 0.004

    # Save every force-controlled push for debugging / paper plots.
    save_logs: bool = True


@dataclass
class ForceFeedbackState:
    raw_force_n: float
    filtered_force_n: float
    reference_force_n: float
    error_n: float
    commanded_speed_mps: float
    contact_triggered: bool
    hard_stop: bool
    torque_z_nm: float = 0.0
    angular_accel_est_rad_s2: float = 0.0


def _unit_xy(direction_world: Sequence[float]) -> np.ndarray:
    d = np.asarray(direction_world, dtype=np.float64).reshape(-1)
    if d.size < 2:
        return np.array([1.0, 0.0], dtype=np.float64)
    d = d[:2]
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        return np.array([1.0, 0.0], dtype=np.float64)
    return d / n


def measure_push_force_pybullet(
    pusher_body_id: int,
    target_body_id: int,
    push_direction_world: Sequence[float],
) -> Dict[str, float]:
    """Measure pusher/target contact force from PyBullet contact points.

    Returns both total normal force and the part associated with a surface whose
    normal is aligned with the commanded planar push direction.  The projected
    value is normally the most useful scalar for this planar pusher.
    """
    if not p.isConnected():
        return {
            "normal_force_n": 0.0,
            "push_axis_force_n": 0.0,
            "num_contacts": 0,
        }

    d = _unit_xy(push_direction_world)
    normal_total = 0.0
    projected_total = 0.0
    count = 0

    for cp in p.getContactPoints(bodyA=int(pusher_body_id), bodyB=int(target_body_id)):
        # PyBullet contact tuple fields used here:
        # cp[7] = contactNormalOnB, cp[9] = normalForce.
        try:
            normal_on_b = np.asarray(cp[7], dtype=np.float64)
            normal_force = max(0.0, float(cp[9]))
        except Exception:
            continue

        normal_total += normal_force
        count += 1

        nxy = normal_on_b[:2]
        nxy_norm = float(np.linalg.norm(nxy))
        if nxy_norm > 1e-9:
            nxy = nxy / nxy_norm
            # For a front-face push the contact normal is approximately parallel
            # or anti-parallel to d.  Absolute projection avoids depending on
            # Bullet's normal sign convention while preserving axis relevance.
            alignment = abs(float(np.dot(nxy, d)))
            projected_total += normal_force * alignment

    # If the contact normal is nearly vertical or numerically awkward, falling
    # back to the total normal force is safer than returning an artificial zero.
    if count > 0 and projected_total < 0.05 * normal_total:
        projected_total = normal_total

    return {
        "normal_force_n": float(normal_total),
        "push_axis_force_n": float(projected_total),
        "num_contacts": int(count),
    }


class CartesianPushForceController:
    """Scalar force regulator for the active push direction.

    This is a practical PyBullet adaptation of the paper's hybrid position/force
    idea: position/IK control is retained for all non-pushing motion and for pose
    holding, while the active push-direction speed is continuously adjusted from
    force error after contact is triggered.
    """

    def __init__(
        self,
        config: Optional[ForceFeedbackConfig] = None,
        log_directory: Optional[Path | str] = None,
    ):
        self.config = config or ForceFeedbackConfig()
        self.log_directory = None if log_directory is None else Path(log_directory)
        if self.log_directory is not None:
            self.log_directory.mkdir(parents=True, exist_ok=True)
        self.reset()

    def reset(self) -> None:
        self.filtered_force_n = 0.0
        self.integral_error = 0.0
        self.previous_error = 0.0
        self.contact_triggered = False
        self.elapsed_s = 0.0
        self.samples = []

    def update(
        self,
        measured_force_n: float,
        dt: float,
        nominal_speed_mps: float,
        reference_force_n: Optional[float] = None,
        lever_arm_m: float = 0.0,
        torque_sign: float = 1.0,
        inertia_zz_kgm2: Optional[float] = None,
    ) -> ForceFeedbackState:
        cfg = self.config
        dt = max(float(dt), 1e-6)
        raw = max(0.0, float(measured_force_n))

        alpha = float(np.clip(cfg.force_filter_alpha, 0.0, 1.0))
        self.filtered_force_n = (
            alpha * raw + (1.0 - alpha) * self.filtered_force_n
        )

        ref = float(cfg.desired_force_n if reference_force_n is None else reference_force_n)
        ref = max(0.0, ref)

        if self.filtered_force_n >= float(cfg.trigger_force_n):
            self.contact_triggered = True

        error = ref - self.filtered_force_n
        self.integral_error += error * dt
        lim = abs(float(cfg.integral_limit_n_s))
        self.integral_error = float(np.clip(self.integral_error, -lim, lim))
        derivative = (error - self.previous_error) / dt
        self.previous_error = error

        # Before the trigger the pusher should continue its nominal contact-seeking
        # motion. Once triggered, force feedback modifies the active-axis speed.
        if self.contact_triggered:
            speed = (
                float(nominal_speed_mps)
                + float(cfg.kp_speed) * error
                + float(cfg.ki_speed) * self.integral_error
                + float(cfg.kd_speed) * derivative
            )
        else:
            speed = float(nominal_speed_mps)

        if ref > 1e-6 and self.filtered_force_n > float(cfg.high_force_ratio) * ref:
            speed = min(speed, float(nominal_speed_mps) * float(cfg.high_force_speed_scale))

        hard_stop = self.filtered_force_n >= float(cfg.hard_force_limit_n)
        if hard_stop:
            speed = 0.0

        speed = float(np.clip(
            speed,
            0.0 if hard_stop else float(cfg.min_push_speed_mps),
            float(cfg.max_push_speed_mps),
        ))

        self.elapsed_s += dt
        torque_z_nm = float(torque_sign) * float(self.filtered_force_n) * max(0.0, float(lever_arm_m))
        inertia_val = None if inertia_zz_kgm2 is None else float(inertia_zz_kgm2)
        angular_accel = 0.0
        if inertia_val is not None and math.isfinite(inertia_val) and inertia_val > 1e-9:
            angular_accel = torque_z_nm / inertia_val

        state = ForceFeedbackState(
            raw_force_n=raw,
            filtered_force_n=float(self.filtered_force_n),
            reference_force_n=ref,
            error_n=float(error),
            commanded_speed_mps=speed,
            contact_triggered=bool(self.contact_triggered),
            hard_stop=bool(hard_stop),
            torque_z_nm=float(torque_z_nm),
            angular_accel_est_rad_s2=float(angular_accel),
        )

        self.samples.append({
            "time_s": float(self.elapsed_s),
            "raw_force_n": state.raw_force_n,
            "filtered_force_n": state.filtered_force_n,
            "reference_force_n": state.reference_force_n,
            "error_n": state.error_n,
            "commanded_speed_mps": state.commanded_speed_mps,
            "contact_triggered": int(state.contact_triggered),
            "hard_stop": int(state.hard_stop),
            "torque_z_nm": state.torque_z_nm,
            "angular_accel_est_rad_s2": state.angular_accel_est_rad_s2,
            "lever_arm_m": float(max(0.0, lever_arm_m)),
            "inertia_zz_kgm2": "" if inertia_zz_kgm2 is None else float(inertia_zz_kgm2),
        })
        return state

    def smoothing_reference(self, elapsed_s: float) -> float:
        """Linear reference-force ramp from F_desired to zero.

        This mirrors the paper's end-of-push force smoothing concept with a
        first-order ramp over T_D (0.2 s by default).
        """
        td = max(float(self.config.smoothing_time_s), 1e-6)
        alpha = 1.0 - float(np.clip(float(elapsed_s) / td, 0.0, 1.0))
        return float(self.config.desired_force_n) * alpha

    def save_csv(self, prefix: str = "push_force") -> Optional[Path]:
        if not self.config.save_logs or not self.samples or self.log_directory is None:
            return None
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = self.log_directory / f"{prefix}_{stamp}.csv"
        keys = list(self.samples[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.samples)
        return path


# A conservative default for the current multi-object PyBullet scene.
# Change only this function if you want a single central place to tune force.
def make_default_force_controller(log_directory: Optional[Path | str] = None):
    return CartesianPushForceController(
        ForceFeedbackConfig(
            trigger_force_n=0.8,
            desired_force_n=6.0,
            hard_force_limit_n=14.0,
            force_filter_alpha=0.25,
            kp_speed=0.0060,
            ki_speed=0.0008,
            kd_speed=0.00010,
            min_push_speed_mps=0.004,
            max_push_speed_mps=0.095,
            high_force_ratio=1.35,
            high_force_speed_scale=0.25,
            smoothing_time_s=0.20,
            smoothing_backoff_m=0.004,
            save_logs=True,
        ),
        log_directory=log_directory,
    )


# =============================================================================
# WHOLE-OBJECT MASS / COM / INERTIA HELPERS
# =============================================================================

def _quat_to_rot(q):
    return np.asarray(p.getMatrixFromQuaternion(q), dtype=np.float64).reshape(3, 3)


def compute_pybullet_body_mass_com_inertia(body_id: int) -> Dict[str, object]:
    """Return exact simulator mass, whole-object COM and inertia tensor.

    The inertia tensor is assembled about the *whole-body COM* using the
    parallel-axis theorem.  This is useful as simulation ground truth and for
    validating the perception-derived inertia estimate.
    """
    if not p.isConnected():
        raise RuntimeError("PyBullet is not connected.")

    parts = []

    # Base. For the procedural objects used by this project the base inertial
    # frame is coincident with the base pose, but we still honor local inertial
    # offsets/orientation when present.
    dyn = p.getDynamicsInfo(int(body_id), -1)
    mass = float(dyn[0])
    if mass > 0.0:
        base_pos, base_orn = p.getBasePositionAndOrientation(int(body_id))
        local_i_pos = dyn[3]
        local_i_orn = dyn[4]
        try:
            com_pos, com_orn = p.multiplyTransforms(base_pos, base_orn, local_i_pos, local_i_orn)
        except Exception:
            com_pos, com_orn = base_pos, base_orn
        parts.append((mass, np.asarray(com_pos, float), np.asarray(dyn[2], float), com_orn))

    # Links, if any. getLinkState()[0:2] are the world COM pose.
    for link in range(p.getNumJoints(int(body_id))):
        dyn = p.getDynamicsInfo(int(body_id), int(link))
        mass = float(dyn[0])
        if mass <= 0.0:
            continue
        ls = p.getLinkState(int(body_id), int(link), computeForwardKinematics=True)
        parts.append((mass, np.asarray(ls[0], float), np.asarray(dyn[2], float), ls[1]))

    if not parts:
        pos, _ = p.getBasePositionAndOrientation(int(body_id))
        return {
            "mass_kg": 0.0,
            "com_world": np.asarray(pos, dtype=np.float64),
            "inertia_world_kgm2": np.zeros((3, 3), dtype=np.float64),
            "Izz_kgm2": 0.0,
        }

    total_mass = sum(x[0] for x in parts)
    com = sum(m * c for m, c, _diag, _orn in parts) / max(total_mass, 1e-12)
    I_total = np.zeros((3, 3), dtype=np.float64)
    eye = np.eye(3, dtype=np.float64)
    for m, c, diag, orn in parts:
        R = _quat_to_rot(orn)
        I_local = np.diag(np.maximum(diag, 0.0))
        I_world_at_part_com = R @ I_local @ R.T
        r = np.asarray(c, float) - com
        I_parallel = m * ((float(np.dot(r, r)) * eye) - np.outer(r, r))
        I_total += I_world_at_part_com + I_parallel

    return {
        "mass_kg": float(total_mass),
        "com_world": np.asarray(com, dtype=np.float64),
        "inertia_world_kgm2": I_total,
        "Izz_kgm2": float(max(I_total[2, 2], 0.0)),
    }


def planar_torque_from_force(
    contact_world: Sequence[float],
    com_world: Sequence[float],
    push_direction_world: Sequence[float],
    force_n: float,
) -> Dict[str, float]:
    """Compute tau_z = r x F and alpha_z = tau_z/Izz externally.

    This helper only computes the moment.  Divide by the chosen whole-object
    Izz to obtain the angular-acceleration proxy.
    """
    c = np.asarray(contact_world, dtype=np.float64).reshape(-1)
    com = np.asarray(com_world, dtype=np.float64).reshape(-1)
    d = _unit_xy(push_direction_world)
    r = c[:2] - com[:2]
    tau = float(force_n) * float(r[0] * d[1] - r[1] * d[0])
    return {
        "lever_arm_m": float(np.linalg.norm(r)),
        "signed_moment_arm_m": float(r[0] * d[1] - r[1] * d[0]),
        "torque_z_nm": tau,
    }
