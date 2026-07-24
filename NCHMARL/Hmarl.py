from __future__ import annotations
import os
import json
import numpy as np
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


def path_loss_db(distance_m: np.ndarray, freq_ghz: float = 5.9) -> np.ndarray:
    """Simple vehicular urban path-loss model (log-distance, dB)."""
    d_km = np.maximum(distance_m, 1.0) / 1000.0
    return 128.1 + 37.6 * np.log10(d_km) + 20 * np.log10(freq_ghz / 5.9)


def rayleigh_fading(rng: np.random.Generator, size) -> np.ndarray:
    """Rayleigh small-scale fading power gain (unit mean)."""
    re = rng.normal(0, 1 / np.sqrt(2), size)
    im = rng.normal(0, 1 / np.sqrt(2), size)
    return re ** 2 + im ** 2

# --------------------------------------------------------------------------- #
# Digital twin: synchronized state estimate + short-horizon forecast
# --------------------------------------------------------------------------- #

class DigitalTwin:
    """
    Minimal alpha-beta (Holt's linear trend) filter used as the network's
    digital-twin state estimator. `sync()` ingests a (possibly noisy /
    delayed) observation of the real system and updates a smoothed level
    + trend estimate; `predict(horizon)` extrapolates forward. Control
    logic (managers, workers) reads from the twin instead of raw
    telemetry, which is the point of a twin: a always-available,
    denoised, predictive proxy for the physical system.
    """

    def __init__(self, dim: int, alpha: float = 0.35, beta: float = 0.25,
                 obs_noise_std: float = 0.02, seed: int | None = None):
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.obs_noise_std = obs_noise_std
        self.level = None
        self.trend = None
        self.rng = np.random.default_rng(seed)

    def sync(self, real_state: np.ndarray) -> np.ndarray:
        obs = real_state + self.rng.normal(0, self.obs_noise_std, self.dim)
        if self.level is None:
            self.level = obs.copy()
            self.trend = np.zeros(self.dim)
        else:
            prev_level = self.level.copy()
            self.level = self.alpha * obs + (1 - self.alpha) * (self.level + self.trend)
            self.trend = self.beta * (self.level - prev_level) + (1 - self.beta) * self.trend
        return self.level

    def predict(self, horizon: int = 1) -> np.ndarray:
        if self.level is None:
            return np.zeros(self.dim)
        return self.level + horizon * self.trend


# --------------------------------------------------------------------------- #
# Vision-Language-Action stub module
# --------------------------------------------------------------------------- #

class VLAPipeline:
    """
    STUB / INTERFACE for a Vision-Language-Action model, structured as an
    explicit three-stage ENCODE -> PROCESS -> DECODE pipeline around the
    information a vehicle transmits:

      encode()  Compresses the vehicle's raw transmitted payload (its V2X
                message: position, speed, queue state, a raw hazard sensor
                reading, ...) jointly with a camera-derived vision embedding
                into a compact semantic latent code z. This is what
                "transmission" means here in a task-oriented / semantic-
                communication sense: the latent z, not the raw payload, is
                what conceptually goes over the air and into the VLA.

      process() The VLA core itself: takes z plus a language embedding
                describing the vehicle's current V2X application context
                (platooning / intersection monitoring / VRU detection) and
                produces (a) a discrete driving intent and (b) a continuous
                action-conditioning embedding h.

      decode()  Reconstructs, at the RSU/edge side, (a) an estimate of the
                original message (so we can measure semantic reconstruction
                fidelity) and (b) a decoded action hint -- a small
                continuous vector downstream scheduling policies can
                condition on, standing in for whatever structured command a
                real VLA's action head would emit.

    All three stages are small fixed (untrained-by-RL) NumPy MLPs standing
    in for a real pretrained encoder / VLA-core / decoder stack. Swap any
    one stage for a real model and the rest of the pipeline is unaffected
    as long as you keep the same tensor shapes at each boundary.
    """

    INTENTS = ["normal", "hazard_alert", "maneuver"]

    def __init__(self, message_dim: int = 6, vision_dim: int = 8, lang_dim: int = 3,
                 latent_dim: int = 6, hidden: int = 8, action_dim: int = 2,
                 seed: int | None = None):
        rng = np.random.default_rng(seed)
        self.message_dim = message_dim
        self.action_dim = action_dim

        # ---- 1) ENCODE: (message, vision) -> latent semantic code z ----
        enc_in = message_dim + vision_dim
        self.We = rng.normal(0, 0.15, (enc_in, latent_dim))
        self.be = np.zeros(latent_dim)
        # Clean pass-through channel: latent z[0] mirrors vision_embed[0]
        # (the ground-truth hazard signal, see V2XEnv.run_vla()) exactly, so
        # the hand-wired hazard detector in process() has a clean signal to
        # key off even after compression.
        self.We[:, 0] = 0.0
        self.We[message_dim + 0, 0] = 1.0

        # ---- 2) PROCESS: VLA core, (z, language) -> intent, action embedding h ----
        core_in = latent_dim + lang_dim
        self.W1 = rng.normal(0, 0.1, (core_in, hidden))
        self.b1 = rng.normal(0, 0.1, hidden)
        self.W2 = rng.normal(0, 0.1, (hidden, len(self.INTENTS)))
        haz_idx = self.INTENTS.index("hazard_alert")
        nrm_idx = self.INTENTS.index("normal")
        self.b2 = np.zeros(len(self.INTENTS))
        self.b2[nrm_idx] = 0.5     # default preference for "normal"
        self.b2[haz_idx] = -1.5    # hazard_alert must be earned by evidence
        self.W1[:, 0] = 0.0
        self.W1[0, 0] = 6.0        # z[0] (hazard channel) -> detector unit
        self.b1[0] = -3.0          # threshold noise rarely clears on its own
        self.W2[0, :] = 0.0
        self.W2[0, haz_idx] = 4.0
        self.W2[0, nrm_idx] = -2.0

        # ---- 3) DECODE: action embedding h -> (reconstructed message, decoded action hint) ----
        self.Wd_msg = rng.normal(0, 0.15, (hidden, message_dim))
        self.bd_msg = np.zeros(message_dim)
        self.Wd_act = rng.normal(0, 0.3, (hidden, action_dim))
        self.bd_act = np.zeros(action_dim)

    def encode(self, message: np.ndarray, vision_embed: np.ndarray) -> np.ndarray:
        x = np.concatenate([message, vision_embed])
        return x @ self.We + self.be

    def process(self, z: np.ndarray, lang_embed: np.ndarray, rng: np.random.Generator):
        x = np.concatenate([z, lang_embed])
        h = relu(x @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        probs = softmax(logits)
        intent_idx = int(rng.choice(len(probs), p=probs))
        return intent_idx, probs, h

    def decode(self, h: np.ndarray):
        recon_message = np.tanh(h @ self.Wd_msg + self.bd_msg)
        decoded_action = np.tanh(h @ self.Wd_act + self.bd_act)
        return recon_message, decoded_action

    def run(self, message: np.ndarray, vision_embed: np.ndarray, lang_embed: np.ndarray,
            rng: np.random.Generator):
        """Full encode -> process -> decode pass for one vehicle, one step."""
        z = self.encode(message, vision_embed)
        intent_idx, probs, h = self.process(z, lang_embed, rng)
        recon_message, decoded_action = self.decode(h)
        return intent_idx, probs, recon_message, decoded_action


# --------------------------------------------------------------------------- #
# Edge server (MEC) with processor-sharing task queue
# --------------------------------------------------------------------------- #

class EdgeServer:
    def __init__(self, capacity_cycles_per_step: float):
        self.capacity_base = capacity_cycles_per_step
        self.capacity = capacity_cycles_per_step   # effective capacity; a disaster can knock this down
        self.tasks = []  # list of [remaining_cycles, deadline_steps_left, owner_idx]

    def enqueue(self, cycles: float, deadline_steps: int, owner_idx: int):
        self.tasks.append([cycles, deadline_steps, owner_idx])

    def step(self):
        """Advance all queued tasks by one step (processor sharing).
        Returns (completed_owner_idxs, failed_owner_idxs)."""
        completed, failed = [], []
        if self.tasks:
            share = self.capacity / len(self.tasks)
            remaining = []
            for cycles, deadline, owner in self.tasks:
                cycles -= share
                deadline -= 1
                if cycles <= 0:
                    completed.append(owner)
                elif deadline <= 0:
                    failed.append(owner)
                else:
                    remaining.append([cycles, deadline, owner])
            self.tasks = remaining
        return completed, failed

    def queue_len(self) -> int:
        return len(self.tasks)
