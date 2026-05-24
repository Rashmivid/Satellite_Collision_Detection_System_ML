"""
stgnn_bridge.py  —  Rashmi's STGNN risk scorer
------------------------------------------------
Drop this file into Shivansh's repo root (F:\Satellite_collision_detection_system).
Copy best_stgnn_model.pth and feature_config.json there too.

Usage (from app.py):
    from stgnn_bridge import STGNNBridge
    bridge = STGNNBridge()                        # loads model once at startup
    enriched = bridge.score_conjunctions(conjunctions, sats_by_name)
"""

import os
import json
import math
import numpy as np

# ── Optional PyTorch import (graceful fallback to heuristic scoring) ──────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[stgnn_bridge] PyTorch not found — using heuristic fallback scorer.")


# ── Model definition (must match your training architecture exactly) ──────────
class SpatioTemporalGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=3, dropout=0.3):
        super().__init__()
        self.spatial_layer1 = nn.Linear(input_dim, hidden_dim)
        self.bn1            = nn.BatchNorm1d(hidden_dim)
        self.dropout1       = nn.Dropout(dropout)

        self.spatial_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2            = nn.BatchNorm1d(hidden_dim)
        self.dropout2       = nn.Dropout(dropout)

        self.temporal_layer = nn.Linear(hidden_dim, hidden_dim)
        self.bn3            = nn.BatchNorm1d(hidden_dim)
        self.dropout3       = nn.Dropout(dropout)

        self.graph_layer = nn.Linear(hidden_dim, hidden_dim)
        self.bn4         = nn.BatchNorm1d(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.spatial_layer1(x))); x = self.dropout1(x)
        x = F.relu(self.bn2(self.spatial_layer2(x))); x = self.dropout2(x)
        x = F.relu(self.bn3(self.temporal_layer(x))); x = self.dropout3(x)
        x = F.relu(self.bn4(self.graph_layer(x)))
        return self.classifier(x)


# ── Orbital helpers (no skyfield dependency needed here) ──────────────────────
MU          = 398600.4418   # km³/s²
EARTH_R     = 6371.0        # km
SECS_PER_DAY = 86400.0


def _orbital_params(sat_tle_obj):
    """
    Extract the features your model was trained on from a Skyfield
    EarthSatellite object.  Returns a dict keyed by feature name.
    """
    model   = sat_tle_obj.model          # sgp4lib.Satrec
    # Mean motion: rev/day  →  keep as-is (matches your training CSV column MEAN_MOTION)
    mm      = model.no_kozai * (SECS_PER_DAY / (2 * math.pi))   # rev/day
    ecc     = model.ecco
    inc_deg = math.degrees(model.inclo)
    raan    = math.degrees(model.nodeo)
    argp    = math.degrees(model.argpo)
    ma      = math.degrees(model.mo)
    bstar   = model.bstar
    mm_dot  = model.ndot  * (SECS_PER_DAY**2 / (2 * math.pi))  # approx
    mm_ddot = model.nddot * (SECS_PER_DAY**3 / (2 * math.pi))

    # Derived
    n_rad_s       = mm * 2 * math.pi / SECS_PER_DAY
    sma           = (MU / (n_rad_s ** 2)) ** (1 / 3)
    perigee_alt   = sma * (1 - ecc) - EARTH_R
    apogee_alt    = sma * (1 + ecc) - EARTH_R
    mean_alt      = (perigee_alt + apogee_alt) / 2
    period_min    = 1440.0 / mm
    velocity      = math.sqrt(MU / sma)

    def wrap_sin(deg): return math.sin(math.radians(deg))
    def wrap_cos(deg): return math.cos(math.radians(deg))

    # epoch timestamp — use jdsatepoch converted to POSIX seconds (approx)
    epoch_ts = (model.jdsatepoch - 2440587.5) * SECS_PER_DAY
    import datetime as _dt
    epoch_dt = _dt.datetime(2000, 1, 1) + _dt.timedelta(days=model.jdsatepoch - 2451545.0)
    day_of_year = epoch_dt.timetuple().tm_yday

    # Orbit type encoding  (LEO=0, MEO=1, GEO=2)
    orbit_enc = 0 if mean_alt < 2000 else (1 if mean_alt < 35786 else 2)

    return {
        "MEAN_MOTION":           mm,
        "ECCENTRICITY":          ecc,
        "SEMI_MAJOR_AXIS":       sma,
        "PERIGEE_ALT":           perigee_alt,
        "APOGEE_ALT":            apogee_alt,
        "MEAN_ALT":              mean_alt,
        "ORBITAL_PERIOD":        period_min,
        "VELOCITY":              velocity,
        "INCLINATION_SIN":       wrap_sin(inc_deg),
        "INCLINATION_COS":       wrap_cos(inc_deg),
        "RA_OF_ASC_NODE_SIN":    wrap_sin(raan),
        "RA_OF_ASC_NODE_COS":    wrap_cos(raan),
        "ARG_OF_PERICENTER_SIN": wrap_sin(argp),
        "ARG_OF_PERICENTER_COS": wrap_cos(argp),
        "MEAN_ANOMALY_SIN":      wrap_sin(ma),
        "MEAN_ANOMALY_COS":      wrap_cos(ma),
        "BSTAR":                 bstar,
        "MEAN_MOTION_DOT":       mm_dot,
        "MEAN_MOTION_DDOT":      mm_ddot,
        "EPOCH_TIMESTAMP":       epoch_ts,
        "DAY_OF_YEAR":           float(day_of_year),
        "ORBIT_TYPE_ENCODED":    float(orbit_enc),
    }


# ── Main bridge class ─────────────────────────────────────────────────────────
class STGNNBridge:
    RISK_LABELS = ["SAFE", "MODERATE_RISK", "HIGH_RISK"]

    def __init__(self,
                 model_path: str = "best_stgnn_model.pth",
                 feature_config_path: str = "feature_config.json"):

        self.model          = None
        self.feature_names  = None
        self.device         = None
        self._loaded        = False

        if not TORCH_AVAILABLE:
            print("[stgnn_bridge] PyTorch unavailable — heuristic mode active.")
            return

        # Load feature order
        config_path = os.path.join(os.path.dirname(__file__), feature_config_path)
        model_file  = os.path.join(os.path.dirname(__file__), model_path)

        if not os.path.exists(config_path):
            print(f"[stgnn_bridge] Warning: {feature_config_path} not found — heuristic mode.")
            return
        if not os.path.exists(model_file):
            print(f"[stgnn_bridge] Warning: {model_path} not found — heuristic mode.")
            return

        with open(config_path) as f:
            cfg = json.load(f)
        self.feature_names = cfg["all_features"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = SpatioTemporalGNN(
            input_dim=len(self.feature_names), hidden_dim=64, num_classes=3
        )
        self.model.load_state_dict(
            torch.load(model_file, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        print(f"[stgnn_bridge] STGNN loaded ({len(self.feature_names)} features, device={self.device})")

    # ── Public API ────────────────────────────────────────────────────────────
    def score_conjunctions(self, conjunctions: list, sats_by_name: dict) -> list:
        """
        Enrich each conjunction dict with STGNN risk fields.

        Parameters
        ----------
        conjunctions : list of dicts from detector.find_conjunctions()
            Required keys: sat1, sat2, distance_km, time
        sats_by_name : dict  {satellite_name: Skyfield EarthSatellite}

        Returns
        -------
        Same list with added keys:
            stgnn_risk          str   "SAFE" | "MODERATE_RISK" | "HIGH_RISK"
            prob_safe           float 0-1
            prob_moderate       float 0-1
            prob_high           float 0-1
            confidence          float 0-100
            risk_color          str   hex color for UI
        """
        enriched = []
        for c in conjunctions:
            sat1_obj = sats_by_name.get(c["sat1"])
            sat2_obj = sats_by_name.get(c["sat2"])

            if self._loaded and sat1_obj and sat2_obj:
                risk, probs = self._predict_pair(sat1_obj, sat2_obj)
            else:
                risk, probs = self._heuristic(c["distance_km"])

            prob_safe, prob_mod, prob_high = probs
            color = (
                "#ef4444" if risk == "HIGH_RISK" else
                "#f59e0b" if risk == "MODERATE_RISK" else
                "#22c55e"
            )
            enriched.append({
                **c,
                "stgnn_risk":     risk,
                "prob_safe":      round(float(prob_safe), 4),
                "prob_moderate":  round(float(prob_mod),  4),
                "prob_high":      round(float(prob_high), 4),
                "confidence":     round(float(max(probs)) * 100, 1),
                "risk_color":     color,
            })
        return enriched

    def score_single(self, sat_obj) -> dict:
        """
        Score a single satellite.
        Returns stgnn_risk, prob_safe, prob_moderate, prob_high, confidence, risk_color.
        """
        if self._loaded and sat_obj:
            risk, probs = self._predict_single(sat_obj)
        else:
            risk, probs = "SAFE", (1.0, 0.0, 0.0)

        color = (
            "#ef4444" if risk == "HIGH_RISK" else
            "#f59e0b" if risk == "MODERATE_RISK" else
            "#22c55e"
        )
        return {
            "stgnn_risk":    risk,
            "prob_safe":     round(float(probs[0]), 4),
            "prob_moderate": round(float(probs[1]), 4),
            "prob_high":     round(float(probs[2]), 4),
            "confidence":    round(float(max(probs)) * 100, 1),
            "risk_color":    color,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _features_from_sat(self, sat_obj) -> np.ndarray:
        params = _orbital_params(sat_obj)
        vec = np.array(
            [params.get(f, 0.0) for f in self.feature_names],
            dtype=np.float32
        )
        return np.nan_to_num(vec, nan=0.0, posinf=1e10, neginf=-1e10)

    def _predict_single(self, sat_obj):
        feat = self._features_from_sat(sat_obj).reshape(1, -1)
        return self._run_model(feat)

    def _predict_pair(self, sat1_obj, sat2_obj):
        """
        Average the two satellites' feature vectors before scoring.
        This mirrors the pair-level analysis your training data used.
        """
        f1   = self._features_from_sat(sat1_obj)
        f2   = self._features_from_sat(sat2_obj)
        feat = ((f1 + f2) / 2.0).reshape(1, -1)
        return self._run_model(feat)

    def _run_model(self, feat_np: np.ndarray):
        tensor = torch.FloatTensor(feat_np).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs  = F.softmax(logits, dim=1).cpu().numpy()[0]
        idx  = int(np.argmax(probs))
        risk = self.RISK_LABELS[idx]
        return risk, probs

    @staticmethod
    def _heuristic(distance_km: float):
        """Simple fallback when model is unavailable."""
        if distance_km < 5:
            return "HIGH_RISK", (0.03, 0.07, 0.90)
        if distance_km < 15:
            return "HIGH_RISK", (0.10, 0.15, 0.75)
        if distance_km < 40:
            return "MODERATE_RISK", (0.20, 0.60, 0.20)
        return "SAFE", (0.85, 0.12, 0.03)