"""
gateway/app.py
==============
Industrial e-Nose — Gateway Service

This single-file service runs three concurrent components:

1. MQTT Subscriber (asyncio)
   - Connects to Mosquitto broker
   - Decodes JSON sensor snapshots from topic ``enose/sensors/raw``
   - Feeds decoded data into an in-memory circular buffer (deque)

2. Data Reconciliation Engine
   - Drift detection: CUSUM algorithm on rolling R_ratio baseline
   - Outlier rejection: Hampel identifier (MAD-based) on each channel
   - Sensor health scoring: cumulative anomaly counter per channel

3. FastAPI REST endpoints + Streamlit dashboard
   - GET /api/v1/latest        — most recent reconciled snapshot
   - GET /api/v1/history?n=N  — last N snapshots
   - GET /api/v1/health       — sensor health status
   - POST /api/v1/recalibrate — send MQTT recalibration command to device
   - GET /dashboard            — Streamlit dashboard (served separately)

Launch instructions
-------------------
    # Full service (API + MQTT + dashboard in separate processes):
    docker-compose up

    # Development:
    uvicorn gateway.app:app --host 0.0.0.0 --port 8000 --reload

Dependencies: see requirements.txt
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

import numpy as np
import paho.mqtt.client as mqtt
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("e-nose.gateway")

# ---------------------------------------------------------------------------
# Configuration (from environment or defaults)
# ---------------------------------------------------------------------------

MQTT_BROKER   = os.getenv("MQTT_BROKER",   "mosquitto")
MQTT_PORT     = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USER     = os.getenv("MQTT_USER",     "enose")
MQTT_PASS     = os.getenv("MQTT_PASS",     "enose_secret")
MQTT_TOPIC_RAW   = "enose/sensors/raw"
MQTT_TOPIC_RECAL = "enose/cmd/recalibrate"
MQTT_TOPIC_STATUS = "enose/sensors/status"

N_SENSORS       = 4
HISTORY_MAXLEN  = 3000       # ~10 min at 200 ms intervals
CUSUM_H         = 5.0        # CUSUM decision threshold (in σ units)
CUSUM_K         = 0.5        # Allowance (slack) parameter — k = δ/2σ
HAMPEL_WINDOW   = 15         # Hampel identifier window (samples each side)
HAMPEL_NSIGMA   = 3.5        # Rejection threshold in MAD units
DRIFT_WARN_THR  = 0.05       # 5% cumulative drift flags a warning
DRIFT_CRIT_THR  = 0.15       # 15% cumulative drift flags a critical alert

GAS_NAMES = {
    0: "clean_air",
    1: "ethanol",
    2: "ammonia",
    3: "co",
    4: "acetone",
}
GAS_LABELS = {v: k for k, v in GAS_NAMES.items()}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ChannelStatus:
    channel_id: int
    r_ratio: float = 1.0
    resistance_ohm: float = 0.0
    baseline_r_air: float = 0.0
    ema_filtered: float = 0.0
    kalman_filtered: float = 0.0
    heater_phase: str = "unknown"
    heater_duty: int = 0
    baseline_locked: bool = False
    drift_cusum_pos: float = 0.0      # CUSUM positive accumulator
    drift_cusum_neg: float = 0.0      # CUSUM negative accumulator
    drift_status: str = "ok"          # ok | warn | critical
    drift_magnitude: float = 0.0      # Estimated drift fraction
    outlier_count_recent: int = 0
    health_score: float = 1.0         # [0.0, 1.0] — 1.0 = perfect


@dataclass
class ReconciledSnapshot:
    snapshot_id: int
    timestamp_utc: str
    device_id: str
    channels: List[ChannelStatus]
    predicted_gas: str = "unknown"
    predicted_gas_id: int = -1
    concentration_ppm: float = 0.0
    model_confidence: float = 0.0
    all_baselines_locked: bool = False
    anomaly_flags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Drift Detection — CUSUM algorithm
# ---------------------------------------------------------------------------

class CUSUMDetector:
    """
    Two-sided CUSUM detector for sensor drift monitoring.

    A CUSUM chart accumulates deviations from a reference value μ₀ beyond
    an allowance k (in σ units). When the positive or negative accumulator
    exceeds a threshold H, a drift event is declared.

    Parameters
    ----------
    mu0 : float
        Expected in-control mean (reference R_ratio = 1.0 for clean air).
    sigma : float
        In-control standard deviation (estimated from calibration window).
    k : float
        Allowance parameter k = δ/(2σ), where δ is the smallest shift to detect.
    H : float
        Decision threshold in σ units (typically 4–6).
    """

    def __init__(self, mu0: float = 1.0, sigma: float = 0.02,
                 k: float = CUSUM_K, H: float = CUSUM_H):
        self.mu0   = mu0
        self.sigma = sigma
        self.k     = k
        self.H     = H
        self._s_pos = 0.0
        self._s_neg = 0.0

    def update(self, x: float) -> tuple[float, float]:
        """
        Update CUSUM with new observation.

        Returns
        -------
        (s_pos, s_neg) — current accumulator values (compared against H)
        """
        z           = (x - self.mu0) / (self.sigma + 1e-9)
        self._s_pos = max(0.0, self._s_pos + z - self.k)
        self._s_neg = max(0.0, self._s_neg - z - self.k)
        return self._s_pos, self._s_neg

    def drift_status(self, s_pos: float, s_neg: float) -> str:
        combined = max(s_pos, s_neg)
        if combined > self.H * (DRIFT_CRIT_THR / DRIFT_WARN_THR):
            return "critical"
        elif combined > self.H:
            return "warn"
        return "ok"

    def reset(self) -> None:
        self._s_pos = 0.0
        self._s_neg = 0.0


# ---------------------------------------------------------------------------
# Outlier Detection — Hampel Identifier
# ---------------------------------------------------------------------------

class HampelFilter:
    """
    Online Hampel identifier for scalar time-series.

    The Hampel identifier replaces a sample x_i with the local median when:
        |x_i - median(window)| > nsigma * 1.4826 * MAD(window)

    The constant 1.4826 makes MAD a consistent estimator of σ for Gaussian data.

    Maintains a sliding window of size 2*half_width+1.
    """

    MAD_SCALE = 1.4826  # Consistency factor for normal distribution

    def __init__(self, half_width: int = HAMPEL_WINDOW, nsigma: float = HAMPEL_NSIGMA):
        self.half_width = half_width
        self.nsigma     = nsigma
        self._window: Deque[float] = collections.deque(maxlen=2 * half_width + 1)
        self.outlier_count = 0

    def push(self, x: float) -> tuple[float, bool]:
        """
        Push a new sample. Returns (cleaned_value, is_outlier).
        """
        self._window.append(x)

        if len(self._window) < self.half_width + 1:
            return x, False

        arr    = np.array(self._window)
        median = float(np.median(arr))
        mad    = float(np.median(np.abs(arr - median)))
        sigma_hat = self.MAD_SCALE * mad

        is_outlier = abs(x - median) > self.nsigma * sigma_hat + 1e-12

        if is_outlier:
            self.outlier_count += 1
            return median, True

        return x, False


# ---------------------------------------------------------------------------
# Shared state (thread-safe)
# ---------------------------------------------------------------------------

class GatewayState:
    """
    Thread-safe shared state for the gateway.
    All fields accessed from both MQTT callback thread and FastAPI handlers.
    """

    def __init__(self):
        self._lock         = threading.Lock()
        self._history: Deque[ReconciledSnapshot] = collections.deque(maxlen=HISTORY_MAXLEN)
        self._latest: Optional[ReconciledSnapshot] = None
        self._channel_status: Dict[int, ChannelStatus] = {
            i: ChannelStatus(channel_id=i) for i in range(N_SENSORS)
        }
        self._cusum: Dict[int, CUSUMDetector] = {
            i: CUSUMDetector() for i in range(N_SENSORS)
        }
        self._hampel: Dict[int, HampelFilter] = {
            i: HampelFilter() for i in range(N_SENSORS)
        }
        # Rolling buffer of R_ratios for sigma estimation (used in CUSUM)
        self._ratio_buffers: Dict[int, Deque[float]] = {
            i: collections.deque(maxlen=100) for i in range(N_SENSORS)
        }
        self._message_count  = 0
        self._start_time     = time.time()
        self._last_device_id = "unknown"

    def process_raw_message(self, payload: dict) -> Optional[ReconciledSnapshot]:
        """
        Parse raw MQTT payload, apply reconciliation, and store result.
        Thread-safe.
        """
        try:
            snapshot_id = int(payload.get("snapshot_id", 0))
            sensors_raw = payload.get("sensors", [])
            all_baselines = bool(payload.get("all_baselines", False))

            channels_out    : List[ChannelStatus] = []
            anomaly_flags   : List[str]            = []
            r_ratios_clean  : List[float]          = []

            with self._lock:
                self._message_count += 1

                for s in sensors_raw:
                    ch     = int(s.get("ch", 0))
                    r_raw  = float(s.get("r_ratio", 1.0))
                    r_ohm  = float(s.get("r_gas",   0.0))
                    r_air  = float(s.get("r_air",   0.0))
                    ema    = float(s.get("ema",      0.0))
                    kalman = float(s.get("kalman",   0.0))
                    phase  = str(s.get("phase",      "unknown"))
                    duty   = int(s.get("duty",       0))
                    b_lock = bool(s.get("baseline",  False))

                    if ch >= N_SENSORS:
                        continue

                    # 1. Hampel outlier rejection
                    r_clean, is_outlier = self._hampel[ch].push(r_raw)
                    if is_outlier:
                        anomaly_flags.append(f"outlier_ch{ch}")

                    r_ratios_clean.append(r_clean)

                    # 2. Update CUSUM for drift detection
                    # Estimate sigma from rolling buffer if enough data
                    self._ratio_buffers[ch].append(r_clean)
                    buf = list(self._ratio_buffers[ch])
                    if len(buf) >= 20:
                        sigma_est = float(np.std(buf))
                        self._cusum[ch].sigma = max(sigma_est, 1e-4)

                    s_pos, s_neg = self._cusum[ch].update(r_clean)
                    drift_stat   = self._cusum[ch].drift_status(s_pos, s_neg)
                    drift_mag    = max(s_pos, s_neg) / (CUSUM_H + 1e-6)

                    if drift_stat == "critical":
                        anomaly_flags.append(f"drift_critical_ch{ch}")
                    elif drift_stat == "warn":
                        anomaly_flags.append(f"drift_warn_ch{ch}")

                    # 3. Health score: combination of drift and outlier rate
                    recent_outliers = self._hampel[ch].outlier_count
                    outlier_rate    = min(1.0, recent_outliers / (self._message_count + 1))
                    health = max(0.0, 1.0
                                 - 0.5 * min(drift_mag, 1.0)
                                 - 0.5 * outlier_rate)

                    cs = ChannelStatus(
                        channel_id         = ch,
                        r_ratio            = r_clean,
                        resistance_ohm     = r_ohm,
                        baseline_r_air     = r_air,
                        ema_filtered       = ema,
                        kalman_filtered    = kalman,
                        heater_phase       = phase,
                        heater_duty        = duty,
                        baseline_locked    = b_lock,
                        drift_cusum_pos    = s_pos,
                        drift_cusum_neg    = s_neg,
                        drift_status       = drift_stat,
                        drift_magnitude    = drift_mag,
                        outlier_count_recent = recent_outliers,
                        health_score       = health,
                    )
                    self._channel_status[ch] = cs
                    channels_out.append(cs)

                # 4. Simple rule-based gas inference from R_ratios
                # (placeholder — in production: call embedded C model via ctypes)
                predicted_gas, confidence, ppm = self._infer_gas(r_ratios_clean)

                snap = ReconciledSnapshot(
                    snapshot_id          = snapshot_id,
                    timestamp_utc        = datetime.now(timezone.utc).isoformat(),
                    device_id            = self._last_device_id,
                    channels             = channels_out,
                    predicted_gas        = predicted_gas,
                    predicted_gas_id     = GAS_LABELS.get(predicted_gas, -1),
                    concentration_ppm    = ppm,
                    model_confidence     = confidence,
                    all_baselines_locked = all_baselines,
                    anomaly_flags        = list(set(anomaly_flags)),
                )

                self._latest  = snap
                self._history.append(snap)
                return snap

        except (KeyError, ValueError, TypeError) as exc:
            log.warning("Failed to parse MQTT payload: %s — %s", exc, payload)
            return None

    def _infer_gas(self, r_ratios: List[float]) -> tuple[str, float, float]:
        """
        Heuristic gas inference from steady-state R_ratios.
        In production this calls the exported C model via ctypes.

        Returns (gas_name, confidence, ppm_estimate)
        """
        if not r_ratios or len(r_ratios) < N_SENSORS:
            return "unknown", 0.0, 0.0

        r = np.array(r_ratios[:N_SENSORS])
        mean_r = float(np.mean(r))

        # All ratios near 1.0 → clean air
        if mean_r > 0.90:
            return "clean_air", 0.95, 0.0

        # Selectivity patterns based on the sensitivity matrix from train.py
        # ch0 dominant + ch2 moderate → ethanol signature
        if r[0] < 0.65 and r[1] > r[0] and r[2] > r[0]:
            conf = min(1.0, (1.0 - r[0]) * 2.0)
            ppm  = max(0.0, (0.65 - r[0]) / 0.012 * 100.0)
            return "ethanol", conf, ppm

        # ch1 dominant → ammonia
        if r[1] < 0.65 and r[3] < 0.75:
            conf = min(1.0, (1.0 - r[1]) * 2.0)
            ppm  = max(0.0, (0.65 - r[1]) / 0.015 * 60.0)
            return "ammonia", conf, ppm

        # ch2 dominant → CO
        if r[2] < 0.70 and r[0] > r[2]:
            conf = min(1.0, (1.0 - r[2]) * 2.0)
            ppm  = max(0.0, (0.70 - r[2]) / 0.010 * 50.0)
            return "co", conf, ppm

        # ch3 dominant → acetone
        if r[3] < 0.65:
            conf = min(1.0, (1.0 - r[3]) * 2.0)
            ppm  = max(0.0, (0.65 - r[3]) / 0.012 * 200.0)
            return "acetone", conf, ppm

        return "unknown", 0.3, 0.0

    def get_latest(self) -> Optional[ReconciledSnapshot]:
        with self._lock:
            return self._latest

    def get_history(self, n: int = 100) -> List[ReconciledSnapshot]:
        with self._lock:
            items = list(self._history)
            return items[-n:] if n < len(items) else items

    def get_channel_health(self) -> Dict[int, dict]:
        with self._lock:
            return {
                ch: {
                    "health_score": cs.health_score,
                    "drift_status": cs.drift_status,
                    "drift_magnitude": cs.drift_magnitude,
                    "outlier_count": cs.outlier_count_recent,
                }
                for ch, cs in self._channel_status.items()
            }

    def set_device_id(self, device_id: str) -> None:
        with self._lock:
            self._last_device_id = device_id

    def get_stats(self) -> dict:
        with self._lock:
            uptime = time.time() - self._start_time
            return {
                "uptime_s": round(uptime, 1),
                "message_count": self._message_count,
                "msg_per_sec": round(self._message_count / max(uptime, 1.0), 2),
                "history_len": len(self._history),
            }


# ---------------------------------------------------------------------------
# MQTT Client
# ---------------------------------------------------------------------------

class MQTTGateway:
    """
    Paho MQTT client that subscribes to sensor topics and feeds GatewayState.
    Runs in its own thread via loop_start().
    """

    def __init__(self, state: GatewayState):
        self._state  = state
        self._client = mqtt.Client(client_id="gateway-001", protocol=mqtt.MQTTv311)
        self._client.username_pw_set(MQTT_USER, MQTT_PASS)
        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message    = self._on_message

        # LWT (Last Will Testament) — gateway offline notification
        self._client.will_set("enose/gateway/status", "offline", qos=1, retain=True)

    def start(self) -> None:
        log.info("MQTT connecting to %s:%d", MQTT_BROKER, MQTT_PORT)
        self._client.connect_async(MQTT_BROKER, MQTT_PORT, keepalive=30)
        self._client.loop_start()

    def stop(self) -> None:
        self._client.publish("enose/gateway/status", "offline", retain=True)
        self._client.loop_stop()
        self._client.disconnect()

    def send_recalibrate(self) -> None:
        self._client.publish(MQTT_TOPIC_RECAL, "1", qos=1)
        log.info("Sent recalibrate command")

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            log.info("MQTT connected (rc=0)")
            client.subscribe(MQTT_TOPIC_RAW,    qos=0)
            client.subscribe(MQTT_TOPIC_STATUS, qos=0)
            client.publish("enose/gateway/status", "online", retain=True)
        else:
            log.error("MQTT connection failed rc=%d", rc)

    def _on_disconnect(self, client, userdata, rc):
        if rc != 0:
            log.warning("MQTT unexpected disconnect (rc=%d) — will auto-reconnect", rc)

    def _on_message(self, client, userdata, msg):
        try:
            topic = msg.topic
            raw   = msg.payload.decode("utf-8", errors="replace")

            if topic == MQTT_TOPIC_STATUS:
                log.debug("Device status: %s", raw)
                try:
                    status = json.loads(raw)
                    if "device" in status:
                        self._state.set_device_id(status["device"])
                except json.JSONDecodeError:
                    pass
                return

            if topic == MQTT_TOPIC_RAW:
                payload = json.loads(raw)
                self._state.process_raw_message(payload)

        except json.JSONDecodeError as exc:
            log.warning("JSON decode error: %s — raw=%s", exc, msg.payload[:80])
        except Exception as exc:
            log.error("Unexpected error in MQTT callback: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

state   = GatewayState()
mqtt_gw = MQTTGateway(state)

app = FastAPI(
    title       = "e-Nose Industrial Gateway",
    description = "REST API for the industrial electronic nose monitoring system",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


@app.on_event("startup")
async def startup_event():
    log.info("Starting e-Nose Gateway service...")
    mqtt_gw.start()
    log.info("MQTT gateway started")


@app.on_event("shutdown")
async def shutdown_event():
    log.info("Shutting down MQTT gateway...")
    mqtt_gw.stop()


@app.get("/api/v1/latest", summary="Latest reconciled sensor snapshot")
async def get_latest():
    snap = state.get_latest()
    if snap is None:
        raise HTTPException(status_code=503, detail="No data received yet")
    return _snapshot_to_dict(snap)


@app.get("/api/v1/history", summary="Last N reconciled snapshots")
async def get_history(n: int = Query(default=100, ge=1, le=HISTORY_MAXLEN)):
    items = state.get_history(n)
    return {"count": len(items), "snapshots": [_snapshot_to_dict(s) for s in items]}


@app.get("/api/v1/health", summary="Sensor array health status")
async def get_health():
    return {
        "gateway_stats": state.get_stats(),
        "channel_health": state.get_channel_health(),
    }


@app.post("/api/v1/recalibrate", summary="Trigger baseline recalibration on device")
async def post_recalibrate():
    mqtt_gw.send_recalibrate()
    return {"status": "recalibrate_command_sent"}


@app.get("/", summary="Service info")
async def root():
    return {
        "service":  "e-Nose Industrial Gateway",
        "version":  "1.0.0",
        "docs":     "/docs",
        "redoc":    "/redoc",
        "stats":    state.get_stats(),
    }


def _snapshot_to_dict(snap: ReconciledSnapshot) -> dict:
    """Convert ReconciledSnapshot to JSON-serializable dict."""
    return {
        "snapshot_id":          snap.snapshot_id,
        "timestamp_utc":        snap.timestamp_utc,
        "device_id":            snap.device_id,
        "predicted_gas":        snap.predicted_gas,
        "predicted_gas_id":     snap.predicted_gas_id,
        "concentration_ppm":    round(snap.concentration_ppm, 2),
        "model_confidence":     round(snap.model_confidence, 4),
        "all_baselines_locked": snap.all_baselines_locked,
        "anomaly_flags":        snap.anomaly_flags,
        "channels": [
            {
                "ch":                ch.channel_id,
                "r_ratio":           round(ch.r_ratio, 5),
                "resistance_ohm":    round(ch.resistance_ohm, 1),
                "baseline_r_air":    round(ch.baseline_r_air, 1),
                "heater_phase":      ch.heater_phase,
                "heater_duty":       ch.heater_duty,
                "baseline_locked":   ch.baseline_locked,
                "drift_status":      ch.drift_status,
                "drift_cusum_pos":   round(ch.drift_cusum_pos, 4),
                "drift_cusum_neg":   round(ch.drift_cusum_neg, 4),
                "drift_magnitude":   round(ch.drift_magnitude, 4),
                "health_score":      round(ch.health_score, 4),
                "outlier_count":     ch.outlier_count_recent,
            }
            for ch in snap.channels
        ],
    }
