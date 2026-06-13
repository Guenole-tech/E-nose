# 🔬 e-Nose Industrial — Electronic Nose for Real-Time Gas Detection

> **Production-grade IoT platform** for industrial gas identification and concentration quantification using a 4-channel MOS sensor array, embedded ML inference, and a cloud-ready data pipeline.

[![Firmware](https://img.shields.io/badge/firmware-ESP32%20%7C%20C%2B%2B17-blue)]()
[![ML](https://img.shields.io/badge/ML-RandomForest%20%7C%20SVR-orange)]()
[![API](https://img.shields.io/badge/API-FastAPI%200.111-green)]()
[![MQTT](https://img.shields.io/badge/broker-Eclipse%20Mosquitto%202.0-purple)]()
[![Docker](https://img.shields.io/badge/deploy-Docker%20Compose-2496ED)]()

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Signal Acquisition & Digital Filters](#2-signal-acquisition--digital-filters)
3. [Machine Learning Pipeline](#3-machine-learning-pipeline)
4. [Data Reconciliation Engine](#4-data-reconciliation-engine)
5. [Project Structure](#5-project-structure)
6. [Firmware Build & Flash](#6-firmware-build--flash)
7. [Gateway Deployment](#7-gateway-deployment)
8. [API Reference](#8-api-reference)
9. [ML Training](#9-ml-training)
10. [Contributing](#10-contributing)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  SENSOR LAYER (ESP32)                                               │
│                                                                     │
│  MOS[0]──ADC1_CH0 ┐                                                 │
│  MOS[1]──ADC1_CH3 ├─→ Oversample x16 → EMA → Kalman → R_ratio     │
│  MOS[2]──ADC1_CH6 │         (sensor_driver.h/cpp)                  │
│  MOS[3]──ADC1_CH7 ┘                                                 │
│                      ↓                                              │
│  Heater PWM (LEDC) ──→ Detection / Regeneration FSM                │
│                      ↓                                              │
│  JSON snapshot → MQTT publish (200 ms) → WiFi → Mosquitto           │
└───────────────────────────────┬─────────────────────────────────────┘
                                │  MQTT (enose/sensors/raw)
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  GATEWAY LAYER (Docker — gateway/app.py)                            │
│                                                                     │
│  MQTT Subscriber                                                    │
│       ↓                                                             │
│  Hampel Outlier Filter  ──→ flag: outlier_chN                       │
│       ↓                                                             │
│  CUSUM Drift Detector   ──→ flag: drift_warn/critical_chN           │
│       ↓                                                             │
│  Gas Inference Engine   ──→ gas_label, ppm, confidence              │
│       ↓                                                             │
│  FastAPI REST + Streamlit Dashboard                                 │
└─────────────────────────────────────────────────────────────────────┘
```

**Communication topology:**
- ESP32 → Mosquitto: MQTT QoS 0, topic `enose/sensors/raw`, 5 Hz
- Gateway → Mosquitto: subscribes, publishes commands on `enose/cmd/*`
- Dashboard → Gateway: HTTP polling at 2 Hz via FastAPI REST

---

## 2. Signal Acquisition & Digital Filters

### 2.1 Oversampling

To increase effective ADC resolution beyond 12 bits, $N_{os} = 16$ samples are averaged per measurement:

$$\bar{x} = \frac{1}{N_{os}} \sum_{i=1}^{N_{os}} x_i$$

This yields an effective resolution of:

$$n_{eff} = n_{ADC} + \frac{1}{2}\log_2(N_{os}) = 12 + 2 = 14 \text{ bits}$$

### 2.2 Exponential Moving Average (EMA)

The EMA filter rejects high-frequency thermal noise while preserving the slow kinetics of MOS sensor response:

$$s_k = \alpha \cdot z_k + (1 - \alpha) \cdot s_{k-1}, \quad \alpha = 0.08$$

The −3 dB cutoff frequency at sampling rate $f_s = 5$ Hz:

$$f_c = \frac{f_s}{2\pi} \cdot \frac{\alpha}{\sqrt{(1-\alpha)^2}} \approx 0.064 \text{ Hz}$$

This is well-matched to MOS response times of 10–60 s.

### 2.3 Scalar Kalman Filter

The Kalman filter provides optimal linear state estimation under Gaussian noise. We model each sensor channel as a random-walk process:

**State model:**
$$x_k = x_{k-1} + w_k, \quad w_k \sim \mathcal{N}(0, Q)$$

**Observation model:**
$$z_k = x_k + v_k, \quad v_k \sim \mathcal{N}(0, R)$$

**Predict step:**
$$\hat{x}_{k|k-1} = \hat{x}_{k-1|k-1}$$
$$P_{k|k-1} = P_{k-1|k-1} + Q$$

**Update step:**

$$\begin{aligned}
K_k &= \frac{P_{k\mid k-1}}{P_{k\mid k-1} + R} \\
\hat{x}_{k\mid k} &= \hat{x}_{k\mid k-1} + K_k(z_k - \hat{x}_{k\mid k-1}) \\
P_{k\mid k} &= (1 - K_k)P_{k\mid k-1}
\end{aligned}$$

Tuning: $Q = 10^{-5}$, $R = 10^{-3}$. The steady-state Kalman gain is:
$$K_\infty = \frac{-R + \sqrt{R^2 + 4QR}}{2Q} \approx 0.095$$

### 2.4 Resistance Derivation & R_ratio

Using the voltage divider model (load resistor $R_L = 10\,\text{k}\Omega$):

$$R_{gas} = R_L \cdot \left(\frac{V_{cc}}{V_{out}} - 1\right) = R_L \cdot \left(\frac{\text{ADC}_{max}}{\text{ADC}_{val}} - 1\right)$$

The normalized ratio fed to the ML model:

$$\rho = \frac{R_{gas}}{R_{air}}$$

where $R_{air}$ is the baseline resistance measured in clean air over 200 calibration samples.

### 2.5 Heater Duty-Cycle FSM

```
   ┌──────────────────────────────────────────────────────┐
   │                                                      │
   ▼   t ≥ 60 s                         t ≥ 30 s         │
DETECTION ──────────────→ REGENERATION ──────────────────┘
(PWM=255)                   (PWM=80)
```

Regeneration at reduced heater power desorbs accumulated gas molecules without full thermal stress, extending sensor lifetime.

---

## 3. Machine Learning Pipeline

### 3.1 Feature Engineering

For each measurement window of $T = 60$ time steps, 7 features are extracted per sensor channel (28 total) plus 4 cross-channel ratios:

| Feature | Formula | Physical meaning |
|---------|---------|-----------------|
| `max_slope` | $\max_t \Delta\rho_t$ | Peak response rate |
| `min_slope` | $\min_t \Delta\rho_t$ | Recovery rate |
| `ss_mean` | $\displaystyle\frac{1}{10}\sum_{t=T-9}^{T}\rho_t$ | Equilibrium ratio |
| `ss_std` | $\sigma(\rho_{T-9:T})$ | Equilibrium stability |
| `auc_norm` | $\displaystyle\frac{1}{T}\int_0^T \rho\,dt$ | Cumulative response |
| `t_half` | $\displaystyle t \text{ s.t. } \rho(t) = \frac{1+\rho_{ss}}{2}$ | Kinetic signature |
| `peak_delta` | $\max_t \lvert\rho_t - 1\rvert$ | Peak sensitivity |
| `ratio_chA_chB` | $\rho_{ss,A} / \rho_{ss,B}$ | Cross-selectivity |

### 3.2 Gas Classifier — Random Forest

A **Random Forest** with 100 CART trees is trained for 5-class gas identification:

- **Ensemble:** Each tree trained on a bootstrapped subset of the training data (bagging)
- **Split criterion:** Gini impurity
- **Feature subsampling:** $\lfloor\sqrt{32}\rfloor = 5$ features per split
- **Class balancing:** `class_weight='balanced'` compensates for imbalanced gas frequencies in deployment

**Inference on microcontroller:**  
The trained forest is exported as a C header (`rf_model.h`) containing $N_{trees} = 100$ parallel arrays encoding left/right children, split features, thresholds, and leaf labels. Inference is a pure integer tree traversal — no floating-point matrix ops, no heap allocation:

```c
uint8_t rf_predict(const float* features, float* confidence_out);
// ~45 µs on ESP32 @ 240 MHz for 100 trees × depth 15
```

### 3.3 Concentration Regressor — SVR

One **Support Vector Regression** model per gas class (excluding clean air), using an RBF kernel:

$$K(\mathbf{x}, \mathbf{x'}) = \exp\!\left(-\gamma\,\|\mathbf{x} - \mathbf{x'}\|^2\right)$$

$$\hat{y} = \sum_{i \in \text{SV}} \alpha_i K(\mathbf{x}, \mathbf{x}_i) + b$$

Hyperparameters: $C = 100$, $\varepsilon = 0.5$, $\gamma = \text{scale}$.

**C export:** The support vectors, dual coefficients, $\gamma$, $b$, and scaler parameters are baked as `static const float` arrays in `svr_model.h`. The RBF kernel is evaluated using `expf()` from `<math.h>`.

### 3.4 Synthetic Data Generation

Training data is generated using a physics-based sensor response model:

$$\rho(t, C) = A \cdot e^{-BC} \cdot \left(1 - e^{-t/\tau}\right) + 1 \cdot e^{-t/\tau}$$

where:
- $A, B$ are gas-specific sensitivity parameters (calibrated to MQ-series datasheets)
- $C$ is the gas concentration in ppm
- $\tau \sim \mathcal{U}(20, 35)$ steps is the sensor rise time constant
- Multiplicative drift: $\rho_{\text{drift}} = \rho \cdot d$, $d \sim \mathcal{U}(1.0, 1.3)$
- Additive noise: $\sigma_\rho = 0.02 \cdot \rho_{\text{drift}}$

---

## 4. Data Reconciliation Engine

### 4.1 Hampel Outlier Identifier

For each sensor channel, a sliding window Hampel filter rejects impulse noise:

$$\hat{\sigma}_k = 1.4826 \cdot \text{MAD}\!\left(\rho_{k-W:k+W}\right)$$

Sample $\rho_k$ is replaced by the local median if:

$$|\rho_k - \tilde{\rho}| > n_\sigma \cdot \hat{\sigma}_k, \quad n_\sigma = 3.5,\ W = 15$$

The constant $1.4826$ makes MAD a consistent estimator of $\sigma$ for Gaussian data.

### 4.2 CUSUM Drift Detector

Sensor baseline drift is detected using a two-sided CUSUM (Cumulative Sum) control chart. Starting from reference mean $\mu_0 = 1.0$ (clean air) and estimated $\sigma$:

$$z_k = \frac{\rho_k - \mu_0}{\sigma}$$

$$S^+_k = \max(0,\; S^+_{k-1} + z_k - k)$$
$$S^-_k = \max(0,\; S^-_{k-1} - z_k - k)$$

A drift event is declared when $\max(S^+_k, S^-_k) > H$, with $k = 0.5$ (allowance) and $H = 5$ (decision threshold). This corresponds to detecting a $1\sigma$ mean shift within approximately 10 samples.

---

## 5. Project Structure

```
e-nose/
├── firmware/
│   ├── platformio.ini              # PlatformIO ESP32 build config
│   └── src/
│       ├── sensor_driver.h         # MOS sensor array driver (interface)
│       ├── sensor_driver.cpp       # Driver implementation
│       ├── main.cpp                # FreeRTOS tasks, WiFi, MQTT, OTA
│       └── ml/                     # Generated by train.py (gitignored)
│           ├── feature_extract.h
│           ├── rf_model.h
│           └── svr_model.h
├── ml_pipeline/
│   └── train.py                    # Full ML pipeline + C export
├── gateway/
│   ├── app.py                      # FastAPI + MQTT client + reconciliation
│   └── dashboard.py                # Streamlit monitoring dashboard
├── mosquitto/
│   └── config/
│       └── mosquitto.conf          # Broker configuration
├── Dockerfile                      # Multi-stage production image
├── docker-compose.yml              # Full stack orchestration
├── requirements.txt                # Python dependencies
├── .gitignore
└── README.md
```

---

## 6. Firmware Build & Flash

### Prerequisites

```bash
pip install platformio
```

### First-time setup

Edit WiFi/MQTT credentials in `firmware/src/main.cpp`:

```cpp
static const char* WIFI_SSID     = "YOUR_SSID";
static const char* WIFI_PASSWORD  = "YOUR_PASSWORD";
static const char* MQTT_BROKER   = "192.168.1.100";  // Gateway host IP
```

### Build

```bash
cd firmware
pio run
```

### Flash

```bash
pio run --target upload --upload-port /dev/ttyUSB0
```

### Monitor serial output

```bash
pio device monitor --baud 115200
```

Expected output after boot:

```
========================================
  e-Nose Industrial Firmware v1.0.0
========================================
[WiFi] Connecting to MyNetwork....... Connected — IP: 192.168.1.42
[MQTT] Connecting... OK
[Sensor] Baseline acquisition started (ch0..ch3)
[Sensor] ch0 baseline locked: R_air = 43200.1 Ω
...
```

### OTA Update

Send the firmware binary URL via MQTT:

```bash
mosquitto_pub -h 192.168.1.100 -u enose -P enose_secret \
  -t enose/cmd/ota_url -m "http://192.168.1.100:8080/firmware_v1_1_0.bin"
```

---

## 7. Gateway Deployment

### Generate MQTT password

```bash
docker run --rm eclipse-mosquitto \
  mosquitto_passwd -c -b /tmp/passwd enose enose_secret && \
  cat /tmp/passwd > mosquitto/config/passwd
```

### Start the full stack

```bash
docker-compose up -d
```

Services:
| Service | Port | URL |
|---------|------|-----|
| MQTT broker | 1883 | `mqtt://localhost:1883` |
| Gateway API | 8000 | http://localhost:8000/docs |
| Dashboard | 8501 | http://localhost:8501 |

### Run ML training (one-shot)

```bash
docker-compose --profile training up trainer
```

This trains on 10,000 synthetic windows and writes the C headers directly into `firmware/src/ml/`. Rebuild and re-flash firmware afterwards.

### Check logs

```bash
docker-compose logs -f gateway
docker-compose logs -f mosquitto
```

### Stop

```bash
docker-compose down
```

---

## 8. API Reference

### `GET /api/v1/latest`

Returns the most recent reconciled sensor snapshot.

```json
{
  "snapshot_id": 4821,
  "timestamp_utc": "2025-06-13T14:22:01.123456+00:00",
  "device_id": "enose-001",
  "predicted_gas": "ethanol",
  "predicted_gas_id": 1,
  "concentration_ppm": 127.4,
  "model_confidence": 0.87,
  "all_baselines_locked": true,
  "anomaly_flags": [],
  "channels": [
    {
      "ch": 0,
      "r_ratio": 0.4821,
      "resistance_ohm": 20900.5,
      "drift_status": "ok",
      "health_score": 0.98
    }
  ]
}
```

### `GET /api/v1/history?n=300`

Returns last `n` snapshots (max 3000).

### `GET /api/v1/health`

Returns gateway uptime stats and per-channel health scores.

### `POST /api/v1/recalibrate`

Publishes a recalibration command to the ESP32 device via MQTT.

---

## 9. ML Training

```bash
cd ml_pipeline
pip install numpy scikit-learn joblib
python train.py --n-samples 10000 --output-dir ../firmware/src/ml
```

**Output:**
```
[1/5] Generating 10000 synthetic windows...
[2/5] Extracting features...
      X shape: (10000, 32)
[3/5] Training gas classifier...
      CV F1-macro: 0.9712 ± 0.0041
      OOB accuracy: 0.9734
[4/5] Training concentration regressors...
      ethanol : MAE=    8.21 ppm  R²=0.9841
      ammonia : MAE=    4.13 ppm  R²=0.9762
      co      : MAE=    3.87 ppm  R²=0.9803
      acetone : MAE=   12.44 ppm  R²=0.9711
[5/5] Exporting models to C...
      Random Forest → firmware/src/ml/rf_model.h  (100 trees)
      SVR regressors → firmware/src/ml/svr_model.h  (4 models)
```

### Adapting to real sensor data

Replace the synthetic generator with your own calibration dataset:

```python
# train.py — replace generate_dataset() call with:
windows = load_real_dataset("path/to/calibration_recordings/")
```

Each real `SensorWindow` requires:
- `r_ratio_sequence`: shape `(60, 4)` float32 array
- `gas_label`: int 0–4
- `concentration_ppm`: float

---

## 10. Contributing

Issues and PRs welcome. Please follow:
- C++17, `clang-format` style (Google profile)
- Python `black` + `ruff` linting
- All new features require unit tests
- Sensor driver changes must include measured SNR data

---

*Built with ❤️ for industrial IoT R&D*
