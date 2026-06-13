#pragma once

/**
 * @file sensor_driver.h
 * @brief Industrial e-Nose MOS Gas Sensor Array Driver
 *
 * Manages a 4-element Metal Oxide Semiconductor (MOS) gas sensor matrix.
 * Each sensor channel implements:
 *   - 12-bit ADC acquisition with x16 oversampling
 *   - Exponential Moving Average (EMA) digital filter for thermal noise rejection
 *   - Scalar Kalman filter for state estimation
 *   - PWM-controlled heater line for detection/regeneration duty cycling
 *
 * Board target: ESP32 (240 MHz dual-core, 12-bit SAR-ADC)
 * Protocol: readings serialized as JSON over UART/MQTT
 *
 * Architecture:
 *   ADC raw → Oversampling accumulator → EMA filter → Kalman filter → R_gas/R_air ratio
 *
 * @author  e-Nose R&D Team
 * @version 1.0.0
 */

#include <cstdint>
#include <array>
#include <functional>

// ---------------------------------------------------------------------------
// Compile-time configuration
// ---------------------------------------------------------------------------

static constexpr uint8_t  SENSOR_COUNT          = 4;
static constexpr uint16_t ADC_RESOLUTION_BITS   = 12;
static constexpr uint32_t ADC_MAX_VALUE         = (1u << ADC_RESOLUTION_BITS) - 1; // 4095
static constexpr uint16_t OVERSAMPLING_FACTOR   = 16;
static constexpr float    SUPPLY_VOLTAGE        = 3.3f;  // V
static constexpr float    LOAD_RESISTANCE_OHM   = 10000.0f; // 10 kΩ load resistor (standard for MQ-series)

// PWM heater parameters (ESP32 LEDC peripheral)
static constexpr uint32_t PWM_FREQ_HZ           = 1000;
static constexpr uint8_t  PWM_RESOLUTION_BITS   = 8;    // 0–255 duty
static constexpr uint8_t  HEATER_FULL_DUTY      = 255;
static constexpr uint8_t  HEATER_IDLE_DUTY      = 80;   // ~31% — keeps sensor warm without full burn

// Heater duty-cycle timing (ms)
static constexpr uint32_t DETECTION_PHASE_MS    = 60000;   // 60 s detection window
static constexpr uint32_t REGENERATION_PHASE_MS = 30000;   // 30 s regeneration (high heat burns off residues)

// Baseline calibration
static constexpr uint32_t BASELINE_SAMPLES      = 200;     // samples averaged for R_air

// EMA filter coefficient α ∈ (0,1]: α=1 → no filter; α≈0.05 → heavy smoothing
static constexpr float    EMA_ALPHA             = 0.08f;

// Kalman filter initial tuning
static constexpr float    KALMAN_Q              = 1e-5f;   // Process noise covariance
static constexpr float    KALMAN_R              = 1e-3f;   // Measurement noise covariance

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/**
 * @brief Per-sensor 1D scalar Kalman filter state.
 *
 * State equation:  x_k = x_{k-1} + w_k,         w_k ~ N(0, Q)
 * Observation:     z_k = x_k + v_k,              v_k ~ N(0, R)
 * Predict:         x̂_k|k-1 = x̂_{k-1|k-1}
 *                  P_k|k-1  = P_{k-1|k-1} + Q
 * Update:          K_k      = P_k|k-1 / (P_k|k-1 + R)
 *                  x̂_k|k   = x̂_k|k-1 + K_k * (z_k - x̂_k|k-1)
 *                  P_k|k    = (1 - K_k) * P_k|k-1
 */
struct KalmanState {
    float x_est;    ///< Current state estimate (filtered ADC value)
    float p_cov;    ///< Estimate error covariance
    float q_noise;  ///< Process noise covariance Q
    float r_noise;  ///< Measurement noise covariance R
};

/**
 * @brief Phase of the heater duty cycle.
 */
enum class HeaterPhase : uint8_t {
    DETECTION     = 0,  ///< Normal operating temperature
    REGENERATION  = 1,  ///< Elevated temperature to desorb contamination
};

/**
 * @brief Full state snapshot for one sensor channel.
 */
struct SensorReading {
    uint8_t  channel_id;        ///< 0–3
    uint32_t timestamp_ms;      ///< System uptime in milliseconds
    float    raw_adc_avg;       ///< Oversampled 12-bit ADC mean (0–4095)
    float    ema_filtered;      ///< EMA output
    float    kalman_filtered;   ///< Kalman filter output
    float    resistance_ohm;    ///< Derived sensor resistance R_gas (Ω)
    float    r_ratio;           ///< R_gas / R_air (dimensionless) — feature for ML
    float    baseline_r_air;    ///< Baseline resistance in clean air (Ω)
    HeaterPhase heater_phase;   ///< Current heater duty-cycle phase
    uint8_t  heater_duty;       ///< PWM duty 0–255
    bool     baseline_locked;   ///< True once BASELINE_SAMPLES have been collected
};

/**
 * @brief Aggregated snapshot of all 4 sensor channels.
 */
struct SensorArraySnapshot {
    std::array<SensorReading, SENSOR_COUNT> sensors;
    uint32_t snapshot_id;       ///< Monotonic counter
    bool     all_baselines_locked;
};

// ---------------------------------------------------------------------------
// Platform abstraction callbacks (injected at construction)
// ---------------------------------------------------------------------------

/**
 * @brief Reads the raw 12-bit ADC value for a given channel.
 * @param channel Channel index 0–3.
 * @return Raw ADC value in [0, 4095].
 */
using AdcReadFn = std::function<uint16_t(uint8_t channel)>;

/**
 * @brief Sets PWM duty on the heater channel.
 * @param channel Channel index 0–3.
 * @param duty    Duty cycle 0–255.
 */
using HeaterSetFn = std::function<void(uint8_t channel, uint8_t duty)>;

/**
 * @brief Returns current system uptime in milliseconds.
 */
using UptimeFn = std::function<uint32_t()>;

// ---------------------------------------------------------------------------
// SensorDriver class
// ---------------------------------------------------------------------------

class SensorDriver {
public:
    /**
     * @brief Construct the sensor driver with platform callbacks.
     * @param adc_fn     Platform ADC read function.
     * @param heater_fn  Platform heater PWM set function.
     * @param uptime_fn  Platform uptime function.
     */
    SensorDriver(AdcReadFn adc_fn, HeaterSetFn heater_fn, UptimeFn uptime_fn);

    /**
     * @brief Initialize all channels: reset Kalman states, start heaters,
     *        begin baseline acquisition.
     */
    void begin();

    /**
     * @brief Acquire one full snapshot from all channels.
     *        Must be called periodically (e.g., every 200 ms from main loop).
     * @return Fully populated SensorArraySnapshot.
     */
    SensorArraySnapshot acquire();

    /**
     * @brief Force re-calibration of baseline R_air for all channels.
     *        Blocks until BASELINE_SAMPLES are collected on each channel.
     */
    void recalibrate_baseline();

    /**
     * @brief Serialize latest snapshot to a compact JSON string.
     * @param snap The snapshot to serialize.
     * @param buf  Output buffer.
     * @param len  Buffer size.
     * @return Number of bytes written (excluding null terminator).
     */
    static int serialize_json(const SensorArraySnapshot& snap, char* buf, size_t len);

private:
    // ---- Platform callbacks ----
    AdcReadFn   m_adc_read;
    HeaterSetFn m_heater_set;
    UptimeFn    m_uptime;

    // ---- Per-channel state ----
    std::array<KalmanState, SENSOR_COUNT>  m_kalman;
    std::array<float, SENSOR_COUNT>        m_ema_state;
    std::array<float, SENSOR_COUNT>        m_baseline_accumulator;
    std::array<uint32_t, SENSOR_COUNT>     m_baseline_sample_count;
    std::array<float, SENSOR_COUNT>        m_baseline_r_air;
    std::array<bool, SENSOR_COUNT>         m_baseline_locked;

    // ---- Heater duty cycle ----
    std::array<HeaterPhase, SENSOR_COUNT>  m_heater_phase;
    std::array<uint32_t, SENSOR_COUNT>     m_heater_phase_start_ms;

    // ---- Snapshot counter ----
    uint32_t m_snapshot_id;

    // ---- Internal methods ----

    /**
     * @brief Perform x16 oversampled ADC read on one channel.
     * @param ch Channel index.
     * @return Averaged 12-bit value (float for precision).
     */
    float oversampled_read(uint8_t ch) const;

    /**
     * @brief Apply EMA filter: s_k = α·z_k + (1-α)·s_{k-1}
     * @param ch  Channel index.
     * @param raw New measurement.
     * @return Filtered value.
     */
    float apply_ema(uint8_t ch, float raw);

    /**
     * @brief Apply scalar Kalman filter update step.
     * @param ch          Channel index.
     * @param measurement New EMA-filtered measurement.
     * @return Kalman posterior estimate.
     */
    float apply_kalman(uint8_t ch, float measurement);

    /**
     * @brief Convert ADC counts to sensor resistance (Ω).
     *
     * Using voltage divider model:
     *   V_out = V_cc * R_L / (R_sensor + R_L)
     *   R_sensor = R_L * (ADC_MAX / adc_val - 1)
     *
     * @param adc_val Filtered ADC value.
     * @return Sensor resistance in Ω.
     */
    static float adc_to_resistance(float adc_val);

    /**
     * @brief Update heater duty cycle state machine for one channel.
     * @param ch         Channel index.
     * @param now_ms     Current uptime in ms.
     */
    void update_heater_fsm(uint8_t ch, uint32_t now_ms);

    /**
     * @brief Reset Kalman state for a channel to a given initial value.
     * @param ch    Channel index.
     * @param x0    Initial state estimate.
     */
    void reset_kalman(uint8_t ch, float x0);
};
