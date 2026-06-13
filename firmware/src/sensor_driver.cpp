/**
 * @file sensor_driver.cpp
 * @brief Implementation of the Industrial e-Nose MOS Sensor Array Driver
 *
 * See sensor_driver.h for mathematical specifications of all filters.
 */

#include "sensor_driver.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <numeric>

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

SensorDriver::SensorDriver(AdcReadFn adc_fn, HeaterSetFn heater_fn, UptimeFn uptime_fn)
    : m_adc_read(std::move(adc_fn))
    , m_heater_set(std::move(heater_fn))
    , m_uptime(std::move(uptime_fn))
    , m_snapshot_id(0u)
{
    // Zero-initialize all per-channel arrays
    m_kalman.fill({0.0f, 1.0f, KALMAN_Q, KALMAN_R});
    m_ema_state.fill(0.0f);
    m_baseline_accumulator.fill(0.0f);
    m_baseline_sample_count.fill(0u);
    m_baseline_r_air.fill(10000.0f);  // Safe default: 10 kΩ
    m_baseline_locked.fill(false);
    m_heater_phase.fill(HeaterPhase::DETECTION);
    m_heater_phase_start_ms.fill(0u);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void SensorDriver::begin()
{
    const uint32_t now_ms = m_uptime();

    for (uint8_t ch = 0; ch < SENSOR_COUNT; ++ch) {
        // Stagger heater start times by 5 s to avoid inrush current peaks
        m_heater_phase[ch]           = HeaterPhase::DETECTION;
        m_heater_phase_start_ms[ch]  = now_ms + static_cast<uint32_t>(ch) * 5000u;
        m_heater_set(ch, HEATER_FULL_DUTY);

        // Warm start: acquire one sample to seed EMA and Kalman
        const float seed = oversampled_read(ch);
        m_ema_state[ch] = seed;
        reset_kalman(ch, seed);
    }
}

SensorArraySnapshot SensorDriver::acquire()
{
    const uint32_t now_ms = m_uptime();
    SensorArraySnapshot snap{};
    snap.snapshot_id       = ++m_snapshot_id;
    snap.all_baselines_locked = true;

    for (uint8_t ch = 0; ch < SENSOR_COUNT; ++ch) {
        // 1. Update heater state machine
        update_heater_fsm(ch, now_ms);

        // 2. Oversampled raw read
        const float raw_avg = oversampled_read(ch);

        // 3. EMA filter
        const float ema_val = apply_ema(ch, raw_avg);

        // 4. Kalman filter
        const float kalman_val = apply_kalman(ch, ema_val);

        // 5. Convert to resistance
        const float r_gas = adc_to_resistance(kalman_val);

        // 6. Baseline accumulation (only during DETECTION and while heater is stable)
        if (!m_baseline_locked[ch] && m_heater_phase[ch] == HeaterPhase::DETECTION) {
            m_baseline_accumulator[ch]  += r_gas;
            m_baseline_sample_count[ch] += 1u;

            if (m_baseline_sample_count[ch] >= BASELINE_SAMPLES) {
                m_baseline_r_air[ch]  = m_baseline_accumulator[ch]
                                      / static_cast<float>(BASELINE_SAMPLES);
                m_baseline_locked[ch] = true;
            }
        }

        // 7. Compute R_gas / R_air ratio
        const float r_air   = m_baseline_r_air[ch];
        const float r_ratio = (r_air > 0.0f) ? (r_gas / r_air) : 1.0f;

        // 8. Populate reading struct
        SensorReading& rd         = snap.sensors[ch];
        rd.channel_id             = ch;
        rd.timestamp_ms           = now_ms;
        rd.raw_adc_avg            = raw_avg;
        rd.ema_filtered           = ema_val;
        rd.kalman_filtered        = kalman_val;
        rd.resistance_ohm         = r_gas;
        rd.r_ratio                = r_ratio;
        rd.baseline_r_air         = r_air;
        rd.heater_phase           = m_heater_phase[ch];
        rd.heater_duty            = (m_heater_phase[ch] == HeaterPhase::DETECTION)
                                      ? HEATER_FULL_DUTY : HEATER_IDLE_DUTY;
        rd.baseline_locked        = m_baseline_locked[ch];

        if (!m_baseline_locked[ch]) {
            snap.all_baselines_locked = false;
        }
    }

    return snap;
}

void SensorDriver::recalibrate_baseline()
{
    for (uint8_t ch = 0; ch < SENSOR_COUNT; ++ch) {
        m_baseline_accumulator[ch]  = 0.0f;
        m_baseline_sample_count[ch] = 0u;
        m_baseline_locked[ch]       = false;
    }

    // Block until all baselines are re-acquired
    // NOTE: In a FreeRTOS context this would use vTaskDelay; here we spin.
    while (true) {
        bool all_done = true;
        for (uint8_t ch = 0; ch < SENSOR_COUNT; ++ch) {
            const float raw_avg    = oversampled_read(ch);
            const float ema_val    = apply_ema(ch, raw_avg);
            const float kalman_val = apply_kalman(ch, ema_val);
            const float r_gas      = adc_to_resistance(kalman_val);

            if (!m_baseline_locked[ch]) {
                m_baseline_accumulator[ch]  += r_gas;
                m_baseline_sample_count[ch] += 1u;

                if (m_baseline_sample_count[ch] >= BASELINE_SAMPLES) {
                    m_baseline_r_air[ch]  = m_baseline_accumulator[ch]
                                          / static_cast<float>(BASELINE_SAMPLES);
                    m_baseline_locked[ch] = true;
                } else {
                    all_done = false;
                }
            }
        }
        if (all_done) break;
    }
}

int SensorDriver::serialize_json(const SensorArraySnapshot& snap, char* buf, size_t len)
{
    int written = 0;
    written += snprintf(buf + written, len - static_cast<size_t>(written),
        "{\"snapshot_id\":%lu,\"all_baselines\":%s,\"sensors\":[",
        static_cast<unsigned long>(snap.snapshot_id),
        snap.all_baselines_locked ? "true" : "false");

    for (uint8_t ch = 0; ch < SENSOR_COUNT; ++ch) {
        const SensorReading& rd = snap.sensors[ch];
        const char* phase_str   = (rd.heater_phase == HeaterPhase::DETECTION)
                                  ? "detection" : "regeneration";

        written += snprintf(buf + written, len - static_cast<size_t>(written),
            "{\"ch\":%u,\"ts\":%lu,\"raw\":%.2f,\"ema\":%.2f,"
            "\"kalman\":%.2f,\"r_gas\":%.1f,\"r_ratio\":%.4f,"
            "\"r_air\":%.1f,\"phase\":\"%s\",\"duty\":%u,\"baseline\":%s}",
            rd.channel_id,
            static_cast<unsigned long>(rd.timestamp_ms),
            rd.raw_adc_avg,
            rd.ema_filtered,
            rd.kalman_filtered,
            rd.resistance_ohm,
            rd.r_ratio,
            rd.baseline_r_air,
            phase_str,
            rd.heater_duty,
            rd.baseline_locked ? "true" : "false");

        if (ch < SENSOR_COUNT - 1) {
            written += snprintf(buf + written, len - static_cast<size_t>(written), ",");
        }
    }

    written += snprintf(buf + written, len - static_cast<size_t>(written), "]}");
    return written;
}

// ---------------------------------------------------------------------------
// Private methods
// ---------------------------------------------------------------------------

float SensorDriver::oversampled_read(uint8_t ch) const
{
    uint32_t accumulator = 0u;
    for (uint16_t i = 0; i < OVERSAMPLING_FACTOR; ++i) {
        accumulator += m_adc_read(ch);
    }
    // Effective resolution gain: log2(OVERSAMPLING_FACTOR)/2 = 2 bits → 14-bit equivalent
    return static_cast<float>(accumulator) / static_cast<float>(OVERSAMPLING_FACTOR);
}

float SensorDriver::apply_ema(uint8_t ch, float raw)
{
    // s_k = α·z_k + (1-α)·s_{k-1}
    m_ema_state[ch] = EMA_ALPHA * raw + (1.0f - EMA_ALPHA) * m_ema_state[ch];
    return m_ema_state[ch];
}

float SensorDriver::apply_kalman(uint8_t ch, float measurement)
{
    KalmanState& ks = m_kalman[ch];

    // --- Predict step ---
    // State prediction (random walk model): x̂_k|k-1 = x̂_{k-1|k-1}
    // The state does not change in prediction; only covariance grows:
    const float p_predict = ks.p_cov + ks.q_noise;

    // --- Update step ---
    // Kalman gain: K = P_predict / (P_predict + R)
    const float kalman_gain = p_predict / (p_predict + ks.r_noise);

    // Posterior state estimate
    ks.x_est = ks.x_est + kalman_gain * (measurement - ks.x_est);

    // Posterior covariance: Joseph form for numerical stability
    // P_k|k = (1 - K) * P_k|k-1
    ks.p_cov = (1.0f - kalman_gain) * p_predict;

    return ks.x_est;
}

float SensorDriver::adc_to_resistance(float adc_val)
{
    // Guard: avoid division by zero or negative values
    if (adc_val < 1.0f) adc_val = 1.0f;

    // Voltage divider: V_out = V_cc * R_L / (R_sensor + R_L)
    // R_sensor = R_L * (V_cc / V_out - 1)
    //          = R_L * (ADC_MAX / adc_val - 1)
    const float ratio = static_cast<float>(ADC_MAX_VALUE) / adc_val;
    return LOAD_RESISTANCE_OHM * (ratio - 1.0f);
}

void SensorDriver::update_heater_fsm(uint8_t ch, uint32_t now_ms)
{
    const uint32_t elapsed_ms = now_ms - m_heater_phase_start_ms[ch];

    switch (m_heater_phase[ch]) {
    case HeaterPhase::DETECTION:
        if (elapsed_ms >= DETECTION_PHASE_MS) {
            m_heater_phase[ch]          = HeaterPhase::REGENERATION;
            m_heater_phase_start_ms[ch] = now_ms;
            m_heater_set(ch, HEATER_IDLE_DUTY);
        }
        break;

    case HeaterPhase::REGENERATION:
        if (elapsed_ms >= REGENERATION_PHASE_MS) {
            m_heater_phase[ch]          = HeaterPhase::DETECTION;
            m_heater_phase_start_ms[ch] = now_ms;
            m_heater_set(ch, HEATER_FULL_DUTY);
        }
        break;
    }
}

void SensorDriver::reset_kalman(uint8_t ch, float x0)
{
    m_kalman[ch] = KalmanState{x0, 1.0f, KALMAN_Q, KALMAN_R};
}
