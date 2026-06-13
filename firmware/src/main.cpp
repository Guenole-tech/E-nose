/**
 * @file main.cpp
 * @brief e-Nose Firmware Entry Point — ESP32 (Arduino framework via PlatformIO)
 *
 * Responsibilities:
 *   1. Initialize hardware peripherals (ADC, PWM, WiFi, MQTT)
 *   2. Instantiate SensorDriver with platform callbacks
 *   3. Run a 200 ms acquisition loop on Core 1 (FreeRTOS)
 *   4. Publish JSON snapshots to MQTT broker on Core 0
 *   5. Handle OTA firmware updates via ESP-IDF OTA partition
 *
 * MQTT Topics:
 *   Publish:   enose/sensors/raw        — full JSON snapshot every 200 ms
 *              enose/sensors/status     — heartbeat every 10 s
 *              enose/alerts             — drift/anomaly alerts
 *   Subscribe: enose/cmd/recalibrate   — triggers baseline recalibration
 *              enose/cmd/ota_url        — triggers OTA from URL
 *
 * PlatformIO environment (platformio.ini):
 *   [env:esp32dev]
 *   platform  = espressif32
 *   board     = esp32dev
 *   framework = arduino
 *   lib_deps  =
 *     knolleary/PubSubClient @ ^2.8
 *     bblanchon/ArduinoJson  @ ^7.0
 *
 * @note All blocking operations in the MQTT task use timeouts to prevent
 *       watchdog resets (ESP32 TWDT default 5 s).
 */

#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <esp_task_wdt.h>
#include <Update.h>

#include "sensor_driver.h"

// ---------------------------------------------------------------------------
// User configuration — replace with your values or pull from NVS/SPIFFS
// ---------------------------------------------------------------------------

static const char* WIFI_SSID       = "YOUR_SSID";
static const char* WIFI_PASSWORD   = "YOUR_PASSWORD";
static const char* MQTT_BROKER     = "192.168.1.100";  // Gateway IP
static const uint16_t MQTT_PORT    = 1883;
static const char* MQTT_CLIENT_ID  = "enose-001";
static const char* MQTT_USER       = "enose";
static const char* MQTT_PASS       = "enose_secret";

// ---------------------------------------------------------------------------
// GPIO mapping (ESP32 DevKit V1 — adjust for your PCB)
// ---------------------------------------------------------------------------

// ADC channels — ESP32 ADC1 pins (ADC2 unavailable with WiFi active)
static const uint8_t ADC_PINS[SENSOR_COUNT] = {36, 39, 34, 35}; // GPIO36–35 (ADC1_CH0–CH7)

// Heater PWM channels (LEDC)
static const uint8_t HEATER_PINS[SENSOR_COUNT]    = {25, 26, 27, 14};
static const uint8_t LEDC_CHANNELS[SENSOR_COUNT]  = {0, 1, 2, 3};

// ---------------------------------------------------------------------------
// MQTT topics
// ---------------------------------------------------------------------------

static const char* TOPIC_SENSORS    = "enose/sensors/raw";
static const char* TOPIC_STATUS     = "enose/sensors/status";
static const char* TOPIC_ALERTS     = "enose/alerts";
static const char* TOPIC_RECAL      = "enose/cmd/recalibrate";
static const char* TOPIC_OTA        = "enose/cmd/ota_url";

// ---------------------------------------------------------------------------
// Task parameters
// ---------------------------------------------------------------------------

static const uint32_t ACQUISITION_INTERVAL_MS  = 200u;
static const uint32_t MQTT_PUBLISH_INTERVAL_MS = 200u;
static const uint32_t HEARTBEAT_INTERVAL_MS    = 10000u;
static const uint32_t WDT_TIMEOUT_S            = 10u;

// ---------------------------------------------------------------------------
// Shared state (protected by mutex)
// ---------------------------------------------------------------------------

static SemaphoreHandle_t        g_snapshot_mutex   = nullptr;
static SensorArraySnapshot      g_latest_snapshot  = {};
static volatile bool            g_snapshot_ready   = false;
static volatile bool            g_recalibrate_flag = false;
static char                     g_ota_url[256]     = {0};
static volatile bool            g_ota_pending      = false;

// ---------------------------------------------------------------------------
// Platform objects
// ---------------------------------------------------------------------------

static WiFiClient   g_wifi_client;
static PubSubClient g_mqtt(g_wifi_client);
static SensorDriver* g_sensor_driver = nullptr;

// ---------------------------------------------------------------------------
// Platform callback implementations
// ---------------------------------------------------------------------------

static uint16_t platform_adc_read(uint8_t channel)
{
    // ESP32 analogRead returns 12-bit [0, 4095]
    return static_cast<uint16_t>(analogRead(ADC_PINS[channel]));
}

static void platform_heater_set(uint8_t channel, uint8_t duty)
{
    ledcWrite(LEDC_CHANNELS[channel], duty);
}

static uint32_t platform_uptime_ms()
{
    return static_cast<uint32_t>(millis());
}

// ---------------------------------------------------------------------------
// WiFi helpers
// ---------------------------------------------------------------------------

static void wifi_connect()
{
    Serial.printf("[WiFi] Connecting to %s", WIFI_SSID);
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    uint32_t timeout = millis() + 30000u;
    while (WiFi.status() != WL_CONNECTED && millis() < timeout) {
        delay(500);
        Serial.print(".");
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("\n[WiFi] Connected — IP: %s\n", WiFi.localIP().toString().c_str());
    } else {
        Serial.println("\n[WiFi] FAILED — rebooting in 5 s");
        delay(5000);
        ESP.restart();
    }
}

// ---------------------------------------------------------------------------
// MQTT helpers
// ---------------------------------------------------------------------------

static void mqtt_callback(char* topic, uint8_t* payload, unsigned int length)
{
    // Null-terminate payload safely
    char msg[256] = {0};
    const size_t copy_len = std::min(static_cast<size_t>(length), sizeof(msg) - 1u);
    memcpy(msg, payload, copy_len);

    Serial.printf("[MQTT] Message on '%s': %s\n", topic, msg);

    if (strcmp(topic, TOPIC_RECAL) == 0) {
        g_recalibrate_flag = true;
        Serial.println("[CMD] Baseline recalibration scheduled.");
        return;
    }

    if (strcmp(topic, TOPIC_OTA) == 0) {
        strncpy(g_ota_url, msg, sizeof(g_ota_url) - 1);
        g_ota_pending = true;
        Serial.printf("[CMD] OTA update scheduled from: %s\n", g_ota_url);
        return;
    }
}

static bool mqtt_connect()
{
    uint8_t retries = 0;
    while (!g_mqtt.connected() && retries < 5) {
        Serial.print("[MQTT] Connecting...");
        if (g_mqtt.connect(MQTT_CLIENT_ID, MQTT_USER, MQTT_PASS,
                           TOPIC_STATUS, 1, true, "offline")) {
            Serial.println(" OK");
            g_mqtt.subscribe(TOPIC_RECAL);
            g_mqtt.subscribe(TOPIC_OTA);
            g_mqtt.publish(TOPIC_STATUS, "online", true);
            return true;
        }
        Serial.printf(" failed (rc=%d), retry %d/5\n", g_mqtt.state(), retries + 1);
        delay(2000 << retries);  // Exponential back-off
        ++retries;
    }
    return g_mqtt.connected();
}

// ---------------------------------------------------------------------------
// OTA update handler
// ---------------------------------------------------------------------------

static void perform_ota_update(const char* url)
{
    Serial.printf("[OTA] Starting update from %s\n", url);

    // Using HTTPClient + Update (ESP32 Arduino)
    // In production: add TLS certificate validation
    HTTPClient http;
    http.begin(url);
    int http_code = http.GET();

    if (http_code != HTTP_CODE_OK) {
        Serial.printf("[OTA] HTTP error: %d\n", http_code);
        http.end();
        return;
    }

    const int content_length = http.getSize();
    WiFiClient* stream = http.getStreamPtr();

    if (!Update.begin(content_length)) {
        Serial.printf("[OTA] Not enough space: %s\n", Update.errorString());
        http.end();
        return;
    }

    const size_t written = Update.writeStream(*stream);
    http.end();

    if (!Update.end() || !Update.isFinished()) {
        Serial.printf("[OTA] Update error: %s\n", Update.errorString());
        return;
    }

    Serial.printf("[OTA] Written %u bytes. Rebooting...\n", written);
    delay(1000);
    ESP.restart();
}

// ---------------------------------------------------------------------------
// FreeRTOS Tasks
// ---------------------------------------------------------------------------

/**
 * @brief Acquisition task (Core 1, highest priority)
 *
 * Runs at 5 Hz (200 ms period). Acquires sensor snapshot, stores in shared
 * buffer protected by mutex.
 */
static void task_acquire(void* /*param*/)
{
    esp_task_wdt_add(nullptr);
    TickType_t  last_wake     = xTaskGetTickCount();
    const TickType_t period   = pdMS_TO_TICKS(ACQUISITION_INTERVAL_MS);

    for (;;) {
        esp_task_wdt_reset();

        if (g_recalibrate_flag) {
            g_recalibrate_flag = false;
            Serial.println("[Sensor] Starting recalibration...");
            g_sensor_driver->recalibrate_baseline();
            Serial.println("[Sensor] Recalibration complete.");
        }

        SensorArraySnapshot snap = g_sensor_driver->acquire();

        if (xSemaphoreTake(g_snapshot_mutex, pdMS_TO_TICKS(50)) == pdTRUE) {
            g_latest_snapshot = snap;
            g_snapshot_ready  = true;
            xSemaphoreGive(g_snapshot_mutex);
        }

        vTaskDelayUntil(&last_wake, period);
    }
}

/**
 * @brief Communication task (Core 0)
 *
 * Handles WiFi reconnection, MQTT keep-alive, and publishes snapshots.
 */
static void task_comms(void* /*param*/)
{
    esp_task_wdt_add(nullptr);

    wifi_connect();

    g_mqtt.setServer(MQTT_BROKER, MQTT_PORT);
    g_mqtt.setCallback(mqtt_callback);
    g_mqtt.setBufferSize(2048);
    mqtt_connect();

    uint32_t last_heartbeat = 0u;
    char json_buf[1536];

    for (;;) {
        esp_task_wdt_reset();

        // Reconnect WiFi if needed
        if (WiFi.status() != WL_CONNECTED) {
            Serial.println("[WiFi] Lost connection — reconnecting...");
            wifi_connect();
        }

        // Reconnect MQTT if needed
        if (!g_mqtt.connected()) {
            mqtt_connect();
        }

        g_mqtt.loop();

        // Publish sensor snapshot
        bool do_publish = false;
        SensorArraySnapshot snap_copy{};

        if (xSemaphoreTake(g_snapshot_mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
            if (g_snapshot_ready) {
                snap_copy      = g_latest_snapshot;
                g_snapshot_ready = false;
                do_publish       = true;
            }
            xSemaphoreGive(g_snapshot_mutex);
        }

        if (do_publish && g_mqtt.connected()) {
            const int len = SensorDriver::serialize_json(snap_copy, json_buf, sizeof(json_buf));
            if (len > 0) {
                g_mqtt.publish(TOPIC_SENSORS, reinterpret_cast<uint8_t*>(json_buf),
                               static_cast<unsigned int>(len), false);
            }
        }

        // Heartbeat every 10 s
        const uint32_t now = millis();
        if (now - last_heartbeat >= HEARTBEAT_INTERVAL_MS) {
            last_heartbeat = now;
            StaticJsonDocument<256> hb;
            hb["device"]   = MQTT_CLIENT_ID;
            hb["uptime_s"] = now / 1000u;
            hb["heap_free"] = esp_get_free_heap_size();
            hb["rssi"]     = WiFi.RSSI();
            char hb_buf[256];
            serializeJson(hb, hb_buf, sizeof(hb_buf));
            g_mqtt.publish(TOPIC_STATUS, hb_buf, true);
        }

        // OTA update (blocks task intentionally)
        if (g_ota_pending) {
            g_ota_pending = false;
            g_mqtt.publish(TOPIC_ALERTS, "{\"type\":\"ota_start\"}");
            g_mqtt.disconnect();
            perform_ota_update(g_ota_url);
        }

        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

// ---------------------------------------------------------------------------
// Arduino entry points
// ---------------------------------------------------------------------------

void setup()
{
    Serial.begin(115200);
    delay(500);
    Serial.println("\n========================================");
    Serial.println("  e-Nose Industrial Firmware v1.0.0");
    Serial.println("========================================\n");

    // Configure ADC: 12-bit, 11 dB attenuation (0–3.9 V range)
    analogReadResolution(12);
    analogSetAttenuation(ADC_11db);
    for (uint8_t ch = 0; ch < SENSOR_COUNT; ++ch) {
        pinMode(ADC_PINS[ch], INPUT);
    }

    // Configure LEDC PWM for heaters
    for (uint8_t ch = 0; ch < SENSOR_COUNT; ++ch) {
        ledcSetup(LEDC_CHANNELS[ch], PWM_FREQ_HZ, PWM_RESOLUTION_BITS);
        ledcAttachPin(HEATER_PINS[ch], LEDC_CHANNELS[ch]);
        ledcWrite(LEDC_CHANNELS[ch], 0);  // Heaters off until begin()
    }

    // Create sensor driver with platform callbacks
    g_sensor_driver = new SensorDriver(
        platform_adc_read,
        platform_heater_set,
        platform_uptime_ms
    );
    g_sensor_driver->begin();

    // Create shared mutex
    g_snapshot_mutex = xSemaphoreCreateMutex();
    configASSERT(g_snapshot_mutex != nullptr);

    // Configure watchdog
    esp_task_wdt_init(WDT_TIMEOUT_S, true);

    // Launch tasks
    // Acquisition: pinned to Core 1, priority 5 (time-critical)
    xTaskCreatePinnedToCore(task_acquire, "acquire", 4096, nullptr, 5, nullptr, 1);
    // Comms: pinned to Core 0, priority 3
    xTaskCreatePinnedToCore(task_comms,   "comms",   8192, nullptr, 3, nullptr, 0);

    Serial.println("[Main] All tasks launched. Loop idle.");
}

void loop()
{
    // Main loop intentionally idle — all work done in FreeRTOS tasks
    vTaskDelay(pdMS_TO_TICKS(1000));
}
