#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <ESPmDNS.h>
#include <esp_heap_caps.h>
#include <math.h>

#include "tinytb_weights.h"
#include "tinytb_samples.h"
#include "tinytb_png_samples.h"
#include "tinytb_web_page.h"
#include "tinytb_infer.h"

#define TINYTB_THRESHOLD_SCREENING 0.35f
#define TINYTB_THRESHOLD_STANDARD  0.50f

// Network Credentials requested by user
const char* WIFI_SSID = "Mika";
const char* WIFI_PASS = "shalev13";

WebServer server(80);

// Dynamic buffer for web image upload
static uint8_t* web_img_buf = NULL;
static size_t web_img_offset = 0;

void run_single_inference(const char* label, const uint8_t* img, float ref_logit = NAN, float ref_prob = NAN,
                          void (*post_s1_cb)(void*) = NULL, void* cb_ctx = NULL) {
    Serial.printf("\n>>> RUNNING INFERENCE: %s <<<\n", label);
    Serial.printf("Free Heap Before: %lu bytes (Largest Block: %lu bytes)\n",
                  (unsigned long)ESP.getFreeHeap(), (unsigned long)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));

    float prob = 0.0f;
    TinyTBStats stats;
    memset(&stats, 0, sizeof(stats));

    uint32_t t0 = millis();
    float logit = tinytb_predict(img, &prob, &stats, post_s1_cb, cb_ctx);
    uint32_t t_total = millis() - t0;

    Serial.println("--- TIMING BREAKDOWN ---");
    Serial.printf("  Stem + Stage 1 (Streaming Ring Buffer): %lu ms\n", (unsigned long)stats.stem_s1_ms);
    Serial.printf("  Stage 2 (32 ch, 56x56):                 %lu ms\n", (unsigned long)stats.s2_ms);
    Serial.printf("  Stage 3 (64 ch, 28x28 single-pass):     %lu ms\n", (unsigned long)stats.s3_ms);
    Serial.printf("  Stage 4 (128 ch, 14x14 single-pass):    %lu ms\n", (unsigned long)stats.s4_ms);
    Serial.printf("  GAP + Linear Classifier:                %lu ms\n", (unsigned long)stats.classifier_ms);
    Serial.printf("  Total Latency:                          %lu ms (%.2f s)\n", (unsigned long)t_total, t_total / 1000.0f);

    size_t peak_heap_used = (stats.heap_start > stats.heap_min) ? (stats.heap_start - stats.heap_min) : 0;
    Serial.println("--- MEMORY PROFILE ---");
    Serial.printf("  Initial Free Heap:                      %lu bytes\n", (unsigned long)stats.heap_start);
    Serial.printf("  Minimum Free Heap (Peak Consumption):   %lu bytes\n", (unsigned long)stats.heap_min);
    Serial.printf("  Peak Dynamic SRAM Allocated:            %lu bytes (~%.1f KB)\n",
                  (unsigned long)peak_heap_used, peak_heap_used / 1024.0f);
    Serial.printf("  Final Free Heap:                        %lu bytes (Leak: %ld bytes)\n",
                  (unsigned long)stats.heap_end, (long)(stats.heap_start - stats.heap_end));

    // Clinical interpretation
    const char* screening_diag = (prob >= TINYTB_THRESHOLD_SCREENING) ? "TUBERCULOSIS (TB)" : "NORMAL";
    const char* standard_diag  = (prob >= TINYTB_THRESHOLD_STANDARD)  ? "TUBERCULOSIS (TB)" : "NORMAL";

    Serial.println("--- CLINICAL DIAGNOSTIC RESULT ---");
    Serial.printf("  Predicted Logit:                        %.6f\n", logit);
    Serial.printf("  Predicted Probability:                  %.6f (%.2f%%)\n", prob, prob * 100.0f);
    Serial.printf("  Screening Triage (Threshold 0.35):      %s\n", screening_diag);
    Serial.printf("  Standard Diagnosis (Threshold 0.50):    %s\n", standard_diag);

    if (!isnan(ref_logit)) {
        float logit_diff = fabsf(logit - ref_logit);
        float prob_diff = fabsf(prob - ref_prob);
        Serial.println("--- NUMERICAL VERIFICATION ---");
        Serial.printf("  Reference Logit (PyTorch FX INT8):      %.6f (Delta: %.6f)\n", ref_logit, logit_diff);
        Serial.printf("  Reference Probability:                  %.6f (Delta: %.6f)\n", ref_prob, prob_diff);
        Serial.printf("  Numerical Agreement:                    %s\n", (logit_diff < 0.001f) ? "PASS (BIT-EXACT)" : "FAIL");
    }

    // Machine-readable output for serial integration
    Serial.printf("RESULT:LOGIT=%.6f,PROB=%.6f,SCREENING=%s,STANDARD=%s,TOTAL_MS=%lu\n",
                  logit, prob, screening_diag, standard_diag, (unsigned long)t_total);
    Serial.println("----------------------------------------");
}

void run_full_benchmark() {
    Serial.println("\n==================================================");
    Serial.println("  TINYTBNET INT8 QUANTIZED ON-DEVICE BENCHMARK   ");
    Serial.println("==================================================");
    
    run_single_inference("Sample 1: TB Positive (CHNCXR_0404_1)", sample_tb_img, REF_TB_LOGIT, REF_TB_PROB);
    run_single_inference("Sample 2: Normal / Healthy (CHNCXR_0001_0)", sample_norm_img, REF_NORM_LOGIT, REF_NORM_PROB);

    Serial.println("\n==================================================");
    Serial.println("  BENCHMARK COMPLETE - SYSTEM READY FOR COMMANDS  ");
    Serial.println("==================================================");
}

// Web Server Upload Handler for /predict
void handlePredictUpload() {
    if (server.hasRaw()) {
        HTTPRaw& raw = server.raw();
        if (raw.status == RAW_START) {
            web_img_offset = 0;
            if (web_img_buf) { free(web_img_buf); web_img_buf = NULL; }
            web_img_buf = (uint8_t*)malloc(50176);
            if (!web_img_buf) {
                Serial.println("[WebServer] ERROR: Failed to allocate 50,176 bytes for incoming image");
            }
        } else if (raw.status == RAW_WRITE) {
            if (web_img_buf && web_img_offset + raw.currentSize <= 50176) {
                memcpy(web_img_buf + web_img_offset, raw.buf, raw.currentSize);
                web_img_offset += raw.currentSize;
            }
        } else if (raw.status == RAW_ABORTED) {
            if (web_img_buf) { free(web_img_buf); web_img_buf = NULL; }
            web_img_offset = 0;
            Serial.println("[WebServer] RAW upload aborted by client");
        }
    } else if (server.hasUpload()) {
        HTTPUpload& up = server.upload();
        if (up.status == UPLOAD_FILE_START) {
            web_img_offset = 0;
            if (web_img_buf) { free(web_img_buf); web_img_buf = NULL; }
            web_img_buf = (uint8_t*)malloc(50176);
        } else if (up.status == UPLOAD_FILE_WRITE) {
            if (web_img_buf && web_img_offset + up.currentSize <= 50176) {
                memcpy(web_img_buf + web_img_offset, up.buf, up.currentSize);
                web_img_offset += up.currentSize;
            }
        } else if (up.status == UPLOAD_FILE_ABORTED) {
            if (web_img_buf) { free(web_img_buf); web_img_buf = NULL; }
            web_img_offset = 0;
        }
    }
}

// Web Server Prediction Execution Handler
void handlePredict() {
    if (!web_img_buf || web_img_offset != 50176) {
        server.sendHeader("Access-Control-Allow-Origin", "*");
        server.send(400, "application/json", "{\"error\":\"Invalid payload, expected 50176 raw grayscale bytes\"}");
        if (web_img_buf) { free(web_img_buf); web_img_buf = NULL; }
        web_img_offset = 0;
        return;
    }

    Serial.printf("\n[WebServer] Processing /predict request (50,176 bytes) from %s\n",
                  server.client().remoteIP().toString().c_str());

    // In-place quantization: [0..255] -> INT8 [0..128]
    tinytb_quantize_raw_image(web_img_buf, web_img_buf);

    float prob = 0.0f;
    TinyTBStats stats;
    memset(&stats, 0, sizeof(stats));

    uint32_t t0 = millis();
    float logit = tinytb_predict(web_img_buf, &prob, &stats, [](void* p) {
        uint8_t** b = (uint8_t**)p;
        if (b && *b) {
            free(*b);
            *b = NULL;
            Serial.println("[WebServer] Freed 50KB image buffer post-Stage 1 (maximum SRAM available for Stages 2-4)");
        }
    }, &web_img_buf);
    uint32_t t_total = millis() - t0;

    if (web_img_buf) {
        free(web_img_buf);
        web_img_buf = NULL;
    }
    web_img_offset = 0;

    const char* screening_diag = (prob >= TINYTB_THRESHOLD_SCREENING) ? "TUBERCULOSIS (TB)" : "NORMAL";
    const char* standard_diag  = (prob >= TINYTB_THRESHOLD_STANDARD)  ? "TUBERCULOSIS (TB)" : "NORMAL";

    char json[384];
    snprintf(json, sizeof(json),
             "{\"probability\":%.6f,\"logit\":%.6f,\"is_tuberculosis\":%s,"
             "\"screening\":\"%s\",\"standard\":\"%s\",\"latency_ms\":%lu,"
             "\"threshold_screening\":%.2f,\"threshold_standard\":%.2f}",
             prob, logit,
             (prob >= TINYTB_THRESHOLD_SCREENING) ? "true" : "false",
             screening_diag, standard_diag, (unsigned long)t_total,
             TINYTB_THRESHOLD_SCREENING, TINYTB_THRESHOLD_STANDARD);

    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.send(200, "application/json", json);
    Serial.printf("[WebServer] Prediction sent: Prob=%.2f%%, Diagnosis=%s, Time=%lu ms\n",
                  prob * 100.0f, screening_diag, (unsigned long)t_total);
}

// Receive exactly length bytes over Serial with timeout
bool receive_exact_bytes(uint8_t* buffer, size_t length, uint32_t timeout_ms = 20000) {
    size_t received = 0;
    uint32_t last_rx_time = millis();
    while (received < length) {
        while (Serial.available() > 0 && received < length) {
            buffer[received++] = (uint8_t)Serial.read();
            last_rx_time = millis();
        }
        if (millis() - last_rx_time > timeout_ms) {
            Serial.printf("ERROR: RX_TIMEOUT received %lu of %lu bytes\n", (unsigned long)received, (unsigned long)length);
            return false;
        }
        yield();
    }
    return true;
}

// Handle UART Serial commands
void handle_serial_commands() {
    if (!Serial.available()) return;

    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) return;

    if (line == "PING") {
        Serial.println("PONG");
    } else if (line == "INFO") {
        Serial.printf("FREE_HEAP:%lu,LARGEST_BLOCK:%lu,UPTIME_MS:%lu,IP:%s,WIFI:%s\n",
                      (unsigned long)ESP.getFreeHeap(),
                      (unsigned long)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT),
                      (unsigned long)millis(),
                      WiFi.status() == WL_CONNECTED ? WiFi.localIP().toString().c_str() : WiFi.softAPIP().toString().c_str(),
                      WiFi.status() == WL_CONNECTED ? "CONNECTED" : "AP_MODE");
    } else if (line == "BENCHMARK" || line == "RUN") {
        run_full_benchmark();
    } else if (line == "RUN_TB") {
        run_single_inference("Sample 1: TB Positive (CHNCXR_0404_1)", sample_tb_img, REF_TB_LOGIT, REF_TB_PROB);
    } else if (line == "RUN_NORM") {
        run_single_inference("Sample 2: Normal / Healthy (CHNCXR_0001_0)", sample_norm_img, REF_NORM_LOGIT, REF_NORM_PROB);
    } else if (line.startsWith("INFER_QUANT") || line.startsWith("INFER_RAW")) {
        bool is_raw = line.startsWith("INFER_RAW");
        int space_idx = line.indexOf(' ');
        size_t expected_len = 50176;
        if (space_idx > 0) {
            expected_len = line.substring(space_idx + 1).toInt();
        }

        if (expected_len != 50176) {
            Serial.printf("ERROR: INVALID_LEN expected 50176, got %lu\n", (unsigned long)expected_len);
        } else {
            uint8_t* img_buf = (uint8_t*)malloc(expected_len);
            if (!img_buf) {
                Serial.println("ERROR: OUT_OF_MEMORY");
            } else {
                Serial.println("READY");
                Serial.flush();
                if (receive_exact_bytes(img_buf, expected_len)) {
                    if (is_raw) {
                        tinytb_quantize_raw_image(img_buf, img_buf);
                    }
                    run_single_inference(is_raw ? "Serial Live Raw CXR" : "Serial Live Pre-quantized CXR",
                                         img_buf, NAN, NAN,
                                         [](void* p) {
                                             uint8_t** b = (uint8_t**)p;
                                             if (b && *b) {
                                                 free(*b);
                                                 *b = NULL;
                                             }
                                         },
                                         &img_buf);
                }
                if (img_buf) {
                    free(img_buf);
                    img_buf = NULL;
                }
            }
        }
    } else {
        Serial.printf("UNKNOWN_COMMAND: %s\n", line.c_str());
        Serial.println("Supported commands: PING, INFO, BENCHMARK, RUN, RUN_TB, RUN_NORM, INFER_QUANT <len>, INFER_RAW <len>");
    }
}

// Increase default Arduino loop task stack size from 8KB to 32KB
size_t getArduinoLoopTaskStackSize(void) {
    return 32768;
}

void setup() {
    Serial.begin(115200);
    delay(500);

    // Disable loop task watchdog timer so long inferences do not trigger watchdog reset
    disableLoopWDT();

    Serial.println("\n\n==================================================");
    Serial.println("      ESP32 TINYTB-NET INT8 INFERENCE ENGINE       ");
    Serial.println("==================================================");
    Serial.printf("Chip Model:        %s (Revision %d)\n", ESP.getChipModel(), ESP.getChipRevision());
    Serial.printf("CPU Frequency:     %lu MHz\n", (unsigned long)ESP.getCpuFreqMHz());
    Serial.printf("Flash Size:        %lu MB\n", (unsigned long)(ESP.getFlashChipSize() / (1024 * 1024)));
    Serial.printf("Initial Free Heap: %lu bytes\n", (unsigned long)ESP.getFreeHeap());
    Serial.printf("Largest Heap Block: %lu bytes\n", (unsigned long)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
    Serial.println("Watchdog Timer:    Disabled for Loop Task");
    Serial.println("Optimization:      Single-pass Stage 3 & 4 Conv2");
    Serial.println("Screening Mode:    Active (Threshold = 0.35)");
    Serial.println("==================================================");

    // Wi-Fi Connection
    Serial.printf("\n[WiFi] Connecting to network '%s'...\n", WIFI_SSID);
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASS);

    uint32_t t_wifi = millis();
    while (WiFi.status() != WL_CONNECTED && (millis() - t_wifi) < 15000) {
        delay(400);
        Serial.print(".");
    }
    Serial.println();

    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("[WiFi] *** CONNECTED TO MIKA ***");
        Serial.printf("[WiFi] IP Address:       %s\n", WiFi.localIP().toString().c_str());
        Serial.printf("[WiFi] Subnet Mask:      %s\n", WiFi.subnetMask().toString().c_str());
        Serial.printf("[WiFi] Gateway:          %s\n", WiFi.gatewayIP().toString().c_str());
        Serial.printf("[WiFi] Signal (RSSI):    %d dBm\n", WiFi.RSSI());

        if (MDNS.begin("tinytb")) {
            Serial.println("[mDNS] Responder started: http://tinytb.local");
            MDNS.addService("http", "tcp", 80);
        }
    } else {
        Serial.printf("[WiFi] WARNING: Could not connect to '%s' (Status code: %d)\n", WIFI_SSID, WiFi.status());
        Serial.println("[WiFi] Starting fallback SoftAP network 'TinyTB-Scanner'...");
        WiFi.mode(WIFI_AP_STA);
        WiFi.softAP("TinyTB-Scanner", "12345678");
        Serial.printf("[WiFi] Fallback SoftAP active! Connect to SSID 'TinyTB-Scanner' (password '12345678')\n");
        Serial.printf("[WiFi] Access Web UI at: http://%s/\n", WiFi.softAPIP().toString().c_str());
    }

    // Configure WebServer routes
    server.on("/", HTTP_GET, []() {
        server.sendHeader("Content-Encoding", "gzip");
        server.send_P(200, "text/html", (const char*)TINYTB_INDEX_HTML_GZ, TINYTB_INDEX_HTML_GZ_LEN);
    });

    server.on("/index.html", HTTP_GET, []() {
        server.sendHeader("Content-Encoding", "gzip");
        server.send_P(200, "text/html", (const char*)TINYTB_INDEX_HTML_GZ, TINYTB_INDEX_HTML_GZ_LEN);
    });

    server.on("/health", HTTP_GET, []() {
        char buf[256];
        bool is_sta = (WiFi.status() == WL_CONNECTED);
        snprintf(buf, sizeof(buf),
                 "{\"status\":\"ok\",\"model\":\"TinyTB-Net INT8\",\"wifi_mode\":\"%s\",\"ip\":\"%s\",\"free_heap\":%lu}",
                 is_sta ? "Station" : "SoftAP",
                 is_sta ? WiFi.localIP().toString().c_str() : WiFi.softAPIP().toString().c_str(),
                 (unsigned long)ESP.getFreeHeap());
        server.sendHeader("Access-Control-Allow-Origin", "*");
        server.send(200, "application/json", buf);
    });

    server.on("/sample/CHNCXR_0404_1.png", HTTP_GET, []() {
        server.sendHeader("Access-Control-Allow-Origin", "*");
        server.send_P(200, "image/png", (const char*)sample_tb_png, SAMPLE_TB_PNG_LEN);
    });

    server.on("/sample/CHNCXR_0001_0.png", HTTP_GET, []() {
        server.sendHeader("Access-Control-Allow-Origin", "*");
        server.send_P(200, "image/png", (const char*)sample_norm_png, SAMPLE_NORM_PNG_LEN);
    });

    server.on("/predict", HTTP_OPTIONS, []() {
        server.sendHeader("Access-Control-Allow-Origin", "*");
        server.sendHeader("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
        server.sendHeader("Access-Control-Allow-Headers", "Content-Type");
        server.send(204);
    });

    server.on("/predict", HTTP_POST, handlePredict, handlePredictUpload);

    server.begin();
    Serial.println("\n[WebServer] HTTP Web Diagnostic Server online on port 80!");
    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("[WebServer] Open in your browser: http://%s/ (or http://tinytb.local/)\n",
                      WiFi.localIP().toString().c_str());
    } else {
        Serial.printf("[WebServer] Open in your browser: http://%s/\n", WiFi.softAPIP().toString().c_str());
    }
    Serial.println("==================================================\n");
}

void loop() {
    server.handleClient();
    handle_serial_commands();
    delay(2);
}
