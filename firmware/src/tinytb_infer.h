#ifndef TINYTB_INFER_H
#define TINYTB_INFER_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(ARDUINO) || defined(ESP32)
#include <Arduino.h>
#include <esp_heap_caps.h>
#include <esp_task_wdt.h>
#include <esp_partition.h>
static inline void feed_wdt_if_enabled() {
#if CONFIG_AUTOSTART_ARDUINO
    extern bool loopTaskWDTEnabled;
    if (loopTaskWDTEnabled) {
        feedLoopWDT();
    }
#endif
}

#define S2_SC_FLASH_OFFSET   0
#define S2_C1_FLASH_OFFSET   102400
#define S2_OUT_FLASH_OFFSET  204800
#define S3_SC_FLASH_OFFSET   307200
#define FLASH_ERASE_TOTAL    360448

static inline const esp_partition_t* get_tinytb_partition() {
    static const esp_partition_t* part = NULL;
    if (!part) {
        part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_DATA_SPIFFS, NULL);
        if (!part) {
            part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, NULL);
        }
    }
    return part;
}

static inline void tinytb_flash_erase(const esp_partition_t* part, size_t offset, size_t size) {
    if (part) esp_partition_erase_range(part, offset, size);
}

static inline void tinytb_flash_write(const esp_partition_t* part, size_t offset, const void* data, size_t size) {
    if (part) esp_partition_write(part, offset, data, size);
}

static inline void tinytb_flash_read(const esp_partition_t* part, size_t offset, void* data, size_t size) {
    if (part) esp_partition_read(part, offset, data, size);
}

#else
static inline void yield() {}
static inline void feed_wdt_if_enabled() {}

static uint8_t host_flash_buf[400000];
#define S2_SC_FLASH_OFFSET   0
#define S2_C1_FLASH_OFFSET   102400
#define S2_OUT_FLASH_OFFSET  204800
#define S3_SC_FLASH_OFFSET   307200
#define FLASH_ERASE_TOTAL    360448
typedef void esp_partition_t;
static inline const esp_partition_t* get_tinytb_partition() { return (const esp_partition_t*)host_flash_buf; }
static inline void tinytb_flash_erase(const esp_partition_t* part, size_t offset, size_t size) { memset(host_flash_buf + offset, 0xFF, size); }
static inline void tinytb_flash_write(const esp_partition_t* part, size_t offset, const void* data, size_t size) { memcpy(host_flash_buf + offset, data, size); }
static inline void tinytb_flash_read(const esp_partition_t* part, size_t offset, void* data, size_t size) { memcpy(data, host_flash_buf + offset, size); }
#endif


#include "tinytb_weights.h"

// Telemetry & profiling structure
struct TinyTBStats {
    uint32_t stem_s1_ms;
    uint32_t s2_ms;
    uint32_t s3_ms;
    uint32_t s4_ms;
    uint32_t classifier_ms;
    uint32_t total_ms;
    size_t heap_start;
    size_t heap_min;
    size_t heap_end;
};

// Fast clamping
static inline uint8_t clamp_u8(int32_t val) {
    if (val < 0) return 0;
    if (val > 255) return 255;
    return (uint8_t)val;
}

// 256-byte LUT mapping raw grayscale pixels [0..255] to INT8 quantized input [0..128]
static const uint8_t raw_to_quant_lut[256] = {
      0,   0,   1,   1,   2,   2,   3,   3,   4,   4,   5,   5,   6,   6,   7,   7,
      8,   8,   9,   9,  10,  10,  11,  11,  12,  12,  13,  13,  14,  14,  15,  15,
     16,  16,  17,  17,  18,  18,  19,  19,  20,  20,  21,  21,  22,  22,  23,  23,
     24,  24,  25,  25,  26,  26,  27,  27,  28,  28,  29,  29,  30,  30,  31,  31,
     32,  32,  33,  33,  34,  35,  35,  36,  36,  37,  37,  38,  38,  39,  39,  40,
     40,  41,  41,  42,  42,  43,  43,  44,  44,  45,  45,  46,  46,  47,  47,  48,
     48,  49,  49,  50,  50,  51,  51,  52,  52,  53,  53,  54,  54,  55,  55,  56,
     56,  57,  57,  58,  58,  59,  59,  60,  60,  61,  61,  62,  62,  63,  63,  64,
     64,  65,  65,  66,  66,  67,  67,  68,  68,  69,  69,  70,  70,  71,  71,  72,
     72,  73,  73,  74,  74,  75,  75,  76,  76,  77,  77,  78,  78,  79,  79,  80,
     80,  81,  81,  82,  82,  83,  83,  84,  84,  85,  85,  86,  86,  87,  87,  88,
     88,  89,  89,  90,  90,  91,  91,  92,  92,  93,  93,  94,  95,  95,  96,  96,
     97,  97,  98,  98,  99,  99, 100, 100, 101, 101, 102, 102, 103, 103, 104, 104,
    105, 105, 106, 106, 107, 107, 108, 108, 109, 109, 110, 110, 111, 111, 112, 112,
    113, 113, 114, 114, 115, 115, 116, 116, 117, 117, 118, 118, 119, 119, 120, 120,
    121, 121, 122, 122, 123, 123, 124, 124, 125, 125, 126, 126, 127, 127, 128, 128
};

static inline void tinytb_quantize_raw_image(const uint8_t* raw_img, uint8_t* quant_out) {
    for (int i = 0; i < 224 * 224; i++) {
        quant_out[i] = raw_to_quant_lut[raw_img[i]];
    }
}

// Channel allocation helpers
static uint8_t** alloc_channels(int num_channels, int plane_size) {
    uint8_t** ptrs = (uint8_t**)malloc(num_channels * sizeof(uint8_t*));
    if (!ptrs) {
#if defined(ARDUINO) || defined(ESP32)
        Serial.printf("[DEBUG] alloc_channels: ptrs array failed (free=%u, max=%u)\n", ESP.getFreeHeap(), ESP.getMaxAllocHeap());
#endif
        return NULL;
    }
    for (int c = 0; c < num_channels; c++) {
        ptrs[c] = (uint8_t*)malloc(plane_size);
        if (!ptrs[c]) {
#if defined(ARDUINO) || defined(ESP32)
            Serial.printf("[DEBUG] alloc_channels: failed at channel %d/%d for size %d (free=%u, max=%u)\n",
                          c, num_channels, plane_size, ESP.getFreeHeap(), ESP.getMaxAllocHeap());
#endif
            for (int k = 0; k < c; k++) free(ptrs[k]);
            free(ptrs);
            return NULL;
        }
    }
    return ptrs;
}

static void free_channels(uint8_t** ptrs, int num_channels) {
    if (!ptrs) return;
    for (int c = 0; c < num_channels; c++) {
        if (ptrs[c]) free(ptrs[c]);
    }
    free(ptrs);
}

// Helper to get current free heap
static inline size_t get_current_free_heap() {
#if defined(ARDUINO) || defined(ESP32)
    return (size_t)ESP.getFreeHeap();
#else
    return 300000;
#endif
}

// Helper to get millis
static inline uint32_t get_current_time_ms() {
#if defined(ARDUINO) || defined(ESP32)
    return (uint32_t)millis();
#else
    return 0;
#endif
}

// Compute one row of Stem (16 channels x 112 width)
static void compute_stem_row(const uint8_t* img, int r, uint8_t* stem_row) {
    int ih_base = r * 2 - 1;
    for (int oc = 0; oc < 16; oc++) {
        const int8_t* w_oc = &stem_conv_weight[oc * 9];
        float w_sc = stem_conv_w_scale[oc];
        float b = stem_conv_bias[oc];

        for (int c = 0; c < 112; c++) {
            int iw_base = c * 2 - 1;
            int32_t acc = 0;

            for (int kh = 0; kh < 3; kh++) {
                int ih = ih_base + kh;
                if (ih >= 0 && ih < 224) {
                    const uint8_t* img_row = &img[ih * 224];
                    for (int kw = 0; kw < 3; kw++) {
                        int iw = iw_base + kw;
                        if (iw >= 0 && iw < 224) {
                            int32_t x = (int32_t)img_row[iw] - INPUT_ZERO_POINT;
                            int32_t w = (int32_t)w_oc[kh * 3 + kw];
                            acc += x * w;
                        }
                    }
                }
            }

            float r_val = (float)acc * (INPUT_SCALE * w_sc) + b;
            if (r_val < 0.0f) r_val = 0.0f; // ReLU
            int32_t q = (int32_t)roundf(r_val / STEM_CONV_OUT_SCALE) + STEM_CONV_OUT_ZERO_POINT;
            stem_row[oc * 112 + c] = clamp_u8(q);
        }
    }
}

// 3x3 Conv row from 3 input row buffers: in_rows[3] each [in_ch * width]
static void conv3x3_row(const uint8_t* in_rows[3], int in_ch, int width,
                        int out_ch, const int8_t* weights, const float* w_scales, const float* biases,
                        float in_sc, int in_zp, float out_sc, int out_zp, bool has_relu,
                        uint8_t* out_row) {
    for (int oc = 0; oc < out_ch; oc++) {
        const int8_t* w_oc = &weights[oc * in_ch * 9];
        float w_sc = w_scales[oc];
        float b = biases[oc];

        for (int c = 0; c < width; c++) {
            int iw_base = c - 1;
            int32_t acc = 0;

            for (int ic = 0; ic < in_ch; ic++) {
                const int8_t* w_k = &w_oc[ic * 9];
                for (int kh = 0; kh < 3; kh++) {
                    const uint8_t* in_p = in_rows[kh];
                    if (in_p != NULL) {
                        const uint8_t* in_ic = &in_p[ic * width];
                        for (int kw = 0; kw < 3; kw++) {
                            int iw = iw_base + kw;
                            if (iw >= 0 && iw < width) {
                                int32_t x = (int32_t)in_ic[iw] - in_zp;
                                acc += x * (int32_t)w_k[kh * 3 + kw];
                            }
                        }
                    }
                }
            }

            float r_val = (float)acc * (in_sc * w_sc) + b;
            if (has_relu && r_val < 0.0f) r_val = 0.0f;
            int32_t q = (int32_t)roundf(r_val / out_sc) + out_zp;
            out_row[oc * width + c] = clamp_u8(q);
        }
    }
}

// Full streaming inference engine
static inline float tinytb_predict(const uint8_t* img, float* out_prob, TinyTBStats* stats = NULL,
                                   void (*post_s1_cb)(void*) = NULL, void* cb_ctx = NULL) {
    uint32_t t_start = get_current_time_ms();
    size_t h_start = get_current_free_heap();
    size_t h_min = h_start;

    const esp_partition_t* sc_part = get_tinytb_partition();
    tinytb_flash_erase(sc_part, 0, FLASH_ERASE_TOTAL);

    // Dynamic buffers for Stage 1 streaming (allocated on heap to protect task stack)
    uint8_t (*stem_buf)[16 * 112] = (uint8_t(*)[16 * 112])malloc(3 * 16 * 112);
    uint8_t (*c1_buf)[16 * 112] = (uint8_t(*)[16 * 112])malloc(3 * 16 * 112);
    uint8_t* c2_row = (uint8_t*)malloc(16 * 112);
    uint8_t (*s1_out_buf)[16 * 112] = (uint8_t(*)[16 * 112])malloc(3 * 16 * 112);
    if (!stem_buf || !c1_buf || !c2_row || !s1_out_buf) {
        if (stem_buf) free(stem_buf);
        if (c1_buf) free(c1_buf);
        if (c2_row) free(c2_row);
        if (s1_out_buf) free(s1_out_buf);
        return -999.0f;
    }


    // ==========================================
    // STAGE 1 - PASS 1: Accumulate SE GAP sums
    // ==========================================
#if defined(ARDUINO) || defined(ESP32)
    Serial.print("  [Pass 1/2: Stem+S1 GAP] ");
#endif
    int64_t s1_c2_sums[16] = {0};

    // Pre-fill stem rows 0 and 1
    compute_stem_row(img, 0, stem_buf[0]);
    compute_stem_row(img, 1, stem_buf[1]);

    // Pre-fill c1 row 0 (needs stem rows -1 (NULL), 0, 1)
    {
        const uint8_t* in_0[3] = {NULL, stem_buf[0], stem_buf[1]};
        conv3x3_row(in_0, 16, 112, 16, s1_c1_weight, s1_c1_w_scale, s1_c1_bias,
                    STEM_CONV_OUT_SCALE, STEM_CONV_OUT_ZERO_POINT, S1_C1_OUT_SCALE, S1_C1_OUT_ZERO_POINT, true,
                    c1_buf[0]);
    }

    for (int r = 0; r < 112; r++) {
#if defined(ARDUINO) || defined(ESP32)
        feed_wdt_if_enabled();
        if (r % 16 == 0) Serial.print(".");
        if (r % 4 == 0) yield();
#endif

        // Need c1 row r+1 (if r+1 < 112)
        if (r + 1 < 112) {
            if (r + 2 < 112) {
                compute_stem_row(img, r + 2, stem_buf[(r + 2) % 3]);
            }
            const uint8_t* in_c1[3] = {
                stem_buf[r % 3],
                stem_buf[(r + 1) % 3],
                (r + 2 < 112) ? stem_buf[(r + 2) % 3] : NULL
            };
            conv3x3_row(in_c1, 16, 112, 16, s1_c1_weight, s1_c1_w_scale, s1_c1_bias,
                        STEM_CONV_OUT_SCALE, STEM_CONV_OUT_ZERO_POINT, S1_C1_OUT_SCALE, S1_C1_OUT_ZERO_POINT, true,
                        c1_buf[(r + 1) % 3]);
        }

        // Compute c2 row r (needs c1 rows r-1, r, r+1)
        const uint8_t* in_c2[3] = {
            (r - 1 >= 0) ? c1_buf[(r - 1) % 3] : NULL,
            c1_buf[r % 3],
            (r + 1 < 112) ? c1_buf[(r + 1) % 3] : NULL
        };
        conv3x3_row(in_c2, 16, 112, 16, s1_c2_weight, s1_c2_w_scale, s1_c2_bias,
                    S1_C1_OUT_SCALE, S1_C1_OUT_ZERO_POINT, S1_C2_OUT_SCALE, S1_C2_OUT_ZERO_POINT, false,
                    c2_row);

        for (int oc = 0; oc < 16; oc++) {
            const uint8_t* p = &c2_row[oc * 112];
            int64_t row_sum = 0;
            for (int c = 0; c < 112; c++) {
                row_sum += p[c];
            }
            s1_c2_sums[oc] += row_sum;
        }
    }

    // Compute Stage 1 SE Sigmoid
    uint8_t s1_se_gap[16];
    for (int oc = 0; oc < 16; oc++) {
        s1_se_gap[oc] = (uint8_t)roundf((float)s1_c2_sums[oc] / (112.0f * 112.0f));
    }

    uint8_t s1_se1_out[4];
    for (int oc = 0; oc < 4; oc++) {
        int32_t acc = 0;
        const int8_t* w_oc = &s1_se1_weight[oc * 16];
        for (int ic = 0; ic < 16; ic++) {
            acc += ((int32_t)s1_se_gap[ic] - S1_C2_OUT_ZERO_POINT) * (int32_t)w_oc[ic];
        }
        float r_val = (float)acc * (S1_C2_OUT_SCALE * s1_se1_w_scale[oc]) + s1_se1_bias[oc];
        if (r_val < 0.0f) r_val = 0.0f;
        int32_t q = (int32_t)roundf(r_val / S1_SE1_OUT_SCALE) + S1_SE1_OUT_ZERO_POINT;
        s1_se1_out[oc] = clamp_u8(q);
    }

    float s1_sig[16];
    for (int oc = 0; oc < 16; oc++) {
        int32_t acc = 0;
        const int8_t* w_oc = &s1_se2_weight[oc * 4];
        for (int ic = 0; ic < 4; ic++) {
            acc += ((int32_t)s1_se1_out[ic] - S1_SE1_OUT_ZERO_POINT) * (int32_t)w_oc[ic];
        }
        float r_val = (float)acc * (S1_SE1_OUT_SCALE * s1_se2_w_scale[oc]) + s1_se2_bias[oc];
        int32_t q = (int32_t)roundf(r_val / S1_SE2_OUT_SCALE) + S1_SE2_OUT_ZERO_POINT;
        q = clamp_u8(q);
        float r_deq = ((float)q - S1_SE2_OUT_ZERO_POINT) * S1_SE2_OUT_SCALE;
        float sig_f = 1.0f / (1.0f + expf(-r_deq));
        int32_t q_sig = (int32_t)roundf(sig_f * 256.0f);
        q_sig = clamp_u8(q_sig);
        s1_sig[oc] = (float)q_sig / 256.0f;
    }

    // ==========================================
    // STAGE 1 - PASS 2 -> STREAM DIRECTLY INTO STAGE 2
    // ==========================================
#if defined(ARDUINO) || defined(ESP32)
    Serial.println(" done.");
    Serial.printf("  [Pass 2/2: S1->S2 Stream] FreeHeap=%u, MaxAlloc=%u\n", ESP.getFreeHeap(), ESP.getMaxAllocHeap());
#endif
    uint8_t* sc_row = (uint8_t*)malloc(32 * 56);
    uint8_t* c1_row = (uint8_t*)malloc(32 * 56);
    if (!sc_row || !c1_row) {
        free(stem_buf);
        free(c1_buf);
        free(c2_row);
        free(s1_out_buf);
        if (sc_row) free(sc_row);
        if (c1_row) free(c1_row);
        return -999.0f;
    }

    size_t cur_h = get_current_free_heap();
    if (cur_h < h_min) h_min = cur_h;

    // Reset stem rows for Pass 2
    compute_stem_row(img, 0, stem_buf[0]);
    compute_stem_row(img, 1, stem_buf[1]);

    // Pre-fill c1 row 0
    {
        const uint8_t* in_0[3] = {NULL, stem_buf[0], stem_buf[1]};
        conv3x3_row(in_0, 16, 112, 16, s1_c1_weight, s1_c1_w_scale, s1_c1_bias,
                    STEM_CONV_OUT_SCALE, STEM_CONV_OUT_ZERO_POINT, S1_C1_OUT_SCALE, S1_C1_OUT_ZERO_POINT, true,
                    c1_buf[0]);
    }

    for (int r = 0; r < 112; r++) {
#if defined(ARDUINO) || defined(ESP32)
        feed_wdt_if_enabled();
        if (r % 16 == 0) Serial.print(".");
        if (r % 4 == 0) yield();
#endif

        // Need c1 row r+1 (if r+1 < 112)
        if (r + 1 < 112) {
            if (r + 2 < 112) {
                compute_stem_row(img, r + 2, stem_buf[(r + 2) % 3]);
            }
            const uint8_t* in_c1[3] = {
                stem_buf[r % 3],
                stem_buf[(r + 1) % 3],
                (r + 2 < 112) ? stem_buf[(r + 2) % 3] : NULL
            };
            conv3x3_row(in_c1, 16, 112, 16, s1_c1_weight, s1_c1_w_scale, s1_c1_bias,
                        STEM_CONV_OUT_SCALE, STEM_CONV_OUT_ZERO_POINT, S1_C1_OUT_SCALE, S1_C1_OUT_ZERO_POINT, true,
                        c1_buf[(r + 1) % 3]);
        }

        // Compute c2 row r
        const uint8_t* in_c2[3] = {
            (r - 1 >= 0) ? c1_buf[(r - 1) % 3] : NULL,
            c1_buf[r % 3],
            (r + 1 < 112) ? c1_buf[(r + 1) % 3] : NULL
        };
        conv3x3_row(in_c2, 16, 112, 16, s1_c2_weight, s1_c2_w_scale, s1_c2_bias,
                    S1_C1_OUT_SCALE, S1_C1_OUT_ZERO_POINT, S1_C2_OUT_SCALE, S1_C2_OUT_ZERO_POINT, false,
                    c2_row);

        // Compute AddReLU(Mul(c2_row, s1_sig), stem_row) into s1_out_buf[r % 3]
        uint8_t* curr_s1_out = s1_out_buf[r % 3];
        const uint8_t* curr_stem = stem_buf[r % 3];

        for (int oc = 0; oc < 16; oc++) {
            float sig = s1_sig[oc];
            const uint8_t* p_c2 = &c2_row[oc * 112];
            const uint8_t* p_stem = &curr_stem[oc * 112];
            uint8_t* p_out = &curr_s1_out[oc * 112];

            for (int c = 0; c < 112; c++) {
                float r_c2 = ((float)p_c2[c] - S1_C2_OUT_ZERO_POINT) * S1_C2_OUT_SCALE;
                float r_mul = r_c2 * sig;
                int32_t q_mul = (int32_t)roundf(r_mul / S1_SE_SCALE) + S1_SE_ZERO_POINT;
                q_mul = clamp_u8(q_mul);

                float r_se = ((float)q_mul - S1_SE_ZERO_POINT) * S1_SE_SCALE;
                float r_sc = ((float)p_stem[c] - STEM_CONV_OUT_ZERO_POINT) * STEM_CONV_OUT_SCALE;
                float r_sum = r_se + r_sc;
                if (r_sum < 0.0f) r_sum = 0.0f;
                int32_t q_out = (int32_t)roundf(r_sum / S1_RELU_SCALE) + S1_RELU_ZERO_POINT;
                p_out[c] = clamp_u8(q_out);
            }
        }

        // Stream into Stage 2 Shortcut: when r is even (r = 2 * oh), write row to flash
        if (r % 2 == 0) {
            int oh = r / 2;
            for (int oc = 0; oc < 32; oc++) {
                const int8_t* w_oc = &s2_sc_weight[oc * 16];
                float w_sc = s2_sc_w_scale[oc];
                float b = s2_sc_bias[oc];

                for (int ow = 0; ow < 56; ow++) {
                    int32_t acc = 0;
                    for (int ic = 0; ic < 16; ic++) {
                        int32_t x = (int32_t)curr_s1_out[ic * 112 + (2 * ow)] - S1_RELU_ZERO_POINT;
                        acc += x * (int32_t)w_oc[ic];
                    }
                    float r_val = (float)acc * (S1_RELU_SCALE * w_sc) + b;
                    int32_t q = (int32_t)roundf(r_val / S2_SC_OUT_SCALE) + S2_SC_OUT_ZERO_POINT;
                    sc_row[oc * 56 + ow] = clamp_u8(q);
                }
            }
            tinytb_flash_write(sc_part, S2_SC_FLASH_OFFSET + (size_t)oh * (32 * 56), sc_row, 32 * 56);
        }

        // Stream into Stage 2 Conv1: needs input rows 2oh-1, 2oh, 2oh+1
        if (r % 2 == 1) {
            int oh = r / 2; // oh in [0, 55]
            const uint8_t* in_3[3] = {
                (2 * oh - 1 >= 0) ? s1_out_buf[(2 * oh - 1) % 3] : NULL,
                s1_out_buf[(2 * oh) % 3],
                s1_out_buf[(2 * oh + 1) % 3]
            };

            for (int oc = 0; oc < 32; oc++) {
                const int8_t* w_oc = &s2_c1_weight[oc * 16 * 9];
                float w_sc = s2_c1_w_scale[oc];
                float b = s2_c1_bias[oc];

                for (int ow = 0; ow < 56; ow++) {
                    int iw_base = ow * 2 - 1;
                    int32_t acc = 0;

                    for (int ic = 0; ic < 16; ic++) {
                        const int8_t* w_k = &w_oc[ic * 9];
                        for (int kh = 0; kh < 3; kh++) {
                            const uint8_t* in_p = in_3[kh];
                            if (in_p != NULL) {
                                const uint8_t* in_ic = &in_p[ic * 112];
                                for (int kw = 0; kw < 3; kw++) {
                                    int iw = iw_base + kw;
                                    if (iw >= 0 && iw < 112) {
                                        int32_t x = (int32_t)in_ic[iw] - S1_RELU_ZERO_POINT;
                                        acc += x * (int32_t)w_k[kh * 3 + kw];
                                    }
                                }
                            }
                        }
                    }

                    float r_val = (float)acc * (S1_RELU_SCALE * w_sc) + b;
                    if (r_val < 0.0f) r_val = 0.0f;
                    int32_t q = (int32_t)roundf(r_val / S2_C1_OUT_SCALE) + S2_C1_OUT_ZERO_POINT;
                    c1_row[oc * 56 + ow] = clamp_u8(q);
                }
            }
            tinytb_flash_write(sc_part, S2_C1_FLASH_OFFSET + (size_t)oh * (32 * 56), c1_row, 32 * 56);
        }
    }

    uint32_t t_s1 = get_current_time_ms();
    if (stats) stats->stem_s1_ms = t_s1 - t_start;

    free(sc_row);
    free(c1_row);
    free(stem_buf);
    free(c1_buf);
    free(c2_row);
    free(s1_out_buf);

    if (post_s1_cb) {
        post_s1_cb(cb_ctx);
    }

    // ==========================================
    // STAGE 2: conv2 (from s2_c1 flash) + SE + Residual AddReLU -> s2_out flash
    // ==========================================
#if defined(ARDUINO) || defined(ESP32)
    Serial.println(" done.");
    Serial.print("  [Stage 2: 32ch 56x56] ");
#endif
    uint8_t (*s2_c1_ring)[32 * 56] = (uint8_t(*)[32 * 56])malloc(3 * 32 * 56);
    uint8_t* s2_c2_row = (uint8_t*)malloc(32 * 56);
    if (!s2_c1_ring || !s2_c2_row) {
        if (s2_c1_ring) free(s2_c1_ring);
        if (s2_c2_row) free(s2_c2_row);
        return -999.0f;
    }

    auto compute_s2_c2_row = [&](const uint8_t* in_rows[3], uint8_t* out_row) {
        for (int oc = 0; oc < 32; oc++) {
            const int8_t* w_oc = &s2_c2_weight[oc * 32 * 9];
            float w_sc = s2_c2_w_scale[oc];
            float b = s2_c2_bias[oc];

            for (int c = 0; c < 56; c++) {
                int iw_base = c - 1;
                int32_t acc = 0;

                for (int ic = 0; ic < 32; ic++) {
                    const int8_t* w_k = &w_oc[ic * 9];

                    if (in_rows[0] != NULL) {
                        const uint8_t* p = &in_rows[0][ic * 56];
                        for (int kw = 0; kw < 3; kw++) {
                            int iw = iw_base + kw;
                            if (iw >= 0 && iw < 56) acc += ((int32_t)p[iw] - S2_C1_OUT_ZERO_POINT) * (int32_t)w_k[kw];
                        }
                    }
                    if (in_rows[1] != NULL) {
                        const uint8_t* p = &in_rows[1][ic * 56];
                        for (int kw = 0; kw < 3; kw++) {
                            int iw = iw_base + kw;
                            if (iw >= 0 && iw < 56) acc += ((int32_t)p[iw] - S2_C1_OUT_ZERO_POINT) * (int32_t)w_k[3 + kw];
                        }
                    }
                    if (in_rows[2] != NULL) {
                        const uint8_t* p = &in_rows[2][ic * 56];
                        for (int kw = 0; kw < 3; kw++) {
                            int iw = iw_base + kw;
                            if (iw >= 0 && iw < 56) acc += ((int32_t)p[iw] - S2_C1_OUT_ZERO_POINT) * (int32_t)w_k[6 + kw];
                        }
                    }
                }

                float r_val = (float)acc * (S2_C1_OUT_SCALE * w_sc) + b;
                int32_t q = (int32_t)roundf(r_val / S2_C2_OUT_SCALE) + S2_C2_OUT_ZERO_POINT;
                out_row[oc * 56 + c] = clamp_u8(q);
            }
        }
    };

    // Stage 2 Pass 1: Accumulate sums for GAP
    tinytb_flash_read(sc_part, S2_C1_FLASH_OFFSET + 0 * (32 * 56), s2_c1_ring[0], 32 * 56);
    tinytb_flash_read(sc_part, S2_C1_FLASH_OFFSET + 1 * (32 * 56), s2_c1_ring[1], 32 * 56);

    int64_t s2_c2_sums[32] = {0};
    for (int r = 0; r < 56; r++) {
#if defined(ARDUINO) || defined(ESP32)
        feed_wdt_if_enabled();
        if (r % 14 == 0) Serial.print(".");
        if (r % 4 == 0) yield();
#endif
        if (r + 1 < 56) {
            tinytb_flash_read(sc_part, S2_C1_FLASH_OFFSET + (size_t)(r + 1) * (32 * 56), s2_c1_ring[(r + 1) % 3], 32 * 56);
        }
        const uint8_t* in_3[3] = {
            (r - 1 >= 0) ? s2_c1_ring[(r - 1) % 3] : NULL,
            s2_c1_ring[r % 3],
            (r + 1 < 56) ? s2_c1_ring[(r + 1) % 3] : NULL
        };
        compute_s2_c2_row(in_3, s2_c2_row);
        for (int oc = 0; oc < 32; oc++) {
            const uint8_t* p = &s2_c2_row[oc * 56];
            for (int c = 0; c < 56; c++) {
                s2_c2_sums[oc] += p[c];
            }
        }
    }

    uint8_t s2_se_gap[32];
    for (int oc = 0; oc < 32; oc++) {
        s2_se_gap[oc] = (uint8_t)roundf((float)s2_c2_sums[oc] / (56.0f * 56.0f));
    }

    uint8_t s2_se1_out[4];
    for (int oc = 0; oc < 4; oc++) {
        int32_t acc = 0;
        const int8_t* w_oc = &s2_se1_weight[oc * 32];
        for (int ic = 0; ic < 32; ic++) {
            acc += ((int32_t)s2_se_gap[ic] - S2_C2_OUT_ZERO_POINT) * (int32_t)w_oc[ic];
        }
        float r_val = (float)acc * (S2_C2_OUT_SCALE * s2_se1_w_scale[oc]) + s2_se1_bias[oc];
        if (r_val < 0.0f) r_val = 0.0f;
        int32_t q = (int32_t)roundf(r_val / S2_SE1_OUT_SCALE) + S2_SE1_OUT_ZERO_POINT;
        s2_se1_out[oc] = clamp_u8(q);
    }

    float s2_sig[32];
    for (int oc = 0; oc < 32; oc++) {
        int32_t acc = 0;
        const int8_t* w_oc = &s2_se2_weight[oc * 4];
        for (int ic = 0; ic < 4; ic++) {
            acc += ((int32_t)s2_se1_out[ic] - S2_SE1_OUT_ZERO_POINT) * (int32_t)w_oc[ic];
        }
        float r_val = (float)acc * (S2_SE1_OUT_SCALE * s2_se2_w_scale[oc]) + s2_se2_bias[oc];
        int32_t q = (int32_t)roundf(r_val / S2_SE2_OUT_SCALE) + S2_SE2_OUT_ZERO_POINT;
        q = clamp_u8(q);
        float r_deq = ((float)q - S2_SE2_OUT_ZERO_POINT) * S2_SE2_OUT_SCALE;
        float sig_f = 1.0f / (1.0f + expf(-r_deq));
        int32_t q_sig = (int32_t)roundf(sig_f * 256.0f);
        q_sig = clamp_u8(q_sig);
        s2_sig[oc] = (float)q_sig / 256.0f;
    }

    // Stage 2 Pass 2: Compute c2 row-by-row, apply SE + Residual AddReLU, write to s2_out flash
    tinytb_flash_read(sc_part, S2_C1_FLASH_OFFSET + 0 * (32 * 56), s2_c1_ring[0], 32 * 56);
    tinytb_flash_read(sc_part, S2_C1_FLASH_OFFSET + 1 * (32 * 56), s2_c1_ring[1], 32 * 56);
    uint8_t* s2_sc_row = (uint8_t*)malloc(32 * 56);
    if (!s2_sc_row) {
        free(s2_c1_ring);
        free(s2_c2_row);
        return -999.0f;
    }

    for (int r = 0; r < 56; r++) {
#if defined(ARDUINO) || defined(ESP32)
        feed_wdt_if_enabled();
        if (r % 14 == 0) Serial.print(".");
        if (r % 4 == 0) yield();
#endif
        if (r + 1 < 56) {
            tinytb_flash_read(sc_part, S2_C1_FLASH_OFFSET + (size_t)(r + 1) * (32 * 56), s2_c1_ring[(r + 1) % 3], 32 * 56);
        }
        const uint8_t* in_3[3] = {
            (r - 1 >= 0) ? s2_c1_ring[(r - 1) % 3] : NULL,
            s2_c1_ring[r % 3],
            (r + 1 < 56) ? s2_c1_ring[(r + 1) % 3] : NULL
        };
        compute_s2_c2_row(in_3, s2_c2_row);
        tinytb_flash_read(sc_part, S2_SC_FLASH_OFFSET + (size_t)r * (32 * 56), s2_sc_row, 32 * 56);

        for (int oc = 0; oc < 32; oc++) {
            float sig = s2_sig[oc];
            uint8_t* p_c2 = &s2_c2_row[oc * 56];
            const uint8_t* p_sc = &s2_sc_row[oc * 56];

            for (int c = 0; c < 56; c++) {
                float r_c2 = ((float)p_c2[c] - S2_C2_OUT_ZERO_POINT) * S2_C2_OUT_SCALE;
                float r_mul = r_c2 * sig;
                int32_t q_mul = (int32_t)roundf(r_mul / S2_SE_SCALE) + S2_SE_ZERO_POINT;
                q_mul = clamp_u8(q_mul);

                float r_se = ((float)q_mul - S2_SE_ZERO_POINT) * S2_SE_SCALE;
                float r_sc = ((float)p_sc[c] - S2_SC_OUT_ZERO_POINT) * S2_SC_OUT_SCALE;
                float r_sum = r_se + r_sc;
                if (r_sum < 0.0f) r_sum = 0.0f;
                int32_t q_out = (int32_t)roundf(r_sum / S2_RELU_SCALE) + S2_RELU_ZERO_POINT;
                p_c2[c] = clamp_u8(q_out);
            }
        }
        tinytb_flash_write(sc_part, S2_OUT_FLASH_OFFSET + (size_t)r * (32 * 56), s2_c2_row, 32 * 56);
    }

    free(s2_c1_ring);
    free(s2_c2_row);
    free(s2_sc_row);

    uint32_t t_s2 = get_current_time_ms();
    if (stats) stats->s2_ms = t_s2 - t_s1;

    // ==========================================
    // STAGE 3 (56x56 -> 28x28)
    // ==========================================
#if defined(ARDUINO) || defined(ESP32)
    Serial.println(" done.");
    Serial.print("  [Stage 3: 64ch 28x28] ");
#endif
    uint8_t* s2_out_row = (uint8_t*)malloc(32 * 56);
    uint8_t* s3_sc_row = (uint8_t*)malloc(64 * 28);
    if (!s2_out_row || !s3_sc_row) {
        if (s2_out_row) free(s2_out_row);
        if (s3_sc_row) free(s3_sc_row);
        return -999.0f;
    }

    cur_h = get_current_free_heap();
    if (cur_h < h_min) h_min = cur_h;

    // Shortcut: 1x1 conv, stride 2, pad 0 -> Stream row-by-row into flash
    for (int oh = 0; oh < 28; oh++) {
#if defined(ARDUINO) || defined(ESP32)
        feed_wdt_if_enabled();
        if (oh % 7 == 0) yield();
#endif
        int ih = oh * 2;
        tinytb_flash_read(sc_part, S2_OUT_FLASH_OFFSET + (size_t)ih * (32 * 56), s2_out_row, 32 * 56);

        for (int oc = 0; oc < 64; oc++) {
            const int8_t* w_oc = &s3_sc_weight[oc * 32];
            float w_sc = s3_sc_w_scale[oc];
            float b = s3_sc_bias[oc];

            for (int ow = 0; ow < 28; ow++) {
                int iw = ow * 2;
                int32_t acc = 0;
                for (int ic = 0; ic < 32; ic++) {
                    int32_t x = (int32_t)s2_out_row[ic * 56 + iw] - S2_RELU_ZERO_POINT;
                    acc += x * (int32_t)w_oc[ic];
                }
                float r_val = (float)acc * (S2_RELU_SCALE * w_sc) + b;
                int32_t q = (int32_t)roundf(r_val / S3_SC_OUT_SCALE) + S3_SC_OUT_ZERO_POINT;
                s3_sc_row[oc * 28 + ow] = clamp_u8(q);
            }
        }
        tinytb_flash_write(sc_part, S3_SC_FLASH_OFFSET + (size_t)oh * (64 * 28), s3_sc_row, 64 * 28);
    }
    free(s2_out_row);

    // Conv1: 3x3 conv, stride 2, pad 1, relu (reads s2_out from flash, writes act64_c1 in SRAM)
    uint8_t (*s2_out_3rows)[32 * 56] = (uint8_t(*)[32 * 56])malloc(3 * 32 * 56);
    uint8_t** act64_c1 = alloc_channels(64, 28 * 28);
    if (!s2_out_3rows || !act64_c1) {
        if (s2_out_3rows) free(s2_out_3rows);
        if (act64_c1) free_channels(act64_c1, 64);
        free(s3_sc_row);
        return -999.0f;
    }

    for (int oh = 0; oh < 28; oh++) {
#if defined(ARDUINO) || defined(ESP32)
        feed_wdt_if_enabled();
        if (oh % 7 == 0) yield();
#endif
        int ih_0 = 2 * oh - 1;
        int ih_1 = 2 * oh;
        int ih_2 = 2 * oh + 1;

        if (ih_0 >= 0 && oh == 0) {
            tinytb_flash_read(sc_part, S2_OUT_FLASH_OFFSET + (size_t)ih_0 * (32 * 56), s2_out_3rows[ih_0 % 3], 32 * 56);
        }
        tinytb_flash_read(sc_part, S2_OUT_FLASH_OFFSET + (size_t)ih_1 * (32 * 56), s2_out_3rows[ih_1 % 3], 32 * 56);
        if (ih_2 < 56) {
            tinytb_flash_read(sc_part, S2_OUT_FLASH_OFFSET + (size_t)ih_2 * (32 * 56), s2_out_3rows[ih_2 % 3], 32 * 56);
        }

        const uint8_t* in_3[3] = {
            (ih_0 >= 0) ? s2_out_3rows[ih_0 % 3] : NULL,
            s2_out_3rows[ih_1 % 3],
            (ih_2 < 56) ? s2_out_3rows[ih_2 % 3] : NULL
        };

        for (int oc = 0; oc < 64; oc++) {
            const int8_t* w_oc = &s3_c1_weight[oc * 32 * 9];
            float w_sc = s3_c1_w_scale[oc];
            float b = s3_c1_bias[oc];

            for (int ow = 0; ow < 28; ow++) {
                int iw_base = ow * 2 - 1;
                int32_t acc = 0;

                for (int ic = 0; ic < 32; ic++) {
                    const int8_t* w_k = &w_oc[ic * 9];
                    for (int kh = 0; kh < 3; kh++) {
                        const uint8_t* in_p = in_3[kh];
                        if (in_p != NULL) {
                            const uint8_t* in_ic = &in_p[ic * 56];
                            for (int kw = 0; kw < 3; kw++) {
                                int iw = iw_base + kw;
                                if (iw >= 0 && iw < 56) {
                                    int32_t x = (int32_t)in_ic[iw] - S2_RELU_ZERO_POINT;
                                    acc += x * (int32_t)w_k[kh * 3 + kw];
                                }
                            }
                        }
                    }
                }

                float r_val = (float)acc * (S2_RELU_SCALE * w_sc) + b;
                if (r_val < 0.0f) r_val = 0.0f;
                int32_t q = (int32_t)roundf(r_val / S3_C1_OUT_SCALE) + S3_C1_OUT_ZERO_POINT;
                act64_c1[oc][oh * 28 + ow] = clamp_u8(q);
            }
        }
    }
    free(s2_out_3rows);

    // Allocate act64_c2
    uint8_t** act64_c2 = alloc_channels(64, 28 * 28);
    if (!act64_c2) {
        free(s3_sc_row);
        free_channels(act64_c1, 64);
        return -999.0f;
    }

    // Stage 3 Conv2 (Single Pass): compute activations and accumulate sums for GAP
    int64_t s3_c2_sums[64] = {0};
    for (int r = 0; r < 28; r++) {
#if defined(ARDUINO) || defined(ESP32)
        feed_wdt_if_enabled();
        if (r % 7 == 0) Serial.print(".");
        if (r % 2 == 0) yield();
#endif
        int r_prev = r - 1;
        int r_curr = r;
        int r_next = r + 1;

        for (int oc = 0; oc < 64; oc++) {
            const int8_t* w_oc = &s3_c2_weight[oc * 64 * 9];
            float w_sc = s3_c2_w_scale[oc];
            float b = s3_c2_bias[oc];

            for (int c = 0; c < 28; c++) {
                int iw_base = c - 1;
                int32_t acc = 0;

                for (int ic = 0; ic < 64; ic++) {
                    const uint8_t* plane = act64_c1[ic];
                    const int8_t* w_k = &w_oc[ic * 9];

                    if (r_prev >= 0) {
                        const uint8_t* p = &plane[r_prev * 28];
                        for (int kw = 0; kw < 3; kw++) {
                            int iw = iw_base + kw;
                            if (iw >= 0 && iw < 28) acc += ((int32_t)p[iw] - S3_C1_OUT_ZERO_POINT) * (int32_t)w_k[kw];
                        }
                    }
                    {
                        const uint8_t* p = &plane[r_curr * 28];
                        for (int kw = 0; kw < 3; kw++) {
                            int iw = iw_base + kw;
                            if (iw >= 0 && iw < 28) acc += ((int32_t)p[iw] - S3_C1_OUT_ZERO_POINT) * (int32_t)w_k[3 + kw];
                        }
                    }
                    if (r_next < 28) {
                        const uint8_t* p = &plane[r_next * 28];
                        for (int kw = 0; kw < 3; kw++) {
                            int iw = iw_base + kw;
                            if (iw >= 0 && iw < 28) acc += ((int32_t)p[iw] - S3_C1_OUT_ZERO_POINT) * (int32_t)w_k[6 + kw];
                        }
                    }
                }

                float r_val = (float)acc * (S3_C1_OUT_SCALE * w_sc) + b;
                int32_t q = (int32_t)roundf(r_val / S3_C2_OUT_SCALE) + S3_C2_OUT_ZERO_POINT;
                uint8_t q_u8 = clamp_u8(q);
                act64_c2[oc][r * 28 + c] = q_u8;
                s3_c2_sums[oc] += q_u8;
            }
        }
    }

    // act64_c1 is done, free immediately
    free_channels(act64_c1, 64);

    uint8_t s3_se_gap[64];
    for (int oc = 0; oc < 64; oc++) {
        s3_se_gap[oc] = (uint8_t)roundf((float)s3_c2_sums[oc] / (28.0f * 28.0f));
    }

    uint8_t s3_se1_out[8];
    for (int oc = 0; oc < 8; oc++) {
        int32_t acc = 0;
        const int8_t* w_oc = &s3_se1_weight[oc * 64];
        for (int ic = 0; ic < 64; ic++) {
            acc += ((int32_t)s3_se_gap[ic] - S3_C2_OUT_ZERO_POINT) * (int32_t)w_oc[ic];
        }
        float r_val = (float)acc * (S3_C2_OUT_SCALE * s3_se1_w_scale[oc]) + s3_se1_bias[oc];
        if (r_val < 0.0f) r_val = 0.0f;
        int32_t q = (int32_t)roundf(r_val / S3_SE1_OUT_SCALE) + S3_SE1_OUT_ZERO_POINT;
        s3_se1_out[oc] = clamp_u8(q);
    }

    float s3_sig[64];
    for (int oc = 0; oc < 64; oc++) {
        int32_t acc = 0;
        const int8_t* w_oc = &s3_se2_weight[oc * 8];
        for (int ic = 0; ic < 8; ic++) {
            acc += ((int32_t)s3_se1_out[ic] - S3_SE1_OUT_ZERO_POINT) * (int32_t)w_oc[ic];
        }
        float r_val = (float)acc * (S3_SE1_OUT_SCALE * s3_se2_w_scale[oc]) + s3_se2_bias[oc];
        int32_t q = (int32_t)roundf(r_val / S3_SE2_OUT_SCALE) + S3_SE2_OUT_ZERO_POINT;
        q = clamp_u8(q);
        float r_deq = ((float)q - S3_SE2_OUT_ZERO_POINT) * S3_SE2_OUT_SCALE;
        float sig_f = 1.0f / (1.0f + expf(-r_deq));
        int32_t q_sig = (int32_t)roundf(sig_f * 256.0f);
        q_sig = clamp_u8(q_sig);
        s3_sig[oc] = (float)q_sig / 256.0f;
    }

    // Allocate act64_out for Stage 3 output
    uint8_t** act64_out = alloc_channels(64, 28 * 28);
    if (!act64_out) {
        free(s3_sc_row);
        free_channels(act64_c2, 64);
        return -999.0f;
    }

    // Apply SE scaling and residual addition row-by-row reading s3_sc from flash
    for (int r = 0; r < 28; r++) {
#if defined(ARDUINO) || defined(ESP32)
        feed_wdt_if_enabled();
        if (r % 7 == 0) yield();
#endif
        tinytb_flash_read(sc_part, S3_SC_FLASH_OFFSET + (size_t)r * (64 * 28), s3_sc_row, 64 * 28);

        for (int oc = 0; oc < 64; oc++) {
            float sig = s3_sig[oc];
            const uint8_t* sc_plane = &s3_sc_row[oc * 28];
            const uint8_t* c2_row = &act64_c2[oc][r * 28];
            uint8_t* out_row = &act64_out[oc][r * 28];

            for (int c = 0; c < 28; c++) {
                uint8_t q_c2 = c2_row[c];
                float r_c2_f = ((float)q_c2 - S3_C2_OUT_ZERO_POINT) * S3_C2_OUT_SCALE;
                float r_mul = r_c2_f * sig;
                int32_t q_mul = (int32_t)roundf(r_mul / S3_SE_SCALE) + S3_SE_ZERO_POINT;
                q_mul = clamp_u8(q_mul);

                float r_se = ((float)q_mul - S3_SE_ZERO_POINT) * S3_SE_SCALE;
                float r_sc = ((float)sc_plane[c] - S3_SC_OUT_ZERO_POINT) * S3_SC_OUT_SCALE;
                float r_sum = r_se + r_sc;
                if (r_sum < 0.0f) r_sum = 0.0f;
                int32_t q_out = (int32_t)roundf(r_sum / S3_RELU_SCALE) + S3_RELU_ZERO_POINT;
                out_row[c] = clamp_u8(q_out);
            }
        }
    }

    free(s3_sc_row);
    free_channels(act64_c2, 64);

    uint32_t t_s3 = get_current_time_ms();
    if (stats) stats->s3_ms = t_s3 - t_s2;

    // ==========================================
    // STAGE 4 (28x28 -> 14x14)
    // ==========================================
#if defined(ARDUINO) || defined(ESP32)
    Serial.println(" done.");
    Serial.print("  [Stage 4: 128ch 14x14] ");
#endif
    uint8_t** act128_sc = alloc_channels(128, 14 * 14);
    uint8_t** act128_c1 = alloc_channels(128, 14 * 14);
    if (!act128_sc || !act128_c1) {
        free_channels(act64_out, 64);
        if (act128_sc) free_channels(act128_sc, 128);
        if (act128_c1) free_channels(act128_c1, 128);
        return -999.0f;
    }

    cur_h = get_current_free_heap();
    if (cur_h < h_min) h_min = cur_h;

    // Shortcut: 1x1 conv, stride 2, pad 0
    for (int oc = 0; oc < 128; oc++) {
#if defined(ARDUINO) || defined(ESP32)
        feed_wdt_if_enabled();
        if (oc % 16 == 0) yield();
#endif
        const int8_t* w_oc = &s4_sc_weight[oc * 64];
        float w_sc = s4_sc_w_scale[oc];
        float b = s4_sc_bias[oc];

        for (int oh = 0; oh < 14; oh++) {
            int ih = oh * 2;
            for (int ow = 0; ow < 14; ow++) {
                int iw = ow * 2;
                int32_t acc = 0;
                for (int ic = 0; ic < 64; ic++) {
                    int32_t x = (int32_t)act64_out[ic][ih * 28 + iw] - S3_RELU_ZERO_POINT;
                    acc += x * (int32_t)w_oc[ic];
                }
                float r_val = (float)acc * (S3_RELU_SCALE * w_sc) + b;
                int32_t q = (int32_t)roundf(r_val / S4_SC_OUT_SCALE) + S4_SC_OUT_ZERO_POINT;
                act128_sc[oc][oh * 14 + ow] = clamp_u8(q);
            }
        }
    }

    // Conv1: 3x3 conv, stride 2, pad 1, relu
    for (int oc = 0; oc < 128; oc++) {
#if defined(ARDUINO) || defined(ESP32)
        feed_wdt_if_enabled();
        if (oc % 16 == 0) yield();
#endif
        const int8_t* w_oc = &s4_c1_weight[oc * 64 * 9];
        float w_sc = s4_c1_w_scale[oc];
        float b = s4_c1_bias[oc];

        for (int oh = 0; oh < 14; oh++) {
            int ih_base = oh * 2 - 1;
            for (int ow = 0; ow < 14; ow++) {
                int iw_base = ow * 2 - 1;
                int32_t acc = 0;

                for (int ic = 0; ic < 64; ic++) {
                    const uint8_t* plane = act64_out[ic];
                    const int8_t* w_k = &w_oc[ic * 9];
                    for (int kh = 0; kh < 3; kh++) {
                        int ih = ih_base + kh;
                        if (ih >= 0 && ih < 28) {
                            const uint8_t* p = &plane[ih * 28];
                            for (int kw = 0; kw < 3; kw++) {
                                int iw = iw_base + kw;
                                if (iw >= 0 && iw < 28) {
                                    acc += ((int32_t)p[iw] - S3_RELU_ZERO_POINT) * (int32_t)w_k[kh * 3 + kw];
                                }
                            }
                        }
                    }
                }

                float r_val = (float)acc * (S3_RELU_SCALE * w_sc) + b;
                if (r_val < 0.0f) r_val = 0.0f;
                int32_t q = (int32_t)roundf(r_val / S4_C1_OUT_SCALE) + S4_C1_OUT_ZERO_POINT;
                act128_c1[oc][oh * 14 + ow] = clamp_u8(q);
            }
        }
    }

    free_channels(act64_out, 64);

    // Allocate act128_c2 now that act64_sc (49KB) has been freed
    uint8_t** act128_c2 = alloc_channels(128, 14 * 14);
    if (!act128_c2) {
        free_channels(act128_sc, 128);
        free_channels(act128_c1, 128);
        return -999.0f;
    }

    // Stage 4 Conv2 (Single Pass): compute activations and accumulate sums for GAP
    int64_t s4_c2_sums[128] = {0};
    for (int r = 0; r < 14; r++) {
#if defined(ARDUINO) || defined(ESP32)
        feed_wdt_if_enabled();
        if (r % 4 == 0) Serial.print(".");
        yield();
#endif
        int r_prev = r - 1;
        int r_curr = r;
        int r_next = r + 1;

        for (int oc = 0; oc < 128; oc++) {
            const int8_t* w_oc = &s4_c2_weight[oc * 128 * 9];
            float w_sc = s4_c2_w_scale[oc];
            float b = s4_c2_bias[oc];

            for (int c = 0; c < 14; c++) {
                int iw_base = c - 1;
                int32_t acc = 0;

                for (int ic = 0; ic < 128; ic++) {
                    const uint8_t* plane = act128_c1[ic];
                    const int8_t* w_k = &w_oc[ic * 9];

                    if (r_prev >= 0) {
                        const uint8_t* p = &plane[r_prev * 14];
                        for (int kw = 0; kw < 3; kw++) {
                            int iw = iw_base + kw;
                            if (iw >= 0 && iw < 14) acc += ((int32_t)p[iw] - S4_C1_OUT_ZERO_POINT) * (int32_t)w_k[kw];
                        }
                    }
                    {
                        const uint8_t* p = &plane[r_curr * 14];
                        for (int kw = 0; kw < 3; kw++) {
                            int iw = iw_base + kw;
                            if (iw >= 0 && iw < 14) acc += ((int32_t)p[iw] - S4_C1_OUT_ZERO_POINT) * (int32_t)w_k[3 + kw];
                        }
                    }
                    if (r_next < 14) {
                        const uint8_t* p = &plane[r_next * 14];
                        for (int kw = 0; kw < 3; kw++) {
                            int iw = iw_base + kw;
                            if (iw >= 0 && iw < 14) acc += ((int32_t)p[iw] - S4_C1_OUT_ZERO_POINT) * (int32_t)w_k[6 + kw];
                        }
                    }
                }

                float r_val = (float)acc * (S4_C1_OUT_SCALE * w_sc) + b;
                int32_t q = (int32_t)roundf(r_val / S4_C2_OUT_SCALE) + S4_C2_OUT_ZERO_POINT;
                uint8_t q_u8 = clamp_u8(q);
                act128_c2[oc][r * 14 + c] = q_u8;
                s4_c2_sums[oc] += q_u8;
            }
        }
    }

    // act128_c1 is done, free immediately
    free_channels(act128_c1, 128);

    uint8_t s4_se_gap[128];
    for (int oc = 0; oc < 128; oc++) {
        s4_se_gap[oc] = (uint8_t)roundf((float)s4_c2_sums[oc] / (14.0f * 14.0f));
    }

    uint8_t s4_se1_out[16];
    for (int oc = 0; oc < 16; oc++) {
        int32_t acc = 0;
        const int8_t* w_oc = &s4_se1_weight[oc * 128];
        for (int ic = 0; ic < 128; ic++) {
            acc += ((int32_t)s4_se_gap[ic] - S4_C2_OUT_ZERO_POINT) * (int32_t)w_oc[ic];
        }
        float r_val = (float)acc * (S4_C2_OUT_SCALE * s4_se1_w_scale[oc]) + s4_se1_bias[oc];
        if (r_val < 0.0f) r_val = 0.0f;
        int32_t q = (int32_t)roundf(r_val / S4_SE1_OUT_SCALE) + S4_SE1_OUT_ZERO_POINT;
        s4_se1_out[oc] = clamp_u8(q);
    }

    float s4_sig[128];
    for (int oc = 0; oc < 128; oc++) {
        int32_t acc = 0;
        const int8_t* w_oc = &s4_se2_weight[oc * 16];
        for (int ic = 0; ic < 16; ic++) {
            acc += ((int32_t)s4_se1_out[ic] - S4_SE1_OUT_ZERO_POINT) * (int32_t)w_oc[ic];
        }
        float r_val = (float)acc * (S4_SE1_OUT_SCALE * s4_se2_w_scale[oc]) + s4_se2_bias[oc];
        int32_t q = (int32_t)roundf(r_val / S4_SE2_OUT_SCALE) + S4_SE2_OUT_ZERO_POINT;
        q = clamp_u8(q);
        float r_deq = ((float)q - S4_SE2_OUT_ZERO_POINT) * S4_SE2_OUT_SCALE;
        float sig_f = 1.0f / (1.0f + expf(-r_deq));
        int32_t q_sig = (int32_t)roundf(sig_f * 256.0f);
        q_sig = clamp_u8(q_sig);
        s4_sig[oc] = (float)q_sig / 256.0f;
    }

    // Apply SE scaling and residual addition pointwise (fast single pass, no conv)
    for (int oc = 0; oc < 128; oc++) {
#if defined(ARDUINO) || defined(ESP32)
        feed_wdt_if_enabled();
        if (oc % 32 == 0) yield();
#endif
        float sig = s4_sig[oc];
        uint8_t* sc_plane = act128_sc[oc];
        const uint8_t* c2_plane = act128_c2[oc];

        for (int i = 0; i < 14 * 14; i++) {
            uint8_t q_c2 = c2_plane[i];
            float r_c2_f = ((float)q_c2 - S4_C2_OUT_ZERO_POINT) * S4_C2_OUT_SCALE;
            float r_mul = r_c2_f * sig;
            int32_t q_mul = (int32_t)roundf(r_mul / S4_SE_SCALE) + S4_SE_ZERO_POINT;
            q_mul = clamp_u8(q_mul);

            float r_se = ((float)q_mul - S4_SE_ZERO_POINT) * S4_SE_SCALE;
            float r_sc = ((float)sc_plane[i] - S4_SC_OUT_ZERO_POINT) * S4_SC_OUT_SCALE;
            float r_sum = r_se + r_sc;
            if (r_sum < 0.0f) r_sum = 0.0f;
            int32_t q_out = (int32_t)roundf(r_sum / S4_RELU_SCALE) + S4_RELU_ZERO_POINT;
            sc_plane[i] = clamp_u8(q_out);
        }
    }

    free_channels(act128_c2, 128);

    uint32_t t_s4 = get_current_time_ms();
    if (stats) stats->s4_ms = t_s4 - t_s3;

    // ==========================================
    // GAP (128 channels x 14x14) & CLASSIFIER
    // ==========================================
#if defined(ARDUINO) || defined(ESP32)
    Serial.println(" done.");
    Serial.println("  [GAP + Classifier] ");
#endif
    uint8_t gap_out[128];
    for (int oc = 0; oc < 128; oc++) {
        const uint8_t* p = act128_sc[oc];
        int32_t sum = 0;
        for (int i = 0; i < 196; i++) {
            sum += p[i];
        }
        gap_out[oc] = (uint8_t)roundf((float)sum / 196.0f);
    }

    free_channels(act128_sc, 128);

    // Linear Classifier
    int32_t acc = 0;
    for (int ic = 0; ic < 128; ic++) {
        acc += ((int32_t)gap_out[ic] - S4_RELU_ZERO_POINT) * (int32_t)fc_weight[ic];
    }
    float raw_val = (float)acc * (S4_RELU_SCALE * fc_w_scale[0]) + fc_bias[0];
    int32_t q_logit = (int32_t)roundf(raw_val / FC_OUT_SCALE) + FC_OUT_ZERO_POINT;
    float logit = ((float)q_logit - FC_OUT_ZERO_POINT) * FC_OUT_SCALE;
    float prob = 1.0f / (1.0f + expf(-logit));

    uint32_t t_end = get_current_time_ms();
    size_t h_end = get_current_free_heap();

    if (stats) {
        stats->classifier_ms = t_end - t_s4;
        stats->total_ms = t_end - t_start;
        stats->heap_start = h_start;
        stats->heap_min = h_min;
        stats->heap_end = h_end;
    }

    if (out_prob) *out_prob = prob;
    return logit;
}

#endif // TINYTB_INFER_H
