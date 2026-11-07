#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "tinytb_weights.h"
#include "tinytb_samples.h"

// Allocation helpers
static uint8_t** alloc_channels(int num_channels, int plane_size) {
    uint8_t** ptrs = (uint8_t**)malloc(num_channels * sizeof(uint8_t*));
    if (!ptrs) return NULL;
    for (int c = 0; c < num_channels; c++) {
        ptrs[c] = (uint8_t*)malloc(plane_size);
        if (!ptrs[c]) {
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
            if (q < 0) q = 0;
            if (q > 255) q = 255;
            stem_row[oc * 112 + c] = (uint8_t)q;
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
            if (q < 0) q = 0;
            if (q > 255) q = 255;
            out_row[oc * width + c] = (uint8_t)q;
        }
    }
}

// Full inference function
float tinytb_predict(const uint8_t* img, float* out_prob) {
    // Ring buffers for Stage 1 streaming (3 slots each, 16*112 bytes)
    uint8_t stem_buf[3][16 * 112];
    uint8_t c1_buf[3][16 * 112];
    uint8_t c2_row[16 * 112];

    // ==========================================
    // STAGE 1 - PASS 1: Accumulate SE GAP sums
    // ==========================================
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
        // Need c1 row r+1 (if r+1 < 112)
        if (r + 1 < 112) {
            // Need stem row r+2 (if r+2 < 112)
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
        if (q < 0) q = 0; if (q > 255) q = 255;
        s1_se1_out[oc] = (uint8_t)q;
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
        if (q < 0) q = 0; if (q > 255) q = 255;
        float r_deq = ((float)q - S1_SE2_OUT_ZERO_POINT) * S1_SE2_OUT_SCALE;
        float sig_f = 1.0f / (1.0f + expf(-r_deq));
        int32_t q_sig = (int32_t)roundf(sig_f * 256.0f);
        if (q_sig < 0) q_sig = 0; if (q_sig > 255) q_sig = 255;
        s1_sig[oc] = (float)q_sig / 256.0f;
    }

    // ==========================================
    // STAGE 1 - PASS 2 -> STREAM DIRECTLY INTO STAGE 2
    // ==========================================
    uint8_t** act32_sc = alloc_channels(32, 56 * 56);
    uint8_t** act32_c1 = alloc_channels(32, 56 * 56);
    uint8_t s1_out_buf[3][16 * 112];

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
                if (q_mul < 0) q_mul = 0; if (q_mul > 255) q_mul = 255;

                float r_se = ((float)q_mul - S1_SE_ZERO_POINT) * S1_SE_SCALE;
                float r_sc = ((float)p_stem[c] - STEM_CONV_OUT_ZERO_POINT) * STEM_CONV_OUT_SCALE;
                float r_sum = r_se + r_sc;
                if (r_sum < 0.0f) r_sum = 0.0f;
                int32_t q_out = (int32_t)roundf(r_sum / S1_RELU_SCALE) + S1_RELU_ZERO_POINT;
                if (q_out < 0) q_out = 0; if (q_out > 255) q_out = 255;
                p_out[c] = (uint8_t)q_out;
            }
        }

        // Stream into Stage 2 Shortcut: when r is even (r = 2 * oh)
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
                    if (q < 0) q = 0; if (q > 255) q = 255;
                    act32_sc[oc][oh * 56 + ow] = (uint8_t)q;
                }
            }
        }

        // Stream into Stage 2 Conv1: needs input rows 2oh-1, 2oh, 2oh+1
        // When r = 2*oh + 1 (i.e. r is odd): row 2*oh+1 has just been computed!
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
                    if (q < 0) q = 0; if (q > 255) q = 255;
                    act32_c1[oc][oh * 56 + ow] = (uint8_t)q;
                }
            }
        }
    }

    // ==========================================
    // STAGE 2: conv2 (from act32_c1) + SE + Residual AddReLU
    // ==========================================
    // Pass 1: Accumulate SE GAP
    int64_t s2_c2_sums[32] = {0};
    for (int r = 0; r < 56; r++) {
        int r_prev = r - 1;
        int r_curr = r;
        int r_next = r + 1;

        for (int oc = 0; oc < 32; oc++) {
            const int8_t* w_oc = &s2_c2_weight[oc * 32 * 9];
            float w_sc = s2_c2_w_scale[oc];
            float b = s2_c2_bias[oc];

            for (int c = 0; c < 56; c++) {
                int iw_base = c - 1;
                int32_t acc = 0;

                for (int ic = 0; ic < 32; ic++) {
                    const uint8_t* plane = act32_c1[ic];
                    const int8_t* w_k = &w_oc[ic * 9];

                    if (r_prev >= 0) {
                        const uint8_t* p = &plane[r_prev * 56];
                        for (int kw = 0; kw < 3; kw++) {
                            int iw = iw_base + kw;
                            if (iw >= 0 && iw < 56) acc += ((int32_t)p[iw] - S2_C1_OUT_ZERO_POINT) * (int32_t)w_k[kw];
                        }
                    }
                    {
                        const uint8_t* p = &plane[r_curr * 56];
                        for (int kw = 0; kw < 3; kw++) {
                            int iw = iw_base + kw;
                            if (iw >= 0 && iw < 56) acc += ((int32_t)p[iw] - S2_C1_OUT_ZERO_POINT) * (int32_t)w_k[3 + kw];
                        }
                    }
                    if (r_next < 56) {
                        const uint8_t* p = &plane[r_next * 56];
                        for (int kw = 0; kw < 3; kw++) {
                            int iw = iw_base + kw;
                            if (iw >= 0 && iw < 56) acc += ((int32_t)p[iw] - S2_C1_OUT_ZERO_POINT) * (int32_t)w_k[6 + kw];
                        }
                    }
                }

                float r_val = (float)acc * (S2_C1_OUT_SCALE * w_sc) + b;
                int32_t q = (int32_t)roundf(r_val / S2_C2_OUT_SCALE) + S2_C2_OUT_ZERO_POINT;
                if (q < 0) q = 0; if (q > 255) q = 255;
                s2_c2_sums[oc] += q;
            }
        }
    }

    // Compute Stage 2 SE Sigmoid
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
        if (q < 0) q = 0; if (q > 255) q = 255;
        s2_se1_out[oc] = (uint8_t)q;
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
        if (q < 0) q = 0; if (q > 255) q = 255;
        float r_deq = ((float)q - S2_SE2_OUT_ZERO_POINT) * S2_SE2_OUT_SCALE;
        float sig_f = 1.0f / (1.0f + expf(-r_deq));
        int32_t q_sig = (int32_t)roundf(sig_f * 256.0f);
        if (q_sig < 0) q_sig = 0; if (q_sig > 255) q_sig = 255;
        s2_sig[oc] = (float)q_sig / 256.0f;
    }

    // Pass 2: Compute c2 row-by-row and write AddReLU(Mul(c2, sig), s2_sc) in-place into act32_sc
    for (int r = 0; r < 56; r++) {
        int r_prev = r - 1;
        int r_curr = r;
        int r_next = r + 1;

        for (int oc = 0; oc < 32; oc++) {
            const int8_t* w_oc = &s2_c2_weight[oc * 32 * 9];
            float w_sc = s2_c2_w_scale[oc];
            float b = s2_c2_bias[oc];
            float sig = s2_sig[oc];
            uint8_t* sc_plane = act32_sc[oc];

            for (int c = 0; c < 56; c++) {
                int iw_base = c - 1;
                int32_t acc = 0;

                for (int ic = 0; ic < 32; ic++) {
                    const uint8_t* plane = act32_c1[ic];
                    const int8_t* w_k = &w_oc[ic * 9];

                    if (r_prev >= 0) {
                        const uint8_t* p = &plane[r_prev * 56];
                        for (int kw = 0; kw < 3; kw++) {
                            int iw = iw_base + kw;
                            if (iw >= 0 && iw < 56) acc += ((int32_t)p[iw] - S2_C1_OUT_ZERO_POINT) * (int32_t)w_k[kw];
                        }
                    }
                    {
                        const uint8_t* p = &plane[r_curr * 56];
                        for (int kw = 0; kw < 3; kw++) {
                            int iw = iw_base + kw;
                            if (iw >= 0 && iw < 56) acc += ((int32_t)p[iw] - S2_C1_OUT_ZERO_POINT) * (int32_t)w_k[3 + kw];
                        }
                    }
                    if (r_next < 56) {
                        const uint8_t* p = &plane[r_next * 56];
                        for (int kw = 0; kw < 3; kw++) {
                            int iw = iw_base + kw;
                            if (iw >= 0 && iw < 56) acc += ((int32_t)p[iw] - S2_C1_OUT_ZERO_POINT) * (int32_t)w_k[6 + kw];
                        }
                    }
                }

                float r_c2 = (float)acc * (S2_C1_OUT_SCALE * w_sc) + b;
                int32_t q_c2 = (int32_t)roundf(r_c2 / S2_C2_OUT_SCALE) + S2_C2_OUT_ZERO_POINT;
                if (q_c2 < 0) q_c2 = 0; if (q_c2 > 255) q_c2 = 255;

                // Mul:
                float r_c2_f = ((float)q_c2 - S2_C2_OUT_ZERO_POINT) * S2_C2_OUT_SCALE;
                float r_mul = r_c2_f * sig;
                int32_t q_mul = (int32_t)roundf(r_mul / S2_SE_SCALE) + S2_SE_ZERO_POINT;
                if (q_mul < 0) q_mul = 0; if (q_mul > 255) q_mul = 255;

                // AddReLU:
                float r_se = ((float)q_mul - S2_SE_ZERO_POINT) * S2_SE_SCALE;
                float r_sc = ((float)sc_plane[r * 56 + c] - S2_SC_OUT_ZERO_POINT) * S2_SC_OUT_SCALE;
                float r_sum = r_se + r_sc;
                if (r_sum < 0.0f) r_sum = 0.0f;
                int32_t q_out = (int32_t)roundf(r_sum / S2_RELU_SCALE) + S2_RELU_ZERO_POINT;
                if (q_out < 0) q_out = 0; if (q_out > 255) q_out = 255;
                sc_plane[r * 56 + c] = (uint8_t)q_out;
            }
        }
    }

    free_channels(act32_c1, 32);

    // ==========================================
    // STAGE 3 (56x56 -> 28x28)
    // ==========================================
    uint8_t** act64_sc = alloc_channels(64, 28 * 28);
    uint8_t** act64_c1 = alloc_channels(64, 28 * 28);

    // Shortcut: 1x1 conv, stride 2, pad 0
    for (int oc = 0; oc < 64; oc++) {
        const int8_t* w_oc = &s3_sc_weight[oc * 32];
        float w_sc = s3_sc_w_scale[oc];
        float b = s3_sc_bias[oc];

        for (int oh = 0; oh < 28; oh++) {
            int ih = oh * 2;
            for (int ow = 0; ow < 28; ow++) {
                int iw = ow * 2;
                int32_t acc = 0;
                for (int ic = 0; ic < 32; ic++) {
                    int32_t x = (int32_t)act32_sc[ic][ih * 56 + iw] - S2_RELU_ZERO_POINT;
                    acc += x * (int32_t)w_oc[ic];
                }
                float r_val = (float)acc * (S2_RELU_SCALE * w_sc) + b;
                int32_t q = (int32_t)roundf(r_val / S3_SC_OUT_SCALE) + S3_SC_OUT_ZERO_POINT;
                if (q < 0) q = 0; if (q > 255) q = 255;
                act64_sc[oc][oh * 28 + ow] = (uint8_t)q;
            }
        }
    }

    // Conv1: 3x3 conv, stride 2, pad 1, relu
    for (int oc = 0; oc < 64; oc++) {
        const int8_t* w_oc = &s3_c1_weight[oc * 32 * 9];
        float w_sc = s3_c1_w_scale[oc];
        float b = s3_c1_bias[oc];

        for (int oh = 0; oh < 28; oh++) {
            int ih_base = oh * 2 - 1;
            for (int ow = 0; ow < 28; ow++) {
                int iw_base = ow * 2 - 1;
                int32_t acc = 0;

                for (int ic = 0; ic < 32; ic++) {
                    const uint8_t* plane = act32_sc[ic];
                    const int8_t* w_k = &w_oc[ic * 9];
                    for (int kh = 0; kh < 3; kh++) {
                        int ih = ih_base + kh;
                        if (ih >= 0 && ih < 56) {
                            const uint8_t* p = &plane[ih * 56];
                            for (int kw = 0; kw < 3; kw++) {
                                int iw = iw_base + kw;
                                if (iw >= 0 && iw < 56) {
                                    acc += ((int32_t)p[iw] - S2_RELU_ZERO_POINT) * (int32_t)w_k[kh * 3 + kw];
                                }
                            }
                        }
                    }
                }

                float r_val = (float)acc * (S2_RELU_SCALE * w_sc) + b;
                if (r_val < 0.0f) r_val = 0.0f;
                int32_t q = (int32_t)roundf(r_val / S3_C1_OUT_SCALE) + S3_C1_OUT_ZERO_POINT;
                if (q < 0) q = 0; if (q > 255) q = 255;
                act64_c1[oc][oh * 28 + ow] = (uint8_t)q;
            }
        }
    }

    // Free act32_sc
    free_channels(act32_sc, 32);

    // Stage 3 Conv2 + SE + Residual AddReLU
    int64_t s3_c2_sums[64] = {0};
    for (int r = 0; r < 28; r++) {
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
                if (q < 0) q = 0; if (q > 255) q = 255;
                s3_c2_sums[oc] += q;
            }
        }
    }

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
        if (q < 0) q = 0; if (q > 255) q = 255;
        s3_se1_out[oc] = (uint8_t)q;
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
        if (q < 0) q = 0; if (q > 255) q = 255;
        float r_deq = ((float)q - S3_SE2_OUT_ZERO_POINT) * S3_SE2_OUT_SCALE;
        float sig_f = 1.0f / (1.0f + expf(-r_deq));
        int32_t q_sig = (int32_t)roundf(sig_f * 256.0f);
        if (q_sig < 0) q_sig = 0; if (q_sig > 255) q_sig = 255;
        s3_sig[oc] = (float)q_sig / 256.0f;
    }

    // Stage 3 Pass 2 into act64_sc in-place
    for (int r = 0; r < 28; r++) {
        int r_prev = r - 1;
        int r_curr = r;
        int r_next = r + 1;

        for (int oc = 0; oc < 64; oc++) {
            const int8_t* w_oc = &s3_c2_weight[oc * 64 * 9];
            float w_sc = s3_c2_w_scale[oc];
            float b = s3_c2_bias[oc];
            float sig = s3_sig[oc];
            uint8_t* sc_plane = act64_sc[oc];

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

                float r_c2 = (float)acc * (S3_C1_OUT_SCALE * w_sc) + b;
                int32_t q_c2 = (int32_t)roundf(r_c2 / S3_C2_OUT_SCALE) + S3_C2_OUT_ZERO_POINT;
                if (q_c2 < 0) q_c2 = 0; if (q_c2 > 255) q_c2 = 255;

                float r_c2_f = ((float)q_c2 - S3_C2_OUT_ZERO_POINT) * S3_C2_OUT_SCALE;
                float r_mul = r_c2_f * sig;
                int32_t q_mul = (int32_t)roundf(r_mul / S3_SE_SCALE) + S3_SE_ZERO_POINT;
                if (q_mul < 0) q_mul = 0; if (q_mul > 255) q_mul = 255;

                float r_se = ((float)q_mul - S3_SE_ZERO_POINT) * S3_SE_SCALE;
                float r_sc = ((float)sc_plane[r * 28 + c] - S3_SC_OUT_ZERO_POINT) * S3_SC_OUT_SCALE;
                float r_sum = r_se + r_sc;
                if (r_sum < 0.0f) r_sum = 0.0f;
                int32_t q_out = (int32_t)roundf(r_sum / S3_RELU_SCALE) + S3_RELU_ZERO_POINT;
                if (q_out < 0) q_out = 0; if (q_out > 255) q_out = 255;
                sc_plane[r * 28 + c] = (uint8_t)q_out;
            }
        }
    }

    free_channels(act64_c1, 64);

    // ==========================================
    // STAGE 4 (28x28 -> 14x14)
    // ==========================================
    uint8_t** act128_sc = alloc_channels(128, 14 * 14);
    uint8_t** act128_c1 = alloc_channels(128, 14 * 14);

    // Shortcut: 1x1 conv, stride 2, pad 0
    for (int oc = 0; oc < 128; oc++) {
        const int8_t* w_oc = &s4_sc_weight[oc * 64];
        float w_sc = s4_sc_w_scale[oc];
        float b = s4_sc_bias[oc];

        for (int oh = 0; oh < 14; oh++) {
            int ih = oh * 2;
            for (int ow = 0; ow < 14; ow++) {
                int iw = ow * 2;
                int32_t acc = 0;
                for (int ic = 0; ic < 64; ic++) {
                    int32_t x = (int32_t)act64_sc[ic][ih * 28 + iw] - S3_RELU_ZERO_POINT;
                    acc += x * (int32_t)w_oc[ic];
                }
                float r_val = (float)acc * (S3_RELU_SCALE * w_sc) + b;
                int32_t q = (int32_t)roundf(r_val / S4_SC_OUT_SCALE) + S4_SC_OUT_ZERO_POINT;
                if (q < 0) q = 0; if (q > 255) q = 255;
                act128_sc[oc][oh * 14 + ow] = (uint8_t)q;
            }
        }
    }

    // Conv1: 3x3 conv, stride 2, pad 1, relu
    for (int oc = 0; oc < 128; oc++) {
        const int8_t* w_oc = &s4_c1_weight[oc * 64 * 9];
        float w_sc = s4_c1_w_scale[oc];
        float b = s4_c1_bias[oc];

        for (int oh = 0; oh < 14; oh++) {
            int ih_base = oh * 2 - 1;
            for (int ow = 0; ow < 14; ow++) {
                int iw_base = ow * 2 - 1;
                int32_t acc = 0;

                for (int ic = 0; ic < 64; ic++) {
                    const uint8_t* plane = act64_sc[ic];
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
                if (q < 0) q = 0; if (q > 255) q = 255;
                act128_c1[oc][oh * 14 + ow] = (uint8_t)q;
            }
        }
    }

    // Free act64_sc
    free_channels(act64_sc, 64);

    // Stage 4 Conv2 + SE + Residual AddReLU
    int64_t s4_c2_sums[128] = {0};
    for (int r = 0; r < 14; r++) {
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
                if (q < 0) q = 0; if (q > 255) q = 255;
                s4_c2_sums[oc] += q;
            }
        }
    }

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
        if (q < 0) q = 0; if (q > 255) q = 255;
        s4_se1_out[oc] = (uint8_t)q;
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
        if (q < 0) q = 0; if (q > 255) q = 255;
        float r_deq = ((float)q - S4_SE2_OUT_ZERO_POINT) * S4_SE2_OUT_SCALE;
        float sig_f = 1.0f / (1.0f + expf(-r_deq));
        int32_t q_sig = (int32_t)roundf(sig_f * 256.0f);
        if (q_sig < 0) q_sig = 0; if (q_sig > 255) q_sig = 255;
        s4_sig[oc] = (float)q_sig / 256.0f;
    }

    // Stage 4 Pass 2 into act128_sc in-place
    for (int r = 0; r < 14; r++) {
        int r_prev = r - 1;
        int r_curr = r;
        int r_next = r + 1;

        for (int oc = 0; oc < 128; oc++) {
            const int8_t* w_oc = &s4_c2_weight[oc * 128 * 9];
            float w_sc = s4_c2_w_scale[oc];
            float b = s4_c2_bias[oc];
            float sig = s4_sig[oc];
            uint8_t* sc_plane = act128_sc[oc];

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

                float r_c2 = (float)acc * (S4_C1_OUT_SCALE * w_sc) + b;
                int32_t q_c2 = (int32_t)roundf(r_c2 / S4_C2_OUT_SCALE) + S4_C2_OUT_ZERO_POINT;
                if (q_c2 < 0) q_c2 = 0; if (q_c2 > 255) q_c2 = 255;

                float r_c2_f = ((float)q_c2 - S4_C2_OUT_ZERO_POINT) * S4_C2_OUT_SCALE;
                float r_mul = r_c2_f * sig;
                int32_t q_mul = (int32_t)roundf(r_mul / S4_SE_SCALE) + S4_SE_ZERO_POINT;
                if (q_mul < 0) q_mul = 0; if (q_mul > 255) q_mul = 255;

                float r_se = ((float)q_mul - S4_SE_ZERO_POINT) * S4_SE_SCALE;
                float r_sc = ((float)sc_plane[r * 14 + c] - S4_SC_OUT_ZERO_POINT) * S4_SC_OUT_SCALE;
                float r_sum = r_se + r_sc;
                if (r_sum < 0.0f) r_sum = 0.0f;
                int32_t q_out = (int32_t)roundf(r_sum / S4_RELU_SCALE) + S4_RELU_ZERO_POINT;
                if (q_out < 0) q_out = 0; if (q_out > 255) q_out = 255;
                sc_plane[r * 14 + c] = (uint8_t)q_out;
            }
        }
    }

    free_channels(act128_c1, 128);

    // ==========================================
    // GAP (128 channels x 14x14) & CLASSIFIER
    // ==========================================
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

    if (out_prob) *out_prob = prob;
    return logit;
}

int main() {
    printf("Running C++ Engine on sample_tb_img...\n");
    float tb_prob = 0.0f;
    float tb_logit = tinytb_predict(sample_tb_img, &tb_prob);
    printf("TB Sample Result: Logit = %f, Prob = %f (Ref Logit = %f, Ref Prob = %f)\n",
           tb_logit, tb_prob, REF_TB_LOGIT, REF_TB_PROB);

    printf("Running C++ Engine on sample_norm_img...\n");
    float norm_prob = 0.0f;
    float norm_logit = tinytb_predict(sample_norm_img, &norm_prob);
    printf("Norm Sample Result: Logit = %f, Prob = %f (Ref Logit = %f, Ref Prob = %f)\n",
           norm_logit, norm_prob, REF_NORM_LOGIT, REF_NORM_PROB);

    return 0;
}
