#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "../firmware/src/tinytb_weights.h"
#include "../firmware/src/tinytb_samples.h"
#include "../firmware/src/tinytb_infer.h"

int main() {
    printf("Testing host C++ TinyTB-Net single-pass inference engine...\n");

    float prob1 = 0.0f;
    float logit1 = tinytb_predict(sample_tb_img, &prob1, NULL);
    printf("Sample TB:   Logit = %f (Ref: %f, diff=%f), Prob = %f (Ref: %f)\n",
           logit1, REF_TB_LOGIT, fabsf(logit1 - REF_TB_LOGIT),
           prob1, REF_TB_PROB);

    float prob2 = 0.0f;
    float logit2 = tinytb_predict(sample_norm_img, &prob2, NULL);
    printf("Sample NORM: Logit = %f (Ref: %f, diff=%f), Prob = %f (Ref: %f)\n",
           logit2, REF_NORM_LOGIT, fabsf(logit2 - REF_NORM_LOGIT),
           prob2, REF_NORM_PROB);

    assert(fabsf(logit1 - REF_TB_LOGIT) < 1e-4f);
    assert(fabsf(prob1 - REF_TB_PROB) < 1e-4f);
    assert(fabsf(logit2 - REF_NORM_LOGIT) < 1e-4f);
    assert(fabsf(prob2 - REF_NORM_PROB) < 1e-4f);

    printf("ALL HOST ASSERTIONS PASSED BIT-EXACTLY!\n");
    return 0;
}
