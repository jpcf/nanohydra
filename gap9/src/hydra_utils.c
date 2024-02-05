#include "../include/hydra.h"

uint16_t pow2(int d, int val) {
    uint16_t out=1;
    for(int i=0; i < val; i++)
        out *=2;
    return out;
}

uint16_t padding_len(uint16_t lenW, uint16_t dilation) {
    return (lenW / 2) * (dilation+1) + 1;
}

uint16_t generate_dilation_val(uint16_t dil_idx) {
    return dil_idx == 0 ? dil_idx : pow2(2, dil_idx-1);
}