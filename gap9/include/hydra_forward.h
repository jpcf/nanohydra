#include <stdlib.h>

void hydra_forward(int16_t  **inX,
                   int16_t  **inX_diff,
                   int16_t ***inW, 
                   int16_t   *featVec, 
                   u_int16_t lenX,  u_int16_t lenW, 
                   u_int16_t H,     u_int16_t K, 
                   u_int8_t N_dil,  u_int8_t  N_diff, 
                   u_int8_t N_chan, u_int8_t  N_feats);