
void hydra_forward(int16_t *inX, 
                   int16_t *inW, 
                   int16_t *featVec, 
                   uint16_t lenX, uint16_t lenW, 
                   uint16_t H,    uint16_t K, 
                   uint8_t N_dil, uint8_t  N_diff, 
                   uint8_t N_chan, uint8_t N_feats);