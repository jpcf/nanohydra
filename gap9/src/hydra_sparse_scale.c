

void hydra_sparse_scale(int16_t featVec, float16_t featMean, float16_t featStd, uint16_t lenFeatVec) {

    float16_t temp;

    for(f=0; f < lenFeatVec; f++) {

        // Under-clip to zero, cast to float
        temp = (float)(featVec[f] < 0 ? 0 : featVec[f])

        // Square root the value
        temp = sqrt(temp);

        // Normalize the value
        temp = (temp - featMean[f]) / featStd[f]

        // Conver the value back to int16_t, in Qn.m
        featVec[f] = (int16_t)temp << QM_SCALER;
    }
}