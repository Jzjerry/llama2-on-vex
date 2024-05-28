#include <stdint.h>


// BitNet SIMD Types
// Activation
typedef uint32_t int8x4_t;

// Quantized Weight
typedef uint8_t int2x4_t;
typedef uint16_t int2x8_t;
typedef uint32_t int2x16_t;
typedef uint8_t int1x8_t;
typedef uint16_t int1x16_t;
typedef uint32_t int1x32_t;

// Reference C Implementation
int32_t addsub4(int8_t* a, int2x4_t w){
    int32_t sum = 0;
    for(int i = 0; i < 4; i++){
        uint8_t w_shift = (w >> (6-(i<<1))) & 0b11;
        sum += w_shift == 1 ? a[i] : (w_shift == 3 ? -a[i] : 0);
    }
    return sum;
}

int32_t addsub4x1b(int8_t* a, int1x8_t w){

    int32_t sum = 0;
    for(int i = 0; i < 4; i++){
        uint8_t w_shift = (w >> (3-i)) & 0b1;
        sum += w_shift == 0 ? a[i] : -a[i];
    }
    return sum;
}

int32_t addsub8(int8_t* a, int2x4_t* w){
    int32_t sum = 0;
    for(int i = 0; i < 8; i++){
        uint8_t w_shift = (w[i>>2] >> (6-((i&0b11)<<1))) & 0b11;
        sum += w_shift == 1 ? a[i] : (w_shift == 3 ? -a[i] : 0);
    }
    return sum;
}

int32_t addsub8x1b(int8_t* a, int1x8_t w){

    int32_t sum = 0;
    for(int i = 0; i < 8; i++){
        uint8_t w_shift = (w >> (7-i)) & 0b1;
        sum += w_shift == 0 ? a[i] : -a[i];
    }
    return sum;
}

int32_t addsub16(int8_t* a, int2x4_t *w){
    int32_t sum = 0;
    return sum;
}

int32_t addsub16x1b(int8_t* a, int1x8_t* w){
    int32_t sum = 0;
    for(int i = 0; i < 2; i++){
        sum += addsub8x1b(a + (i << 3), w[i]);
    }
    return sum;
}