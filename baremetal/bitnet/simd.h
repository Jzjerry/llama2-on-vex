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

// Custom Instruction Calls
extern int32_t __bitnetadd4(int8x4_t a, int2x4_t w);
extern int32_t __bitnetadd8(int8x4_t a1, int8x4_t a2, int2x8_t w);
#if BITNET_QUANT == 2
extern int32_t __bitnetadd16(int8_t *a, int1x16_t w);
extern int32_t __bitnetadd32(int8_t *a, int1x32_t w, int1x32_t dummy);
extern int32_t __bitnetadd64(int8_t *a, int1x32_t w1, int1x32_t w2);
#else
extern int32_t __bitnetadd16(int8_t *a, int2x16_t w);
extern int32_t __bitnetadd32(int8_t *a, int2x16_t w1, int2x16_t w2);
#endif


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
    // TODO: Implement 16-bit Quantization
    return sum;
}

int32_t addsub16x1b(int8_t* a, int1x8_t* w){
    int32_t sum = 0;
    for(int i = 0; i < 2; i++){
        sum += addsub8x1b(a + (i << 3), w[i]);
    }
    return sum;
}

int32_t addsub32x1b(int8_t* a, int1x8_t* w){
    int32_t sum = 0;
    for(int i = 0; i < 4; i++){
        sum += addsub8x1b(a + (i << 3), w[i]);
    }
    return sum;
}

int32_t addsub64x1b(int8_t* a, int1x8_t* w){
    int32_t sum = 0;
    for(int i = 0; i < 8; i++){
        sum += addsub8x1b(a + (i << 3), w[i]);
    }
    return sum;
}
// BitNet Test
#if BITNET_QUANT == 2
#if USE_SIMD == 4
void test(){
    int8_t a[] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint8_t w = 0b00001000;
    int32_t c = __bitnetadd4(*(uint32_t*)a, w>>4) + __bitnetadd4(*(uint32_t*)(a + 4), w);
    int32_t c_ref = addsub4x1b(a, w>>4) + addsub4x1b(a + 4, w);
    printf("BitNet Add: %d\n", c);
    printf("BitNet Add Ref: %d\n", c_ref);
}
#elif USE_SIMD == 8
void test(){
    int8_t a[] = {1,2,3,4,5,6,7,8};
    uint8_t w = 0b00001000;
    int32_t c = __bitnetadd8(*(uint32_t*)(&a[0]), *(uint32_t*)(&a[4]), w);
    int32_t c_ref = addsub8x1b(a, w);
    printf("BitNet Add: %d\n", c);
    printf("BitNet Add Ref: %d\n", c_ref);
}
#elif USE_SIMD == 16
void test(){
    int8_t a[] = {127,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85};
    uint8_t w[] = {0, 0b11111000};
    int32_t c = __bitnetadd16(a, *(uint16_t*)w);
    int32_t c_ref = addsub16x1b(a, w);
    printf("Weight: %d\n", *(uint16_t*)w);
    printf("BitNet Add: %d\n", c);
    printf("BitNet Add Ref: %d\n", c_ref);
}
#elif USE_SIMD == 32
void test(){
    int8_t a[] = {
        1,2,3,4,5,6,7,8,
        2,2,3,4,5,6,7,8,
        3,2,3,4,5,6,7,8,
        4,2,3,4,5,6,7,8 };
    uint8_t w[] = {0, 0b00000001, 0b10001000, 0b00001001};
    int32_t c = __bitnetadd32(a, *(uint32_t*)w, 0);
    int32_t c_ref = addsub32x1b(a, w);
    printf("Weight: %d\n", *(uint32_t*)w);
    printf("BitNet Add: %d\n", c);
    printf("BitNet Add Ref: %d\n", c_ref);
}
#elif USE_SIMD == 64
void test(){
    int8_t a[] = {
        1,2,3,4,5,6,7,8,
        6,2,3,4,5,6,8,3,
        2,4,6,8,9,0,1,2,
        3,4,6,8,9,0,1,2,
        5,6,7,8,9,1,2,3,
        4,6,5,2,3,4,5,6,
        9,2,1,3,4,5,6,7,
        8,3,4,6,1,2,3,5 };
    uint8_t w[] = {
        0, 0b00000001, 0b10001000, 0b00001001,
        1, 0b10001000, 0b00000001, 0b00001001};
    uint32_t w1 = w[0] | (w[1] << 8) | (w[2] << 16) | (w[3] << 24);
    uint32_t w2 = w[4] | (w[5] << 8) | (w[6] << 16) | (w[7] << 24);
    int32_t c = __bitnetadd64(a, *(uint32_t*)w, *(uint32_t*)(w+4));
    int32_t c_ref = addsub64x1b(a, w);
    printf("Weight: %d %d\n", w1, w2);
    printf("BitNet Add: %d\n", c);
    printf("BitNet Add Ref: %d\n", c_ref);
}
#endif
#else
#if USE_SIMD == 4
void test(){
    int8_t a[] = {1, 2, 3, 4};
    uint8_t w = 0b11010101;
    int32_t c = __bitnetadd4(*(uint32_t*)a, w);
    int32_t c_ref = addsub4(a, w);
    printf("BitNet Add: %d\n", c);
    printf("BitNet Add Ref: %d\n", c_ref);
}
#elif USE_SIMD == 8
void test(){
    int8_t a[] = {1,2,3,4,5,6,7,8};
    uint8_t w[] = {0b11010101, 0b01010101};
    int32_t c = __bitnetadd8(*(uint32_t*)(&a[0]), *(uint32_t*)(&a[4]), *(uint16_t*)w);
    int32_t c_ref = addsub8(a, w);
    printf("BitNet Add: %d\n", c);
    printf("BitNet Add Ref: %d\n", c_ref);
}
#elif USE_SIMD == 16
void test(){
    int8_t a[] = {1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8};
    uint8_t w[] = {0b11010101, 0b01010101, 0b11010101, 0b01010101};
    printf("BitNet Add: %d\n", c);
    printf("BitNet Add Ref: %d\n", c_ref);
}
#endif
#endif