#include <stdint.h>
#include <math.h>

// Profiler Variables
long addsub4_time = 0;
long matmul_time = 0;
long rmsnorm_time = 0;
long act_quantize_time = 0;
long dequantize_time = 0;

int8_t unpack_table[4] = {0, 1, 0, -1};

int32_t addsub4(int8_t* a, uint8_t w){
    long start = time();

    int32_t sum = 0;
    for(int i = 0; i < 4; i++){
        // Multiply Implementation
        // sum += unpack_table[(w >> ((3-i)*2)) & 0x03] * a[i];

        // Mux Implementation
        uint8_t w_shift = (w >> (6-(i<<1))) & 0x03;
        sum += w_shift == 1 ? a[i] : (w_shift == 3 ? -a[i] : 0);
    }

    addsub4_time += time() - start;
    return sum;
}

void matmul(int8_t *input, int32_t *output, uint8_t *weight, int n, int d){
    long start = time();

    for (int i=0; i < d; i++){
        output[i] = 0;
        for(int j = 0; j < n; j += 4){
            int8_t temp[4];
            for (int k = 0; k < 4; k++){
                if (j + k < n){
                    temp[k] = *(input + j + k);
                }
                else{
                    temp[k] = 0;
                }
            }
            int offset = (i*n + j) % 4;
            int base = (i*n + j) >> 2;
            if (offset == 0){
                output[i] += addsub4(temp, weight[base]);
            }
            else{
                // Align the weight in 4*2 bits
                uint8_t temp_w_u = weight[base] << (offset*2);
                uint8_t temp_w_l = weight[base + 1] >> ((4-offset)*2);
                uint8_t temp_weight = temp_w_u | temp_w_l;
                output[i] += addsub4(temp, temp_weight);
            }
        }
    }

    matmul_time += time() - start;
}

void dequantize(int32_t *a, float *af, float s, int d){
    long start = time();

    for(int i = 0; i < d; i++){
        af[i] = a[i] * s;
    }

    dequantize_time += time() - start;
}

void rmsnorm(float *a, int n){
    long start = time();

    float scale = 0;
    for (int i = 0; i < n; i++){
        scale += a[i]*a[i];
    }
    scale /= n;
    scale = 1.0f / sqrtf(scale);
    scale /= n;
    for (int i = 0; i < n; i++){
        a[i] *= scale;
    }

    rmsnorm_time += time() - start;
}

float act_scale(float *a, int n){

    float max = -1;
    for (int i = 0; i < n; i++){
        if (fabs(a[i]) > max){
            max = fabs(a[i]);
        }
    }
    return max/127.0;
}

void act_quantize(float *a, int8_t *qa, float s, int n){
    long start = time();

    float scale = 1.0/s;
    for (int i = 0; i < n; i++){
        qa[i] = (int8_t)round(a[i]*scale);
    }

    act_quantize_time += time() - start;
}

void forward(float *a, float *o, uint8_t *w, float s, int n, int d){
    int8_t *qa = (int8_t*)malloc(n * sizeof(int8_t));
    int32_t *qo = (int32_t*)malloc(d * sizeof(int32_t));

    rmsnorm(a, n);
    float a_s = act_scale(a, n);
    act_quantize(a, qa, a_s, n);

    matmul(qa, qo, w, n, d);
    dequantize(qo, o, s*a_s, d);

    free(qa);
    free(qo);
}

void ReLU(float* a, int n){
    for (int i = 0; i < n; i++){
        a[i] = a[i] > 0 ? a[i] : 0;
    }
}

int argmax(float *a, int n){
    float max = -1;
    int idx = -1;
    for (int i = 0; i < n; i++){
        if (a[i] > max){
            max = a[i];
            idx = i;
        }
    }
    return idx;
}