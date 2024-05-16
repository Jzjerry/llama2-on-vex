
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "sim_stdlib.h"
#include "bitnet.h"


typedef struct {
    int n_layer;
    int *layer_n, *layer_m;
} Config;

typedef struct{
    float s;      // scaling factors
    uint8_t* q_4;  // 4 x 2-bit quantized weights
} BitNetWeight;

Config config;
BitNetWeight* bitnet_weight;

// Model File
extern char _binary_bin_model_bin_start;

void read_checkpoint() {

    char* ptr = &_binary_bin_model_bin_start; 
    if (ptr == NULL) { printf("[Error] Couldn't open file\n"); exit(EXIT_FAILURE); }

    uint32_t magic_number;
    memcpy(&magic_number, (uint32_t*)ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    if (magic_number != 0x616b3432) { printf("[Error] Bad magic number\n"); exit(EXIT_FAILURE); }
    
    memcpy(&config.n_layer, (int*)ptr, sizeof(int));
    ptr += sizeof(int);

    config.layer_n = (int*)malloc(config.n_layer * sizeof(int));
    config.layer_m = (int*)malloc(config.n_layer * sizeof(int));

    bitnet_weight = (BitNetWeight*)malloc(config.n_layer * sizeof(BitNetWeight));
    printf("n_layer: %d\n", config.n_layer);
    for (int i = 0; i < config.n_layer; i++) {
        
        memcpy(&config.layer_n[i], (int*)ptr, sizeof(int));
        ptr += sizeof(int);

        memcpy(&config.layer_m[i], (int*)ptr, sizeof(int));
        ptr += sizeof(int);

        memcpy(&bitnet_weight[i].s, (float*)ptr, sizeof(float));
        ptr += sizeof(float);
        
        if ((config.layer_n[i] * config.layer_m[i]) % 4 != 0) { exit(EXIT_FAILURE); }
        size_t weight_size = config.layer_n[i] * config.layer_m[i] / 4;

        bitnet_weight[i].q_4 = (uint8_t*)ptr;
        ptr += weight_size;
    }
}

size_t find_largest_act_size(Config* config) {
    size_t largest_act_size = config->layer_n[0];
    for(int i = 0; i < config->n_layer; i++) {
        if (config->layer_m[i] > largest_act_size)
            largest_act_size = config->layer_m[i];
    }
    return largest_act_size;
}

int main() {
    printf("Reading checkpoint...\n");
    read_checkpoint();
    for(int i = 0; i < config.n_layer; i++) {
        printf("Layer %d: %d x %d \n", i, config.layer_n[i], config.layer_m[i]);
    }

    size_t largest_act_size = find_largest_act_size(&config);
    float *input = (float*)malloc(largest_act_size * sizeof(float));
    float *output = (float*)malloc(largest_act_size * sizeof(float));
    float *temp;
    for (int i=0; i < 256; i++){
    	input[i] = 1.0;
    }   
    input[0] = 1.5;

    int n_layer = config.n_layer;
    int result = -1;

    long start = time();

    for(int i = 0; i < n_layer; i++){
        int n = config.layer_n[i];
        int d = config.layer_m[i];
        forward(input, output, bitnet_weight[i].q_4, bitnet_weight[i].s, n, d);
        if (i < n_layer - 1) {ReLU(output, d);}
        else {
            result = argmax(output, d);
        }
        temp = input;
        input = output;
        output = temp;
    }
    long end = time() - start;
    free(input);
    free(output);
    printf("Result: %d\n", result);
    printf("Total Time: %d\n", end);

    // Profiler Output
    printf("RMSNorm Time: %d\n", rmsnorm_time);
    printf("ActQuantize Time: %d\n", act_quantize_time);
    printf("MatMul Time: %d\n", matmul_time);
    printf("Dequantize Time: %d\n", dequantize_time);
    printf("AddSub Time: %d\n", addsub4_time);

    // memset(test_input, 1, 256);
    // int8_t output[256];
    return 0;
}
