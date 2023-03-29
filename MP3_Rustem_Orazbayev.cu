


#include "slenet_params.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INSIZE 28
#define NUM_TEST_IMAGES 10000
#define X1 6 // Number of blocks for kernel_conv_filter
#define Y1 24 // Number of threads per block for kernel_conv_filter
#define X2 6 // Number of blocks for kernel_conv_bias
#define Y2 24 // Number of threads per block for kernel_conv_bias
#define X3 6 // Number of blocks for kernel_conv_sigmoid
#define Y3 24 // Number of threads per block for kernel_conv_sigmoid

typedef struct mnist_data{
    double data[INSIZE][INSIZE];
    unsigned int label;
} mnist_data;



static unsigned int mnist_bin_to_int(char *tmp) {
    unsigned int val = 0;
    for (int i = 0; i < 4; i++) {
        val <<= 8; // bit shift
        val |= (unsigned char)tmp[i];// bitwise or
    }
    return val;
}


static int mnist_load(const char *image_filename, const char *label_filename, mnist_data **data_set, unsigned int *count) {
    // 1. opens image and label files of the test
    FILE *image_file = fopen(image_filename, "rb");
    FILE *label_file = fopen(label_filename, "rb");
    if (label_file == NULL || image_file == NULL) {
        printf("Failed to open image file or label file\n");
        return -1;
    }

    // 2. check the file formats of the test files

    // 2-1. the magic numbers of image and label files
    char image_magic_number[4]; 
    char label_magic_number[4];
    fread(&image_magic_number, sizeof(char), 4, image_file);
    fread(&label_magic_number, sizeof(char), 4, label_file);
    if (mnist_bin_to_int(image_magic_number) != 2051 || mnist_bin_to_int(label_magic_number) != 2049) {
        printf("Invalid magic numbers in files\n");
        fclose(image_file);
        fclose(label_file);
        return -1;
    }else{
      printf("image magic number= 2051\n");
      printf("label magic number = 2049\n");
    }


    // 2-2. numbers of images and labels
    char num_images[4];
    char num_labels[4];
    fread(&num_images, sizeof(char), 4, image_file);
    fread(&num_labels, sizeof(char), 4, label_file);

    if (mnist_bin_to_int(num_images) != NUM_TEST_IMAGES || mnist_bin_to_int(num_labels) != NUM_TEST_IMAGES) {
        printf("Invalid number of images/labels in files\n");
        fclose(image_file);
        fclose(label_file);
        return -1;
    }else{
      printf("image total number = 10000\n");
      printf("label total number = 10000\n");
    }

    // 2-4. check the number of rows and columns
    // Check the number of rows and columns
    char num_rows[4];
    char num_cols[4];
    fread(&num_rows, sizeof(char), 4, image_file);
    fread(&num_cols, sizeof(char), 4, image_file);
    if (mnist_bin_to_int(num_rows) != INSIZE || mnist_bin_to_int(num_cols) != INSIZE) {
        printf("Invalid image size\n");
        fclose(image_file);
        fclose(label_file);
        return -1;
    }else{
      printf("rows = 28, cols = 28\n");
    }

    // Allocate memory for the data set
    mnist_data *data = (mnist_data*)malloc(NUM_TEST_IMAGES * sizeof(mnist_data));

    int counter = 0;

    // 3. loads image data as double type (from 0.0 to 1.0 dividing unsigned char values by 255.0) 
    unsigned char image[INSIZE][INSIZE];
    for (int i = 0; i < NUM_TEST_IMAGES; i++) {
        fread(image, sizeof(image), 1, image_file);
        data[i].label = fgetc(label_file);
        counter++;
        for (int j = 0; j < INSIZE; j++) {
            for (int k = 0; k < INSIZE; k++) {
                data[i].data[j][k] = (double)image[j][k] / 255.0;
            }
        }
    }

    // 4. closes opened files
    fclose(image_file);
    fclose(label_file);

    // Set the output variables
    *data_set = data;
    *count = counter;

    return 0;
}



    // CUDA kernel functions for filtering, bias, and sigmoid activation

    __global__ void kernel_conv_filter(float (*input)[28], float (*pre_output)[24][24], float (*weight)[5][5]) {
            int t = threadIdx.x + blockIdx.x * blockDim.x;
            int i = t / (24 * 24);
            int j = (t / 24) % 24;
            int k = t % 24;

            if (t < 6 * 24 * 24) {
                float sum = 0.0f;
                for (int m = 0; m < 5; m++) {
                    for (int n = 0; n < 5; n++) {
                        sum += weight[i][m][n] * input[j + m][k + n];
                    }
                }
                pre_output[i][j][k] = sum;
            }
    }


    __global__ void kernel_conv_bias(float (*pre_output)[24][24], float *bias) {
        int t = threadIdx.x + blockIdx.x * blockDim.x;
        int i = t / (24 * 24);
        int j = (t / 24) % 24;
        int k = t % 24;

        if (t < 6 * 24 * 24) {
            pre_output[i][j][k] += bias[i];
        }
    }

    __global__ void kernel_conv_sigmoid(float pre_output[6][24][24], float output[6][24][24]) {
        int t = threadIdx.x + blockIdx.x * blockDim.x;
        int i = t / (24 * 24);
        int j = (t / 24) % 24;
        int k = t % 24;
        if (t < 6 * 24 * 24) {
           output[i][j][k] = 1.0f / (1.0f + expf(-pre_output[i][j][k]));
        }
    }


    __global__ void kernel_ss1_filter(float input[6][24][24], float output[6][6][6], float weight[4][4]) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        int col = blockIdx.y * blockDim.y + threadIdx.y;
        int feature = blockIdx.z;
        
        if (row < 6 && col < 6) {
            float sum = 0.0;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    sum += input[feature][row*4+i][col*4+j] * weight[i][j];
                }
            }
            output[feature][row][col] = sum;
        }
    }

    __global__ void kernel_ss1_bias(float output[6][6][6]) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        int col = blockIdx.y * blockDim.y + threadIdx.y;
        int feature = blockIdx.z;
        if (row < 6 && col < 6) {
            output[feature][row][col] = output[feature][row][col] + 0.827946;
        }
    }

    __global__ void kernel_ss1_sigmoid(float output[6][6][6]) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        int col = blockIdx.y * blockDim.y + threadIdx.y;
        int feature = blockIdx.z;
        if (row < 6 && col < 6) {
          output[feature][row][col] = 1.0f / (1.0f + expf(-output[feature][row][col]));
        }
    }

__global__ void kernel_fc1(float input[6][6][6], float pre_output[10], float weight[10][6][6][6]) {
    // Calculate global thread index
    int t = threadIdx.x + blockIdx.x * blockDim.x ;
    if(t<10){
          // Compute dot product of input and weight for current output index
    float dot_product = 0.0f;
    for (int x = 0; x < 6; x++) {
        for (int y = 0; y < 6; y++) {
            for (int z = 0; z < 6; z++) {
                dot_product += input[x][y][z] * weight[t][x][y][z];
            }
        }
    }
    // Store dot product in pre_output array
    pre_output[t] = dot_product;
    }
}


__global__ void kernel_fc1_bias(float pre_output[10], float bias[10]) {
    int t = threadIdx.x +  blockIdx.x * blockDim.x;
    if (t < 10) {
        pre_output[t] += bias[t];
    }
}

__global__ void kernel_fc1_sigmoid(float pre_output[10], float output[10]) {
    int t = threadIdx.x + blockIdx.x * blockDim.x;
    if (t < 10) {
        output[t] = 1.0f / (1.0f + expf(-pre_output[t]));
    }
}


class Layer {
public:
    int M, N, O;
    float pre_output[6][24][24], output[6][24][24];
    float weight[6][5][5], bias[24];

    float ssweight[4][4];
    float ssoutput[6][6][6];

    float (*dpre_output)[24][24], (*doutput)[24][24];
    float (*dweight)[5][5], (*dbias);
    float (*dssweight)[4], (*dssoutput)[6][6];

    float fc_output[10]; 
    float fc_weight[10][6][6][6]; 
    float fc_bias[10];

    float (*dfc_pre_output);
    float (*dfc_output);
    float (*dfc_weight)[6][6][6]; 
    float (*dfc_bias);



    Layer(int M, int N, int O) {
        this->M = M;
        this->N = N;
        this->O = O;
/*
        this->weight = c1_weight;
        this->bias = c1_bias;

        this->ssweight = s2_weight;

        this->fc_weight = f3_weight;
        this->fc_bias = f3_bias;
*/
        memcpy(weight, c1_weight, sizeof(c1_weight));
        memcpy(bias, c1_bias, sizeof(c1_bias));
        memcpy(ssweight, s2_weight, sizeof(s2_weight));
        memcpy(fc_weight, f3_weight, sizeof(f3_weight));
        memcpy(fc_bias, f3_bias, sizeof(f3_bias));


        // Allocate memory on the GPU
        cudaMalloc(&dpre_output, 6 * O * O * sizeof(float));
        cudaMalloc(&doutput, 6 * O * O * sizeof(float));
        cudaMalloc(&dweight, M * 5 * 5 * sizeof(float));
        cudaMalloc(&dbias, O * sizeof(float));
        cudaMalloc(&dssweight, 4 * 4 *sizeof(float));
        cudaMalloc(&dssoutput, 6 * 6 * 6 * sizeof(float));

        cudaMalloc(&dfc_pre_output, 10 * sizeof(float));
        cudaMalloc(&dfc_output, 10 * sizeof(float));
        cudaMalloc(&dfc_weight, 10 * 6 * 6 * 6 * sizeof(float));
        cudaMalloc(&dfc_bias, 10 * sizeof(float));
  }
    ~Layer() {
        // Free memory on the GPU
        cudaFree(dpre_output);
        cudaFree(doutput);
        cudaFree(dweight);
        cudaFree(dbias);
    }
};

static double forward_pass(double data[28][28],Layer layer) {
        float input[28][28];
        // Convert the input data to -1 to simulate test
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                input[i][j] = data[i][j];
            }
        }
        

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); // cudaEventRecord(start, 0); over 10.x

        //CONVOLUTION

        float (*dinput)[28];
        cudaMalloc(&dinput, 28 * 28 * sizeof(float));
        cudaMemcpy(dinput, input, 28 * 28 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(layer.dweight, layer.weight, 6 * 5 * 5 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(layer.dbias, layer.bias, 6 * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemcpy(layer.dssweight, layer.ssweight, 4 * 4 * sizeof(float), cudaMemcpyHostToDevice);

        // Perform the filtering operation
        kernel_conv_filter<<<6, 576>>>(dinput, layer.dpre_output, layer.dweight);
        
        // Add the bias term
        kernel_conv_bias<<<6, 576>>>(layer.dpre_output, layer.dbias);

        // Apply the sigmoid activation function
        kernel_conv_sigmoid<<<6, 576>>>(layer.dpre_output, layer.doutput);


        //SUBSAMPLING

        dim3 blockDim(16, 16);
        dim3 gridDim(2, 2, 6);

        kernel_ss1_filter<<<gridDim, blockDim>>>(layer.doutput, layer.dssoutput, layer.dssweight);

        kernel_ss1_bias<<<gridDim, blockDim>>>(layer.dssoutput);

        kernel_ss1_sigmoid<<<gridDim, blockDim>>>(layer.dssoutput);


        // Call kernel_fc1 function
        cudaMemcpy(layer.dfc_weight, layer.fc_weight, 10 * 6 * 6 * 6 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(layer.dfc_bias, layer.fc_bias, 10 * sizeof(float), cudaMemcpyHostToDevice);


        kernel_fc1<<<1, 10>>>(layer.dssoutput, layer.dfc_pre_output, layer.dfc_weight);
    
        kernel_fc1_bias<<<1, 10>>>(layer.dfc_pre_output, layer.dfc_bias);
      
        kernel_fc1_sigmoid<<<1, 10>>>(layer.dfc_pre_output, layer.dfc_output);

        cudaMemcpy(layer.fc_output, layer.dfc_output, 10 * sizeof(float), cudaMemcpyDeviceToHost);
        
        
        /*
        for (int k = 0; k < 10; k++) {
              printf("%f  .  ", layer.fc_output[k]);

              printf("%d  .  \n", k);
        }
        */
        cudaEventRecord(stop); // cudaEventRecord(stop, 0); over 10.x
        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        return (double)elapsedTime; 
              
}   


double time_taken = 0.0;

int main() {
    const int M = 6;
    const int N = 1;
    const int O = 24;

    int ret, i; 
    mnist_data *test_set; 
    static unsigned int test_cnt;

    // Initialize convolutional layer
    Layer layer(M, N, O);
    

    // 1. load data
    if(ret = mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", &test_set, &test_cnt) != 0){
        printf("An error occured: %d \n", ret);
    } else {
        printf("test_cnt = %d \n", test_cnt); // test_cnt must have the number of test images (i.e., 10K) // copy the trained parameters to GPU device (in here or another layer)
    }
    

    // forward pass
    unsigned int error = 0; 
    unsigned int max = 0; 
    float res[10];
    for (i=0; i<test_cnt; i++){
        time_taken += forward_pass(test_set[i].data, layer);
        cudaMemcpy(res, layer.dfc_output, sizeof(float)*10, cudaMemcpyDeviceToHost); 
        for(int j=0; j<10; j++){
            if (res[max] < res[j]){
                max = j; 
            }
        }    
        if (max != test_set[i].label){
            ++error;
        } 
    }
    printf("Error Rate = %f%% (%d out of 10,000)\n", double(error)/double(test_cnt)*100.0, error); 
    printf("Accuracy = %.3f%% (%d out of 10,000)\n", 100.0 - double(error)/double(test_cnt)*100.0, test_cnt - error);
    printf("Ex time = %f (ms) \n", time_taken); //NOTE: cudaMemcpy operations also should be added



    return 0;

}


/*START_CITE
https://chat.openai.com/chat
END_CITE*/
