#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define TRAIN_IMAGE "archive/train-images.idx3-ubyte"
#define TRAIN_LABEL "archive/train-labels.idx1-ubyte"
#define TEST_IMAGE  "archive/t10k-images.idx3-ubyte"
#define TEST_LABEL  "archive/t10k-labels.idx1-ubyte"


float** read_mnist_images(const char* filename, int num_images) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Unable to open %s\n", filename);
        return NULL;
    }

    int magic_number = 0, number_of_images = 0, rows = 0, cols = 0;
    fread(&magic_number, sizeof(int), 1, file);
    fread(&number_of_images, sizeof(int), 1, file);
    fread(&rows, sizeof(int), 1, file);
    fread(&cols, sizeof(int), 1, file);

    magic_number = __builtin_bswap32(magic_number);
    number_of_images = __builtin_bswap32(number_of_images);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    float** images = malloc(num_images * sizeof(float*));
    unsigned char* buffer = malloc(rows * cols);
    for (int i = 0; i < num_images; ++i) {
        fread(buffer, sizeof(unsigned char), rows * cols, file);
        images[i] = malloc(rows * cols * sizeof(float));
        for (int j = 0; j < rows * cols; ++j)
            images[i][j] = buffer[j] / 255.0f;
    }
    free(buffer);
    fclose(file);
    return images;
}

unsigned char* read_mnist_labels(const char* filename, int num_labels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Unable to open %s\n", filename);
        return NULL;
    }
    int magic = 0, num = 0;
    fread(&magic, sizeof(int), 1, file);
    fread(&num, sizeof(int), 1, file);
    magic = __builtin_bswap32(magic);
    num = __builtin_bswap32(num);

    unsigned char* labels = malloc(num_labels);
    fread(labels, sizeof(unsigned char), num_labels, file);
    fclose(file);
    return labels;
}

typedef struct{
    int input_size, output_size;
    float **weights;
    float *bias;
    float *outputs;
    float *inputs;
    float *deltas;
}layer;

layer create_layer(int input_size,int output_size){
    layer l;
    l.input_size = input_size;
    l.output_size = output_size;
    l.weights = malloc(output_size*sizeof(float*));
    for (int i = 0 ; i < output_size ; i++){
        l.weights[i] = malloc(input_size*sizeof(float));
        for (int j = 0 ; j < input_size ;j++){
            l.weights[i][j] = ((float)rand()/ RAND_MAX - 0.5f)*0.1f;
        }
    }
    l.bias = calloc(output_size,sizeof(float));
    l.outputs = calloc(output_size,sizeof(float));
    l.inputs = calloc(input_size,sizeof(float));
    l.deltas = calloc(output_size,sizeof(float));
    return l;
}

void forward_input_to_layer1(layer *l1,float *input){
    memcpy(l1->inputs,input,l1->input_size*sizeof(float));
    for(int i = 0 ; i<l1->output_size ; i++){
        float z = l1->bias[i];
        for(int j = 0 ;j < l1->input_size ; j++){
            z += l1->weights[i][j]*input[j];
        }
        l1->outputs[i] =fmaxf(0.0f,z);
    }
}

void forward_layer1_to_layer2(layer *l2, layer *l1){
    memcpy(l2->inputs,l1->outputs,l2->input_size*sizeof(float));
    for(int i = 0 ; i < l2->output_size; i++){
        float z = l2->bias[i];
        for(int j = 0 ; j < l2->input_size; j++){
            z += l2->weights[i][j]*l1->outputs[j];
        }
        l2->outputs[i] = fmaxf(0.0f,z);
    }
}

void forward_layer2_to_output(layer *l3,layer *l2){
    memcpy(l3->inputs,l2->outputs,l3->input_size*sizeof(float));
    for(int i = 0; i<l3->output_size; i++){
        float z = l3->bias[i];
        for(int j = 0; j < l3->input_size; j++){
            z += l3->weights[i][j]*l2->outputs[j];
        }
        l3->outputs[i] = z; // Lớp đầu ra không sử dụng ReLU
    }
}

void softmax(float *input, int size){
    float max = input[0];
    for(int i = 0; i < size; i++){
        if(max < input[i]) max = input[i];
    }
    float sum = 0.0f;
    for(int i = 0; i < size; i++){
        input[i] = expf(input[i]-max);
        sum += input[i];
    }
    for(int i = 0; i < size; i++){
        input[i] /=sum;
    }
}

float cross_entropy(float *probs,int label){
    return -logf(probs[label]);
}

void backward_outputs_to_l2(layer *l3, float *grad_output,float *grad_l2,float learning_rate){
    for(int i = 0; i < l3->output_size; i++){
        float grad = grad_output[i];
        l3->deltas[i] = grad;
        l3->bias[i] -= learning_rate * grad;
        for(int j = 0; j < l3->input_size; j++){
            l3->weights[i][j] -= learning_rate * grad * l3->inputs[j];
            grad_l2[j] += grad * l3->weights[i][j];
        }
    }
}

void backward_l2_to_l1(layer *l2,float *grad_l2,float *grad_l1,float learning_rate){
    for(int i = 0; i < l2->output_size; i++){
        float grad = grad_l2[i]*(l2->outputs[i]>0? 1.0f : 0.0f);
        l2->deltas[i]= grad;
        l2->bias[i] -= learning_rate*grad;
        for(int j = 0 ; j < l2->input_size; j++){
            l2->weights[i][j] -= learning_rate*grad*l2->inputs[j];
            grad_l1[j] += grad*l2->weights[i][j];
        }
    }
}

void backward_l1_to_input(layer *l1, float *grad_l1, float learning_rate){
    for(int i = 0; i < l1->output_size; i++){
        float grad = grad_l1[i]*(l1->outputs[i]>0? 1.0f : 0.0f);
        l1->deltas[i] = grad;
        l1->bias[i] -= learning_rate*grad;
        for(int j = 0; j < l1->input_size ; j++){
            l1->weights[i][j] -= learning_rate*grad*l1->inputs[j];
        }
    }
}

int main(){
    srand(time(NULL));

    float **train_images = read_mnist_images(TRAIN_IMAGE,60000);
    unsigned char *train_labels = read_mnist_labels(TRAIN_LABEL, 60000);
    float **test_images = read_mnist_images(TEST_IMAGE,10000);
    unsigned char *test_labels = read_mnist_labels(TEST_LABEL,10000);

    if (!train_images || !train_labels || !test_images || !test_labels) {
        printf("Failed to load MNIST data.\n");
        return 1;
    }

    layer l1 = create_layer(784,128);
    layer l2 = create_layer(128,64);
    layer l3 = create_layer(64,10);

    int epochs = 5;
    float learning_rate = 0.0005f;

    for(int epoch = 0; epoch < epochs ;epoch++){
        float loss = 0;
        int correct = 0;

        for(int i = 0; i < 60000; i++){
            forward_input_to_layer1(&l1, train_images[i]);
            forward_layer1_to_layer2(&l2, &l1);
            forward_layer2_to_output(&l3, &l2);
            softmax(l3.outputs, 10);

            loss += cross_entropy(l3.outputs,train_labels[i]);

            int predict = 0;
            for(int j = 1; j < 10; j++){
                if(l3.outputs[j]>l3.outputs[predict]) predict = j;
            }
            if(predict == train_labels[i]) correct++;

            float grad_l3[10] = {0};
            float grad_l2[64] = {0};
            float grad_l1[128] = {0};
            for(int j = 0; j < 10 ; j++){
                grad_l3[j] = l3.outputs[j] - (j == train_labels[i]? 1.0f : 0.0f);
            }

            backward_outputs_to_l2(&l3, grad_l3, grad_l2, learning_rate);
            backward_l2_to_l1(&l2, grad_l2,grad_l1, learning_rate);
            backward_l1_to_input(&l1, grad_l1, learning_rate);
        }
        printf("Epoch %d - Loss: %.4f - Accuracy: %.2f%%\n", epoch + 1, loss / 60000, 100.0f * correct / 60000);
    } 

    // Ghi nhớ giải phóng bộ nhớ đã cấp phát
    for(int i = 0; i < 60000; ++i) free(train_images[i]);
    free(train_images);
    free(train_labels);
    for(int i = 0; i < 10000; ++i) free(test_images[i]);
    free(test_images);
    free(test_labels);
    // Cần giải phóng bộ nhớ cho các layer
    // ...
    return 0;
}