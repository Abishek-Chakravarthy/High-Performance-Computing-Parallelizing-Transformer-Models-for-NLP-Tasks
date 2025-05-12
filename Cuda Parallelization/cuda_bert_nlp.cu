#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand_kernel.h>

using namespace std;


const int SEQ_LEN    = 1024;
const int EMBED_DIM  = 1024;
const int HIDDEN_DIM = 2048;
const int NUM_HEADS  = 8;
const int HEAD_DIM   = EMBED_DIM / NUM_HEADS;
//const int BATCH_SIZE = 8;   // Number of sequences in a batch

CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            cerr << "CUDA error: " << cudaGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


void print_vector(const vector<float> &vec, const string &name) {
    cout << name << ": [ ";
    for (float val : vec)
        cout << fixed << setprecision(4) << val << " ";
    cout << "]\n";
}

void print_matrix(const vector<vector<float>> &matrix, const string &name, int rows = 10, int cols = 10) {
    cout << name << " (Showing first " << min(rows, (int)matrix.size()) 
         << "x" << min(cols, (int)matrix[0].size()) << " values):\n";
    for (int i = 0; i < min(rows, (int)matrix.size()); i++) {
        cout << "[ ";
        for (int j = 0; j < min(cols, (int)matrix[i].size()); j++) {
            cout << fixed << setprecision(6) << matrix[i][j] << " ";
        }
        if (cols < (int)matrix[i].size())
            cout << "... ]\n";
        else
            cout << "]\n";
    }
    cout << "...\n";
}

// Random Matrix Generation
vector<vector<float>> generate_random_matrix(int rows, int cols) {
    vector<vector<float>> matrix(rows, vector<float>(cols));
    random_device rd;
    mt19937 gen(rd());
    float scale = sqrt(2.0f / (rows + cols));
    normal_distribution<float> dist(0.0f, scale);

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            matrix[i][j] = dist(gen);

    return matrix;
}

// CUDA kernel for matrix multiplication
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// CUDA kernel for layer normalization
__global__ void layer_norm_kernel(const float* input, float* output, int rows, int cols, float eps) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        float mean = 0.0f;
        for (int j = 0; j < cols; j++) {
            mean += input[row * cols + j];
        }
        mean /= cols;
        float var = 0.0f;
        for (int j = 0; j < cols; j++) {
            float diff = input[row * cols + j] - mean;
            var += diff * diff;
        }
        var /= cols;
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int j = 0; j < cols; j++) {
            output[row * cols + j] = (input[row * cols + j] - mean) * inv_std;
        }
    }
}

// CUDA kernel for positional encoding
__global__ void positional_encoding_kernel(float* pe, int seq_len, int embed_dim) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (pos < seq_len && dim < embed_dim) {
        float angle = pos / powf(10000, (2 * (dim / 2)) / static_cast<float>(embed_dim));
        if (dim % 2 == 0)
            pe[pos * embed_dim + dim] = sinf(angle);
        else
            pe[pos * embed_dim + dim] = cosf(angle);
    }
}

// CUDA kernel for dropout
__global__ void dropout_kernel(float* input, float* output, float dropout_rate, int size, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        
        float r = curand_uniform(&state);
        if (r < dropout_rate)
            output[idx] = 0.0f;
        else
            output[idx] = input[idx] / (1.0f - dropout_rate);
    }
}

// CUDA kernel for GELU activation
__global__ void gelu_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
    }
}

// CUDA kernel for adding residual connection
__global__ void add_residual_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] += input[idx];
    }
}

// CUDA kernel for softmax
__global__ void softmax_kernel(float* input, float* output, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        float max_val = -INFINITY;
        for (int j = 0; j < cols; j++) {
            max_val = fmaxf(max_val, input[row * cols + j]);
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < cols; j++) {
            output[row * cols + j] = expf(input[row * cols + j] - max_val);
            sum_exp += output[row * cols + j];
        }
        for (int j = 0; j < cols; j++) {
            output[row * cols + j] /= sum_exp;
        }
    }
}

// CUDA kernel for computing attention scores
__global__ void attention_scores_kernel(const float* Q, const float* K, float* scores, int seq_len, int head_dim, float scale) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < seq_len && j < seq_len) {
        float dot = 0.0f;
        for (int k = 0; k < head_dim; k++) {
            dot += Q[i * head_dim + k] * K[j * head_dim + k];
        }
        scores[i * seq_len + j] = dot * scale;
    }
}

// CUDA kernel for adding positional encoding to embeddings
__global__ void add_pos_encoding_kernel(float* embeddings, const float* pos_encoding, int seq_len, int embed_dim) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < seq_len && j < embed_dim) {
        embeddings[i * embed_dim + j] += pos_encoding[i * embed_dim + j];
    }
}

// CUDA kernel for attention head output concat
__global__ void concat_head_outputs_kernel(float** head_outputs, float* output, int seq_len, int head_dim, int num_heads) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int h = blockIdx.z;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < seq_len && j < head_dim && h < num_heads) {
        output[i * (head_dim * num_heads) + h * head_dim + j] = head_outputs[h][i * head_dim + j];
    }
}

// Host function to convert 2D vector to 1D array for CUDA
float* flatten_matrix(const vector<vector<float>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    float* flat = new float[rows * cols];
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            flat[i * cols + j] = matrix[i][j];
        }
    }
    
    return flat;
}

// Host function to convert 1D array back to 2D vector
vector<vector<float>> unflatten_matrix(float* flat, int rows, int cols) {
    vector<vector<float>> matrix(rows, vector<float>(cols));
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = flat[i * cols + j];
        }
    }
    
    return matrix;
}

// CUDA implementation of matmul
vector<vector<float>> cuda_matmul(const vector<vector<float>>& A, const vector<vector<float>>& B, int num_threads) {
    int M = A.size();
    int K = A[0].size();
    int N = B[0].size();
    
    if (K != B.size())
        throw runtime_error("Matrix dimension mismatch in multiplication");

    float* h_A = flatten_matrix(A);
    float* h_B = flatten_matrix(B);
    float* h_C = new float[M * N];
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    

    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 threadsPerBlock(min(num_threads, 32), min(num_threads, 32));
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    vector<vector<float>> C = unflatten_matrix(h_C, M, N);
    
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    
    return C;
}

// CUDA implementation of layer normalization
vector<vector<float>> cuda_layer_norm(const vector<vector<float>>& input, float eps, int num_threads) {
    int rows = input.size();
    int cols = input[0].size();
    
  
    float* h_input = flatten_matrix(input);
    float* h_output = new float[rows * cols];
    
    
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, rows * cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, rows * cols * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, rows * cols * sizeof(float), cudaMemcpyHostToDevice));
    
    int threadsPerBlock = min(num_threads, 1024);
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;

    layer_norm_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, rows, cols, eps);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_output, d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    

    vector<vector<float>> normalized = unflatten_matrix(h_output, rows, cols);

    delete[] h_input;
    delete[] h_output;
    
    return normalized;
}

// CUDA implementation of positional encoding
vector<vector<float>> cuda_positional_encoding(int seq_len, int embed_dim, int num_threads) {
    
    float* h_pe = new float[seq_len * embed_dim]();
    
    
    float* d_pe;
    CUDA_CHECK(cudaMalloc(&d_pe, seq_len * embed_dim * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_pe, 0, seq_len * embed_dim * sizeof(float)));
    
    dim3 threadsPerBlock(min(num_threads, 32), min(num_threads, 32));
    dim3 blocksPerGrid((embed_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (seq_len + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    positional_encoding_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_pe, seq_len, embed_dim);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaMemcpy(h_pe, d_pe, seq_len * embed_dim * sizeof(float), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_pe));
    
    vector<vector<float>> pe = unflatten_matrix(h_pe, seq_len, embed_dim);
    
    delete[] h_pe;
    
    return pe;
}

// CUDA implementation of dropout
vector<vector<float>> cuda_dropout(const vector<vector<float>>& input, float dropout_rate, int num_threads) {
    int rows = input.size();
    int cols = input[0].size();
    int size = rows * cols;
    
    float* h_input = flatten_matrix(input);
    float* h_output = new float[size];
    
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice));
    
    int threadsPerBlock = min(num_threads, 1024);
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    unsigned int seed = static_cast<unsigned int>(time(nullptr));
    dropout_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, dropout_rate, size, seed);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    vector<vector<float>> output = unflatten_matrix(h_output, rows, cols);
    
    delete[] h_input;
    delete[] h_output;
    
    return output;
}

// CUDA implementation of GELU activation
vector<vector<float>> cuda_gelu(const vector<vector<float>>& input, int num_threads) {
    int rows = input.size();
    int cols = input[0].size();
    int size = rows * cols;
    
    float* h_input = flatten_matrix(input);
    float* h_output = new float[size];
    
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice));
    
    int threadsPerBlock = min(num_threads, 1024);
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    gelu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, size);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    vector<vector<float>> output = unflatten_matrix(h_output, rows, cols);
    
    delete[] h_input;
    delete[] h_output;
    
    return output;
}

// CUDA implementation of self-attention
vector<vector<float>> cuda_self_attention(const vector<vector<float>>& input, float dropout_rate, int num_threads) {
    int rows = input.size();
    int cols = input[0].size();
    float scale_factor = 1.0f / sqrt(static_cast<float>(HEAD_DIM));
    
    vector<vector<float>> concatenated(rows, vector<float>(cols, 0.0f));
    
    for (int h = 0; h < NUM_HEADS; h++) {
        auto Wq = generate_random_matrix(EMBED_DIM, HEAD_DIM);
        auto Wk = generate_random_matrix(EMBED_DIM, HEAD_DIM);
        auto Wv = generate_random_matrix(EMBED_DIM, HEAD_DIM);
        
        auto Q = cuda_matmul(input, Wq, num_threads);
        auto K = cuda_matmul(input, Wk, num_threads);
        auto V = cuda_matmul(input, Wv, num_threads);
        
        float* h_Q = flatten_matrix(Q);
        float* h_K = flatten_matrix(K);
        float* h_scores = new float[rows * rows];
        
        float *d_Q, *d_K, *d_scores;
        CUDA_CHECK(cudaMalloc(&d_Q, rows * HEAD_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_K, rows * HEAD_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_scores, rows * rows * sizeof(float)));
        
        CUDA_CHECK(cudaMemcpy(d_Q, h_Q, rows * HEAD_DIM * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, h_K, rows * HEAD_DIM * sizeof(float), cudaMemcpyHostToDevice));
        
        dim3 threadsPerBlock(min(num_threads, 32), min(num_threads, 32));
        dim3 blocksPerGrid((rows + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
        
        attention_scores_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_Q, d_K, d_scores, rows, HEAD_DIM, scale_factor);
        CUDA_CHECK(cudaGetLastError());
        
        CUDA_CHECK(cudaMemcpy(h_scores, d_scores, rows * rows * sizeof(float), cudaMemcpyDeviceToHost));
        
        float *d_softmax_output;
        CUDA_CHECK(cudaMalloc(&d_softmax_output, rows * rows * sizeof(float)));
        
        int softmax_threads = min(num_threads, 1024);
        int softmax_blocks = (rows + softmax_threads - 1) / softmax_threads;
        
        softmax_kernel<<<softmax_blocks, softmax_threads>>>(d_scores, d_softmax_output, rows, rows);
        CUDA_CHECK(cudaGetLastError());
        
        CUDA_CHECK(cudaMemcpy(h_scores, d_softmax_output, rows * rows * sizeof(float), cudaMemcpyDeviceToHost));
        
        vector<vector<float>> scores = unflatten_matrix(h_scores, rows, rows);
        
        auto head_output = cuda_matmul(scores, V, num_threads);
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < HEAD_DIM; j++) {
                concatenated[i][h * HEAD_DIM + j] = head_output[i][j];
            }
        }
        
        CUDA_CHECK(cudaFree(d_Q));
        CUDA_CHECK(cudaFree(d_K));
        CUDA_CHECK(cudaFree(d_scores));
        CUDA_CHECK(cudaFree(d_softmax_output));
        
        delete[] h_Q;
        delete[] h_K;
        delete[] h_scores;
    }
    
    auto Wo = generate_random_matrix(EMBED_DIM, EMBED_DIM);
    auto projected = cuda_matmul(concatenated, Wo, num_threads);
    
    auto dropped = cuda_dropout(projected, dropout_rate, num_threads);
    
    vector<vector<float>> residual = input;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            residual[i][j] += dropped[i][j];
        }
    }
    
    return cuda_layer_norm(residual, 1e-5, num_threads);
}

vector<vector<float>> cuda_feedforward_layer(const vector<vector<float>>& input, float dropout_rate, int num_threads) {
    auto W1 = generate_random_matrix(EMBED_DIM, HIDDEN_DIM);
    auto W2 = generate_random_matrix(HIDDEN_DIM, EMBED_DIM);
    
    auto hidden = cuda_matmul(input, W1, num_threads);
    
    hidden = cuda_gelu(hidden, num_threads);
    
    auto output = cuda_matmul(hidden, W2, num_threads);
    
    auto dropped = cuda_dropout(output, dropout_rate, num_threads);
    
    
    vector<vector<float>> residual = input;
    for (int i = 0; i < input.size(); i++) {
        for (int j = 0; j < input[0].size(); j++) {
            residual[i][j] += dropped[i][j];
        }
    }
    
    
    return cuda_layer_norm(residual, 1e-5, num_threads);
}

// CUDA implementation of softmax
vector<float> cuda_softmax(const vector<float>& logits, int num_threads) {
    int size = logits.size();
    
    
    float* h_input = new float[size];
    float* h_output = new float[size];
    
    
    for (int i = 0; i < size; i++) {
        h_input[i] = logits[i];
    }
    
    
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(float)));
    
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice));
    
    
    int threadsPerBlock = min(num_threads, 1024);
    int blocksPerGrid = 1; 
    
    
    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, 1, size);
    CUDA_CHECK(cudaGetLastError());
    
    
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    
    vector<float> result(size);
    for (int i = 0; i < size; i++) {
        result[i] = h_output[i];
    }
    
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    
    delete[] h_input;
    delete[] h_output;
    
    return result;
}

// Helper function to add positional encoding to embeddings
void cuda_add_positional_encoding(vector<vector<float>>& embeddings, const vector<vector<float>>& pos_encoding, int num_threads) {
    int seq_len = embeddings.size();
    int embed_dim = embeddings[0].size();
    
    
    float* h_embeddings = flatten_matrix(embeddings);
    float* h_pos_encoding = flatten_matrix(pos_encoding);
    
    
    float *d_embeddings, *d_pos_encoding;
    CUDA_CHECK(cudaMalloc(&d_embeddings, seq_len * embed_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pos_encoding, seq_len * embed_dim * sizeof(float)));
    
    
    CUDA_CHECK(cudaMemcpy(d_embeddings, h_embeddings, seq_len * embed_dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_encoding, h_pos_encoding, seq_len * embed_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    
    dim3 threadsPerBlock(min(num_threads, 32), min(num_threads, 32));
    dim3 blocksPerGrid((embed_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (seq_len + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    
    add_pos_encoding_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_embeddings, d_pos_encoding, seq_len, embed_dim);
    CUDA_CHECK(cudaGetLastError());
    
    
    CUDA_CHECK(cudaMemcpy(h_embeddings, d_embeddings, seq_len * embed_dim * sizeof(float), cudaMemcpyDeviceToHost));
        
    
    CUDA_CHECK(cudaFree(d_embeddings));
    CUDA_CHECK(cudaFree(d_pos_encoding));

    
    embeddings = unflatten_matrix(h_embeddings, seq_len, embed_dim);

    
    delete[] h_embeddings;
    delete[] h_pos_encoding;
}
// Full transformer encoder layer
vector<vector<float>> transformer_encoder_layer(const vector<vector<float>>& input, float dropout_rate, int num_threads) {
    
    auto attention_output = cuda_self_attention(input, dropout_rate, num_threads);
    
    
    return cuda_feedforward_layer(attention_output, dropout_rate, num_threads);
}

// Full transformer encoder (multi-layer)
vector<vector<float>> transformer_encoder(const vector<vector<float>>& input, int num_layers, float dropout_rate, int num_threads) {
    vector<vector<float>> output = input;
    
    for (int i = 0; i < num_layers; i++) {
        output = transformer_encoder_layer(output, dropout_rate, num_threads);
    }
    
    return output;
}

// Serial implementation of matmul for benchmarking
vector<vector<float>> serial_matmul(const vector<vector<float>>& A, const vector<vector<float>>& B) {
    int M = A.size();
    int K = A[0].size();
    int N = B[0].size();
    
    if (K != B.size())
        throw runtime_error("Matrix dimension mismatch in multiplication");
    
    vector<vector<float>> C(M, vector<float>(N, 0.0f));
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    
    return C;
}

// Serial implementation of layer normalization for benchmarking
vector<vector<float>> serial_layer_norm(const vector<vector<float>>& input, float eps) {
    int rows = input.size();
    int cols = input[0].size();
    
    vector<vector<float>> normalized(rows, vector<float>(cols));
    
    for (int i = 0; i < rows; i++) {
        
        float mean = 0.0f;
        for (int j = 0; j < cols; j++) {
            mean += input[i][j];
        }
        mean /= cols;
        
        
        float var = 0.0f;
        for (int j = 0; j < cols; j++) {
            float diff = input[i][j] - mean;
            var += diff * diff;
        }
        var /= cols;
        
        
        float inv_std = 1.0f / sqrt(var + eps);
        for (int j = 0; j < cols; j++) {
            normalized[i][j] = (input[i][j] - mean) * inv_std;
        }
    }
    
    return normalized;
}

// Serial implementation of positional encoding for benchmarking
vector<vector<float>> serial_positional_encoding(int seq_len, int embed_dim) {
    vector<vector<float>> pe(seq_len, vector<float>(embed_dim, 0.0f));
    
    for (int pos = 0; pos < seq_len; pos++) {
        for (int i = 0; i < embed_dim; i++) {
            float angle = pos / pow(10000, (2 * (i / 2)) / static_cast<float>(embed_dim));
            if (i % 2 == 0)
                pe[pos][i] = sin(angle);
            else
                pe[pos][i] = cos(angle);
        }
    }
    
    return pe;
}

// Serial implementation of transformer encoder layer for benchmarking
vector<vector<float>> serial_transformer_encoder_layer(const vector<vector<float>>& input, float dropout_rate) {
    int seq_len = input.size();
    int embed_dim = input[0].size();
    
    
    
    vector<vector<float>> attention_output = input;
    
    
    
    auto W1 = generate_random_matrix(embed_dim, HIDDEN_DIM);
    auto W2 = generate_random_matrix(HIDDEN_DIM, embed_dim);
    
    
    auto hidden = serial_matmul(attention_output, W1);

    for (auto& row : hidden) {
        for (auto& val : row) {
            val = max(0.0f, val);
        }
    }
    
    auto output = serial_matmul(hidden, W2);
    
    
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < embed_dim; j++) {
            output[i][j] += input[i][j];
        }
    }
    
    
    return serial_layer_norm(output, 1e-5);
}

// Function to run benchmark for a specific thread count
void run_benchmark(int num_threads, vector<double>& times, vector<double>& speedups, vector<double>& parallel_factors) {
    cout << "Running benchmark with " << num_threads << " threads per dimension..." << endl;
    
    
    vector<vector<float>> input(SEQ_LEN, vector<float>(EMBED_DIM));
    
    
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            input[i][j] = dist(gen);
        }
    }
    
    
    auto start_serial = chrono::high_resolution_clock::now();
    auto serial_output = serial_transformer_encoder_layer(input, 0.1);
    auto end_serial = chrono::high_resolution_clock::now();
    chrono::duration<double> serial_time = end_serial - start_serial;
    
   
    auto start_parallel = chrono::high_resolution_clock::now();
    auto parallel_output = transformer_encoder_layer(input, 0.1, num_threads);
    auto end_parallel = chrono::high_resolution_clock::now();
    chrono::duration<double> parallel_time = end_parallel - start_parallel;
    
    
    double speedup = serial_time.count() / parallel_time.count();
    double parallel_factor = 1.0 / ((1.0 / speedup) + (1.0 - 1.0) / num_threads);
    
    cout << "Serial time: " << serial_time.count() << " seconds" << endl;
    cout << "Parallel time: " << parallel_time.count() << " seconds" << endl;
    cout << "Speedup: " << speedup << "x" << endl;
    cout << "Parallel factor: " << parallel_factor << endl;
    cout << "-------------------------------------------" << endl;
    
    
    times.push_back(parallel_time.count());
    speedups.push_back(speedup);
    parallel_factors.push_back(parallel_factor);
}

// Function to check CUDA device properties
void check_cuda_device() {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        cerr << "No CUDA devices found!" << endl;
        exit(EXIT_FAILURE);
    }
    
    cout << "Found " << deviceCount << " CUDA device(s)" << endl;
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        
        cout << "Device " << i << ": " << prop.name << endl;
        cout << "  Compute capability: " << prop.major << "." << prop.minor << endl;
        cout << "  Max threads per block: " << prop.maxThreadsPerBlock << endl;
        cout << "  Max threads dimensions: (" 
             << prop.maxThreadsDim[0] << ", " 
             << prop.maxThreadsDim[1] << ", " 
             << prop.maxThreadsDim[2] << ")" << endl;
        cout << "  Max grid dimensions: (" 
             << prop.maxGridSize[0] << ", " 
             << prop.maxGridSize[1] << ", " 
             << prop.maxGridSize[2] << ")" << endl;
        cout << "  Global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << endl;
        cout << "  Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB" << endl;
        cout << "  Warp size: " << prop.warpSize << endl;
        cout << "-------------------------------------------" << endl;
    }
}

// Main function for benchmarking
int main() {
    
    check_cuda_device();
    
    
    vector<int> thread_counts = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    
    
    vector<double> parallel_times;
    vector<double> speedups;
    vector<double> parallel_factors;
    
    
    for (int threads : thread_counts) {
        run_benchmark(threads, parallel_times, speedups, parallel_factors);
    }
    
   
    cout << "\nBenchmark Summary:" << endl;
    cout << "-------------------------------------------" << endl;
    cout << "Threads | Parallel Time (s) | Speedup | Parallel Factor" << endl;
    cout << "-------------------------------------------" << endl;
    
    for (size_t i = 0; i < thread_counts.size(); i++) {
        cout << setw(7) << thread_counts[i] << " | " 
             << setw(17) << fixed << setprecision(6) << parallel_times[i] << " | " 
             << setw(7) << fixed << setprecision(2) << speedups[i] << " | " 
             << setw(15) << fixed << setprecision(4) << parallel_factors[i] << endl;
    }
    
    
    ofstream csv_file("transformer_benchmark_results.csv");
    if (csv_file.is_open()) {
        
        csv_file << "Threads,Parallel Time (s),Speedup,Parallel Factor\n";
        
        
        for (size_t i = 0; i < thread_counts.size(); i++) {
            csv_file << thread_counts[i] << "," 
                     << fixed << setprecision(6) << parallel_times[i] << "," 
                     << fixed << setprecision(4) << speedups[i] << "," 
                     << fixed << setprecision(4) << parallel_factors[i] << "\n";
        }
        
        csv_file.close();
        cout << "\nResults saved to transformer_benchmark_results.csv" << endl;
    } else {
        cerr << "Error: Could not open CSV file for writing" << endl;
    }
    
    return 0;
}