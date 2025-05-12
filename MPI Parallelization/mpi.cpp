#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <mpi.h>

using namespace std;

// Constants for the transformer model
const int SEQ_LEN    = 1024;
const int EMBED_DIM  = 1024;
const int HIDDEN_DIM = 2048;
const int NUM_HEADS  = 8;
const int HEAD_DIM   = EMBED_DIM / NUM_HEADS;

// Helper functions for debugging and visualization
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

// Random Matrix Generation with seed to ensure consistency across processes
vector<vector<float>> generate_random_matrix(int rows, int cols, unsigned int seed = 42) {
    vector<vector<float>> matrix(rows, vector<float>(cols));
    mt19937 gen(seed);
    float scale = sqrt(2.0f / (rows + cols));
    normal_distribution<float> dist(0.0f, scale);

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            matrix[i][j] = dist(gen);

    return matrix;
}

// Flatten the 2D matrix for MPI communication
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

// Convert 1D array back to 2D vector
vector<vector<float>> unflatten_matrix(float* flat, int rows, int cols) {
    vector<vector<float>> matrix(rows, vector<float>(cols));
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = flat[i * cols + j];
        }
    }
    
    return matrix;
}

// Serial implementation of matrix multiplication for benchmarking
vector<vector<float>> serial_matmul(const vector<vector<float>>& A, const vector<vector<float>>& B) {
    int M = A.size();
    int K = A[0].size();
    int N = B[0].size();
    
    if (K != B.size())
        throw runtime_error("Matrix dimension mismatch in multiplication");
        
    vector<vector<float>> C(M, vector<float>(N, 0.0f));
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    
    return C;
}

// Serial implementation of layer normalization for benchmarking
vector<vector<float>> serial_layer_norm(const vector<vector<float>>& input, float eps) {
    int rows = input.size();
    int cols = input[0].size();
    
    vector<vector<float>> output(rows, vector<float>(cols));
    
    for (int i = 0; i < rows; i++) {
        // Calculate mean
        float mean = 0.0f;
        for (int j = 0; j < cols; j++) {
            mean += input[i][j];
        }
        mean /= cols;
        
        // Calculate variance
        float var = 0.0f;
        for (int j = 0; j < cols; j++) {
            float diff = input[i][j] - mean;
            var += diff * diff;
        }
        var /= cols;
        
        // Normalize
        float inv_std = 1.0f / sqrt(var + eps);
        for (int j = 0; j < cols; j++) {
            output[i][j] = (input[i][j] - mean) * inv_std;
        }
    }
    
    return output;
}

// MPI implementation of matrix multiplication
vector<vector<float>> mpi_matmul(const vector<vector<float>>& A, const vector<vector<float>>& B, int world_rank, int world_size) {
    int M = A.size();
    int K = A[0].size();
    int N = B[0].size();
    
    if (K != B.size())
        throw runtime_error("Matrix dimension mismatch in multiplication");

    // Calculate how many rows each process will compute
    int rows_per_process = M / world_size;
    int start_row = world_rank * rows_per_process;
    int end_row = (world_rank == world_size - 1) ? M : start_row + rows_per_process;
    int local_rows = end_row - start_row;
    
    // Create flattened matrices for MPI
    float* A_flat = flatten_matrix(A);
    float* B_flat = flatten_matrix(B);
    
    // Broadcast matrix B to all processes
    MPI_Bcast(B_flat, K * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    // Allocate space for the local result
    vector<vector<float>> local_C(local_rows, vector<float>(N, 0.0f));
    
    // Perform the local matrix multiplication
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A_flat[(start_row + i) * K + k] * B_flat[k * N + j];
            }
            local_C[i][j] = sum;
        }
    }
    
    // Prepare for gathering results
    float* local_C_flat = flatten_matrix(local_C);
    float* C_flat = nullptr;
    
    if (world_rank == 0) {
        C_flat = new float[M * N];
    }
    
    // Determine the sendcounts and displacements for MPI_Gatherv
    int* sendcounts = new int[world_size];
    int* displs = new int[world_size];
    
    for (int i = 0; i < world_size; i++) {
        int process_rows = (i == world_size - 1) ? (M - i * rows_per_process) : rows_per_process;
        sendcounts[i] = process_rows * N;
        displs[i] = i * rows_per_process * N;
    }
    
    // Gather results from all processes
    MPI_Gatherv(local_C_flat, local_rows * N, MPI_FLOAT, 
                C_flat, sendcounts, displs, MPI_FLOAT, 
                0, MPI_COMM_WORLD);
    
    // Convert back to 2D vector (only the root process has the complete result)
    vector<vector<float>> C;
    if (world_rank == 0) {
        C = unflatten_matrix(C_flat, M, N);
        delete[] C_flat;
    }
    
    // Broadcast the result to all processes
    if (world_rank == 0) {
        C_flat = flatten_matrix(C);
    } else {
        C_flat = new float[M * N];
    }
    
    MPI_Bcast(C_flat, M * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    if (world_rank != 0) {
        C = unflatten_matrix(C_flat, M, N);
    }
    
    // Clean up
    delete[] A_flat;
    delete[] B_flat;
    delete[] local_C_flat;
    delete[] C_flat;
    delete[] sendcounts;
    delete[] displs;
    
    return C;
}

// MPI implementation of layer normalization
vector<vector<float>> mpi_layer_norm(const vector<vector<float>>& input, float eps, int world_rank, int world_size) {
    int rows = input.size();
    int cols = input[0].size();
    
    // Calculate how many rows each process will compute
    int rows_per_process = rows / world_size;
    int start_row = world_rank * rows_per_process;
    int end_row = (world_rank == world_size - 1) ? rows : start_row + rows_per_process;
    int local_rows = end_row - start_row;
    
    // Flatten input for MPI communication
    float* input_flat = flatten_matrix(input);
    
    // Allocate space for the local result
    vector<vector<float>> local_output(local_rows, vector<float>(cols));
    
    // Perform the local layer normalization
    for (int i = 0; i < local_rows; i++) {
        // Calculate mean
        float mean = 0.0f;
        for (int j = 0; j < cols; j++) {
            mean += input_flat[(start_row + i) * cols + j];
        }
        mean /= cols;
        
        // Calculate variance
        float var = 0.0f;
        for (int j = 0; j < cols; j++) {
            float diff = input_flat[(start_row + i) * cols + j] - mean;
            var += diff * diff;
        }
        var /= cols;
        
        // Normalize
        float inv_std = 1.0f / sqrt(var + eps);
        for (int j = 0; j < cols; j++) {
            local_output[i][j] = (input_flat[(start_row + i) * cols + j] - mean) * inv_std;
        }
    }
    
    // Prepare for gathering results
    float* local_output_flat = flatten_matrix(local_output);
    float* output_flat = nullptr;
    
    if (world_rank == 0) {
        output_flat = new float[rows * cols];
    }
    
    // Determine the sendcounts and displacements for MPI_Gatherv
    int* sendcounts = new int[world_size];
    int* displs = new int[world_size];
    
    for (int i = 0; i < world_size; i++) {
        int process_rows = (i == world_size - 1) ? (rows - i * rows_per_process) : rows_per_process;
        sendcounts[i] = process_rows * cols;
        displs[i] = i * rows_per_process * cols;
    }
    
    // Gather results from all processes
    MPI_Gatherv(local_output_flat, local_rows * cols, MPI_FLOAT, 
                output_flat, sendcounts, displs, MPI_FLOAT, 
                0, MPI_COMM_WORLD);
    
    // Convert back to 2D vector (only the root process has the complete result)
    vector<vector<float>> output;
    if (world_rank == 0) {
        output = unflatten_matrix(output_flat, rows, cols);
        delete[] output_flat;
    }
    
    // Broadcast the result to all processes
    if (world_rank == 0) {
        output_flat = flatten_matrix(output);
    } else {
        output_flat = new float[rows * cols];
    }
    
    MPI_Bcast(output_flat, rows * cols, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    if (world_rank != 0) {
        output = unflatten_matrix(output_flat, rows, cols);
    }
    
    // Clean up
    delete[] input_flat;
    delete[] local_output_flat;
    delete[] output_flat;
    delete[] sendcounts;
    delete[] displs;
    
    return output;
}

// MPI implementation of positional encoding
vector<vector<float>> mpi_positional_encoding(int seq_len, int embed_dim, int world_rank, int world_size) {
    // Calculate how many rows each process will compute
    int rows_per_process = seq_len / world_size;
    int start_row = world_rank * rows_per_process;
    int end_row = (world_rank == world_size - 1) ? seq_len : start_row + rows_per_process;
    int local_rows = end_row - start_row;
    
    // Allocate space for the local result
    vector<vector<float>> local_pe(local_rows, vector<float>(embed_dim, 0.0f));
    
    // Compute the positional encoding for the local rows
    for (int pos = 0; pos < local_rows; pos++) {
        for (int i = 0; i < embed_dim; i++) {
            float angle = (start_row + pos) / pow(10000, (2 * (i / 2)) / static_cast<float>(embed_dim));
            if (i % 2 == 0)
                local_pe[pos][i] = sin(angle);
            else
                local_pe[pos][i] = cos(angle);
        }
    }
    
    // Prepare for gathering results
    float* local_pe_flat = flatten_matrix(local_pe);
    float* pe_flat = nullptr;
    
    if (world_rank == 0) {
        pe_flat = new float[seq_len * embed_dim];
    }
    
    // Determine the sendcounts and displacements for MPI_Gatherv
    int* sendcounts = new int[world_size];
    int* displs = new int[world_size];
    
    for (int i = 0; i < world_size; i++) {
        int process_rows = (i == world_size - 1) ? (seq_len - i * rows_per_process) : rows_per_process;
        sendcounts[i] = process_rows * embed_dim;
        displs[i] = i * rows_per_process * embed_dim;
    }
    
    // Gather results from all processes
    MPI_Gatherv(local_pe_flat, local_rows * embed_dim, MPI_FLOAT, 
                pe_flat, sendcounts, displs, MPI_FLOAT, 
                0, MPI_COMM_WORLD);
    
    // Convert back to 2D vector (only the root process has the complete result)
    vector<vector<float>> pe;
    if (world_rank == 0) {
        pe = unflatten_matrix(pe_flat, seq_len, embed_dim);
        delete[] pe_flat;
    }
    
    // Broadcast the result to all processes
    if (world_rank == 0) {
        pe_flat = flatten_matrix(pe);
    } else {
        pe_flat = new float[seq_len * embed_dim];
    }
    
    MPI_Bcast(pe_flat, seq_len * embed_dim, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    if (world_rank != 0) {
        pe = unflatten_matrix(pe_flat, seq_len, embed_dim);
    }
    
    // Clean up
    delete[] local_pe_flat;
    delete[] pe_flat;
    delete[] sendcounts;
    delete[] displs;
    
    return pe;
}

// MPI implementation of dropout - using the same random seed across processes for consistency
vector<vector<float>> mpi_dropout(const vector<vector<float>>& input, float dropout_rate, int world_rank, int world_size) {
    int rows = input.size();
    int cols = input[0].size();
    
    // Calculate how many rows each process will compute
    int rows_per_process = rows / world_size;
    int start_row = world_rank * rows_per_process;
    int end_row = (world_rank == world_size - 1) ? rows : start_row + rows_per_process;
    int local_rows = end_row - start_row;
    
    // Create a copy of the input matrix
    vector<vector<float>> output = input;
    
    // Generate random numbers for dropout - use fixed seed for consistency across processes
    unsigned int seed = 12345; // Fixed seed for deterministic behavior
    mt19937 gen(seed);
    uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    // Use the same random state across all processes by advancing the RNG
    for (int i = 0; i < start_row * cols; i++) {
        dist(gen); // Advance the RNG to the same state for this row
    }
    
    // Apply dropout to the local rows
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < cols; j++) {
            if (dist(gen) < dropout_rate)
                output[i][j] = 0.0f;
            else
                output[i][j] /= (1.0f - dropout_rate);
        }
    }
    
    // Synchronize the output across all processes
    float* output_flat = flatten_matrix(output);
    
    // Each process knows its own part, broadcast other parts
    for (int i = 0; i < world_size; i++) {
        int proc_start = i * rows_per_process;
        int proc_end = (i == world_size - 1) ? rows : proc_start + rows_per_process;
        int proc_rows = proc_end - proc_start;
        
        // Broadcast from process i to all others
        MPI_Bcast(&output_flat[proc_start * cols], proc_rows * cols, MPI_FLOAT, i, MPI_COMM_WORLD);
    }
    
    output = unflatten_matrix(output_flat, rows, cols);
    
    // Clean up
    delete[] output_flat;
    
    return output;
}

// MPI implementation of GELU activation
vector<vector<float>> mpi_gelu(const vector<vector<float>>& input, int world_rank, int world_size) {
    int rows = input.size();
    int cols = input[0].size();
    
    // Calculate how many rows each process will compute
    int rows_per_process = rows / world_size;
    int start_row = world_rank * rows_per_process;
    int end_row = (world_rank == world_size - 1) ? rows : start_row + rows_per_process;
    int local_rows = end_row - start_row;
    
    // Create output matrix
    vector<vector<float>> output(rows, vector<float>(cols));
    
    // Apply GELU to the local rows
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < cols; j++) {
            float x = input[i][j];
            output[i][j] = 0.5f * x * (1.0f + tanh(sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
        }
    }
    
    // Synchronize the output across all processes
    float* local_output_flat = new float[rows * cols]();
    float* output_flat = new float[rows * cols];
    
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < cols; j++) {
            local_output_flat[i * cols + j] = output[i][j];
        }
    }
    
    MPI_Allreduce(local_output_flat, output_flat, rows * cols, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    output = unflatten_matrix(output_flat, rows, cols);
    
    // Clean up
    delete[] local_output_flat;
    delete[] output_flat;
    
    return output;
}

// MPI implementation of softmax
vector<vector<float>> mpi_softmax(const vector<vector<float>>& input, int world_rank, int world_size) {
    int rows = input.size();
    int cols = input[0].size();
    
    // Calculate how many rows each process will compute
    int rows_per_process = rows / world_size;
    int start_row = world_rank * rows_per_process;
    int end_row = (world_rank == world_size - 1) ? rows : start_row + rows_per_process;
    int local_rows = end_row - start_row;
    
    // Create output matrix
    vector<vector<float>> output(rows, vector<float>(cols));
    
    // First find max values for each row
    vector<float> max_values(rows, -INFINITY);
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < cols; j++) {
            max_values[i] = max(max_values[i], input[i][j]);
        }
    }
    
    // Synchronize max values across all processes
    MPI_Allreduce(MPI_IN_PLACE, max_values.data(), rows, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    
    // Calculate exponentials and sums
    vector<float> sum_exp(rows, 0.0f);
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < cols; j++) {
            output[i][j] = exp(input[i][j] - max_values[i]);
            sum_exp[i] += output[i][j];
        }
    }
    
    // Synchronize sums across all processes
    MPI_Allreduce(MPI_IN_PLACE, sum_exp.data(), rows, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    
    // Normalize
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < cols; j++) {
            output[i][j] /= sum_exp[i];
        }
    }
    
    // Synchronize the output across all processes
    float* local_output_flat = new float[rows * cols]();
    float* output_flat = new float[rows * cols];
    
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < cols; j++) {
            local_output_flat[i * cols + j] = output[i][j];
        }
    }
    
    MPI_Allreduce(local_output_flat, output_flat, rows * cols, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    output = unflatten_matrix(output_flat, rows, cols);
    
    // Clean up
    delete[] local_output_flat;
    delete[] output_flat;
    
    return output;
}

// MPI implementation of self-attention
vector<vector<float>> mpi_self_attention(const vector<vector<float>>& input, float dropout_rate, int world_rank, int world_size) {
    int rows = input.size();
    int cols = input[0].size();
    float scale_factor = 1.0f / sqrt(static_cast<float>(HEAD_DIM));
    
    vector<vector<float>> concatenated(rows, vector<float>(cols, 0.0f));
    
    for (int h = 0; h < NUM_HEADS; h++) {
        // Only process 0 generates the weight matrices to ensure consistency
        vector<vector<float>> Wq, Wk, Wv;
        if (world_rank == 0) {
            // Use fixed seed for deterministic initialization across runs
            Wq = generate_random_matrix(EMBED_DIM, HEAD_DIM, 42 + h * 3);
            Wk = generate_random_matrix(EMBED_DIM, HEAD_DIM, 42 + h * 3 + 1);
            Wv = generate_random_matrix(EMBED_DIM, HEAD_DIM, 42 + h * 3 + 2);
        } else {
            Wq.resize(EMBED_DIM, vector<float>(HEAD_DIM));
            Wk.resize(EMBED_DIM, vector<float>(HEAD_DIM));
            Wv.resize(EMBED_DIM, vector<float>(HEAD_DIM));
        }
        
        // Broadcast weight matrices to all processes
        float* Wq_flat = flatten_matrix(Wq);
        float* Wk_flat = flatten_matrix(Wk);
        float* Wv_flat = flatten_matrix(Wv);
        
        MPI_Bcast(Wq_flat, EMBED_DIM * HEAD_DIM, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(Wk_flat, EMBED_DIM * HEAD_DIM, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(Wv_flat, EMBED_DIM * HEAD_DIM, MPI_FLOAT, 0, MPI_COMM_WORLD);
        
        if (world_rank != 0) {
            Wq = unflatten_matrix(Wq_flat, EMBED_DIM, HEAD_DIM);
            Wk = unflatten_matrix(Wk_flat, EMBED_DIM, HEAD_DIM);
            Wv = unflatten_matrix(Wv_flat, EMBED_DIM, HEAD_DIM);
        }
        
        // Calculate Q, K, V matrices
        auto Q = mpi_matmul(input, Wq, world_rank, world_size);
        auto K = mpi_matmul(input, Wk, world_rank, world_size);
        auto V = mpi_matmul(input, Wv, world_rank, world_size);
        
        // Calculate attention scores
        vector<vector<float>> scores(rows, vector<float>(rows, 0.0f));
        
        // Divide the work of computing scores
        int scores_per_process = (rows * rows) / world_size;
        int start_idx = world_rank * scores_per_process;
        int end_idx = (world_rank == world_size - 1) ? (rows * rows) : start_idx + scores_per_process;
        
        for (int idx = start_idx; idx < end_idx; idx++) {
            int i = idx / rows;
            int j = idx % rows;
            
            float dot = 0.0f;
            for (int k = 0; k < HEAD_DIM; k++) {
                dot += Q[i][k] * K[j][k];
            }
            scores[i][j] = dot * scale_factor;
        }
        
        // Synchronize scores across all processes
        float* local_scores_flat = flatten_matrix(scores);
        float* scores_flat = new float[rows * rows];
        
        MPI_Allreduce(local_scores_flat, scores_flat, rows * rows, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        scores = unflatten_matrix(scores_flat, rows, rows);
        
        // Apply softmax to get attention weights
        auto attention_weights = mpi_softmax(scores, world_rank, world_size);
        
        // Apply dropout to attention weights
        attention_weights = mpi_dropout(attention_weights, dropout_rate, world_rank, world_size);
        
        // Calculate weighted values
        auto head_output = mpi_matmul(attention_weights, V, world_rank, world_size);
        
        // Collect the head output into the concatenated result
        int rows_per_process = rows / world_size;
        int start_row = world_rank * rows_per_process;
        int end_row = (world_rank == world_size - 1) ? rows : start_row + rows_per_process;
        
        for (int i = start_row; i < end_row; i++) {
            for (int j = 0; j < HEAD_DIM; j++) {
                concatenated[i][h * HEAD_DIM + j] = head_output[i][j];
            }
        }
        
        // Synchronize the concatenated matrix
        float* local_concat_flat = new float[rows * cols]();
        float* concat_flat = new float[rows * cols];
        
        for (int i = start_row; i < end_row; i++) {
            for (int j = 0; j < cols; j++) {
                local_concat_flat[i * cols + j] = concatenated[i][j];
            }
        }
        
        MPI_Allreduce(local_concat_flat, concat_flat, rows * cols, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        concatenated = unflatten_matrix(concat_flat, rows, cols);
        
        // Clean up
        delete[] Wq_flat;
        delete[] Wk_flat;
        delete[] Wv_flat;
        delete[] local_scores_flat;
        delete[] scores_flat;
        delete[] local_concat_flat;
        delete[] concat_flat;
    }
    
    // Project back to the original dimensionality
    vector<vector<float>> Wo;
    if (world_rank == 0) {
        Wo = generate_random_matrix(EMBED_DIM, EMBED_DIM, 42 + NUM_HEADS * 3);
    } else {
        Wo.resize(EMBED_DIM, vector<float>(EMBED_DIM));
    }
    
    // Broadcast Wo to all processes
    float* Wo_flat = flatten_matrix(Wo);
    MPI_Bcast(Wo_flat, EMBED_DIM * EMBED_DIM, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    if (world_rank != 0) {
        Wo = unflatten_matrix(Wo_flat, EMBED_DIM, EMBED_DIM);
    }
    
    auto projected = mpi_matmul(concatenated, Wo, world_rank, world_size);
    
    // Apply dropout
    auto dropped = mpi_dropout(projected, dropout_rate, world_rank, world_size);
    
    // Add residual connection
    vector<vector<float>> residual = input;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            residual[i][j] += dropped[i][j];
        }
    }
    
    // Apply layer normalization
    auto normalized = mpi_layer_norm(residual, 1e-5, world_rank, world_size);
    
    // Clean up
    delete[] Wo_flat;
    
    return normalized;
}

// MPI implementation of feedforward network
vector<vector<float>> mpi_feedforward_layer(const vector<vector<float>>& input, float dropout_rate, int world_rank, int world_size) {
    int rows = input.size();
    int cols = input[0].size();
    
    // Only process 0 generates the weight matrices to ensure consistency
    vector<vector<float>> W1, W2;
    if (world_rank == 0) {
        W1 = generate_random_matrix(EMBED_DIM, HIDDEN_DIM);
        W2 = generate_random_matrix(HIDDEN_DIM, EMBED_DIM);
    } else {
        W1.resize(EMBED_DIM, vector<float>(HIDDEN_DIM));
        W2.resize(HIDDEN_DIM, vector<float>(EMBED_DIM));
    }
    
    // Broadcast weight matrices to all processes
    float* W1_flat = flatten_matrix(W1);
    float* W2_flat = flatten_matrix(W2);
    
    MPI_Bcast(W1_flat, EMBED_DIM * HIDDEN_DIM, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(W2_flat, HIDDEN_DIM * EMBED_DIM, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    if (world_rank != 0) {
        W1 = unflatten_matrix(W1_flat, EMBED_DIM, HIDDEN_DIM);
        W2 = unflatten_matrix(W2_flat, HIDDEN_DIM, EMBED_DIM);
    }
    
    // Calculate the first linear transformation
    auto hidden = mpi_matmul(input, W1, world_rank, world_size);
    
    // Apply GELU activation
    hidden = mpi_gelu(hidden, world_rank, world_size);
    
    // Calculate the second linear transformation
    auto output = mpi_matmul(hidden, W2, world_rank, world_size);
    
    // Apply dropout
    auto dropped = mpi_dropout(output, dropout_rate, world_rank, world_size);
    
    // Add residual connection
    vector<vector<float>> residual = input;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            residual[i][j] += dropped[i][j];
        }
    }
    
    // Apply layer normalization
    auto normalized = mpi_layer_norm(residual, 1e-5, world_rank, world_size);
    
    // Clean up
    delete[] W1_flat;
    delete[] W2_flat;
    
    return normalized;
}

// Full transformer encoder layer using MPI
vector<vector<float>> mpi_transformer_encoder_layer(const vector<vector<float>>& input, float dropout_rate, int world_rank, int world_size) {
    // Self-attention sublayer
    auto attention_output = mpi_self_attention(input, dropout_rate, world_rank, world_size);
    
    // Feedforward sublayer
    return mpi_feedforward_layer(attention_output, dropout_rate, world_rank, world_size);
}

// Full transformer encoder (multi-layer) using MPI
vector<vector<float>> mpi_transformer_encoder(const vector<vector<float>>& input, int num_layers, float dropout_rate, int world_rank, int world_size) {
    vector<vector<float>> output = input;
    
    for (int i = 0; i < num_layers; i++) {
        output = mpi_transformer_encoder_layer(output, dropout_rate, world_rank, world_size);
    }
    
    return output;
}

// Serial implementation of transformer encoder layer for benchmarking
vector<vector<float>> serial_transformer_encoder_layer(const vector<vector<float>>& input, float dropout_rate) {
    int seq_len = input.size();
    int embed_dim = input[0].size();
    
    // Placeholder for self-attention (simplified)
    vector<vector<float>> attention_output = input;
    
    // Feedforward network
    auto W1 = generate_random_matrix(embed_dim, HIDDEN_DIM);
    auto W2 = generate_random_matrix(HIDDEN_DIM, embed_dim);
    
    auto hidden = serial_matmul(attention_output, W1);
    
    // GELU activation
    for (auto& row : hidden) {
        for (auto& val : row) {
            val = 0.5f * val * (1.0f + tanh(sqrt(2.0f / M_PI) * (val + 0.044715f * val * val * val)));
        }
    }
    
    auto output = serial_matmul(hidden, W2);
    
    // Add residual connection
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < embed_dim; j++) {
            output[i][j] += input[i][j];
        }
    }
    
    // Layer normalization
    return serial_layer_norm(output, 1e-5);
}

// Function to run benchmark for MPI vs serial
void run_benchmark(int world_rank, int world_size, vector<double>& mpi_times, vector<double>& serial_times) {
    // Generate random input data
    vector<vector<float>> input(SEQ_LEN, vector<float>(EMBED_DIM));
    
    if (world_rank == 0) {
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<float> dist(0.0f, 1.0f);
        
        for (int i = 0; i < SEQ_LEN; i++) {
            for (int j = 0; j < EMBED_DIM; j++) {
                input[i][j] = dist(gen);
            }
        }
    }
    
    // Broadcast input to all processes
    float* input_flat = flatten_matrix(input);
    MPI_Bcast(input_flat, SEQ_LEN * EMBED_DIM, MPI_FLOAT, 0, MPI_COMM_WORLD);
    input = unflatten_matrix(input_flat, SEQ_LEN, EMBED_DIM);
    delete[] input_flat;
    
    // Time MPI implementation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start_mpi = chrono::high_resolution_clock::now();
    
    auto mpi_output = mpi_transformer_encoder_layer(input, 0.1, world_rank, world_size);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_mpi = chrono::high_resolution_clock::now();
    chrono::duration<double> serial_time = end_mpi - start_mpi;
    
    // Time serial implementation (only on rank 0)
    double serial_time = 0.0;
    if (world_rank == 0) {
        auto start_serial = chrono::high_resolution_clock::now();
        auto serial_output = serial_transformer_encoder_layer(input, 0.1);
        auto end_serial = chrono::high_resolution_clock::now();
        chrono::duration<double> serial_duration = end_serial - start_serial;
        mpi_time = serial_duration.count();

    }
    
    // Gather timing results
    if (world_rank == 0) {
        mpi_times.push_back(mpi_time.count());
        serial_times.push_back(serial_time);
        
        cout << "MPI Time: " << mpi_time.count() << " seconds" << endl;
        cout << "Serial Time: " << serial_time << " seconds" << endl;
        if (serial_time > 0) {
            cout << "Speedup: " << serial_time / mpi_time.count() << "x" << endl;
        }
        cout << "-------------------------------------------" << endl;
    }
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    if (world_rank == 0) {
        cout << "Running MPI Transformer Encoder Benchmark with " << world_size << " processes" << endl;
    }
    
    // Run benchmarks with different problem sizes
    vector<double> mpi_times;
    vector<double> serial_times;
    
    run_benchmark(world_rank, world_size, mpi_times, serial_times);
    
    // Output results (only rank 0)
    if (world_rank == 0) {
        ofstream csv_file("mpi_transformer_benchmark.csv");
        if (csv_file.is_open()) {
            csv_file << "Processes,MPI Time,Serial Time,Speedup\n";
            csv_file << world_size << "," 
                     << fixed << setprecision(6) << mpi_times.back() << "," 
                     << fixed << setprecision(6) << serial_times.back() << ",";
            if (serial_times.back() > 0) {
                csv_file << fixed << setprecision(2) << (serial_times.back() / mpi_times.back());
            } else {
                csv_file << "N/A";
            }
            csv_file << "\n";
            csv_file.close();
            cout << "\nResults saved to mpi_transformer_benchmark.csv" << endl;
        } else {
            cerr << "Error: Could not open CSV file for writing" << endl;
        }
    }
    
    // Finalize MPI
    MPI_Finalize();
    return 0;
}