#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <omp.h>
using namespace std;

const int SEQ_LEN = 1024;
const int EMBED_DIM = 1024;
const int HIDDEN_DIM = 2048;
const int NUM_HEADS = 8;
const int HEAD_DIM = EMBED_DIM / NUM_HEADS;

void print_vector(const vector<float> &vec, const string &name) {
    cout << name << ": [ ";
    for (float val : vec) {
        cout << fixed << setprecision(4) << val << " ";
    }
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


vector<vector<float>> generate_random_matrix(int rows, int cols) {
    vector<vector<float>> matrix(rows, vector<float>(cols));
    float scale = sqrt(2.0f / (rows + cols));

    #pragma omp parallel
    {
        mt19937 gen(hash<int>{}(omp_get_thread_num()));
        normal_distribution<float> dist(0.0f, scale);

        #pragma omp for schedule(guided)
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = dist(gen);
            }
        }
    }
    return matrix;
}

vector<vector<float>> matmul(const vector<vector<float>> &A, const vector<vector<float>> &B) {
    int M = A.size();
    int K = A[0].size();
    int N = B[0].size();
    if (K != B.size()) {
        throw runtime_error("Matrix dimension mismatch in multiplication");
    } 
    
    vector<vector<float>> C(M, vector<float>(N, 0.0f));
    const int BLOCK_SIZE = 32;  
    
    #pragma omp parallel for collapse(2) schedule(guided)
    for (int i = 0; i < M; i += BLOCK_SIZE) {
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            for (int k = 0; k < K; k += BLOCK_SIZE) {
                for (int ii = i; ii < min(i + BLOCK_SIZE, M); ii++) {
                    for (int kk = k; kk < min(k + BLOCK_SIZE, K); kk++) {
                        float temp = A[ii][kk];
                        #pragma omp simd
                        for (int jj = j; jj < min(j + BLOCK_SIZE, N); jj++) {
                            C[ii][jj] += temp * B[kk][jj];
                        }
                    }
                }
            }
        }
    }
    return C;
}

vector<vector<float>> layer_norm(const vector<vector<float>> &input, float eps = 1e-5) {
    int rows = input.size();
    int cols = input[0].size();
    vector<vector<float>> normalized(rows, vector<float>(cols));

    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < rows; i++) {
        float mean = 0.0f;
        #pragma omp simd reduction(+:mean)
        for (int j = 0; j < cols; j++) {
            mean += input[i][j];
        }
        mean /= cols;

        float var = 0.0f;
        #pragma omp simd reduction(+:var)
        for (int j = 0; j < cols; j++) {
            float diff = input[i][j] - mean;
            var += diff * diff;
        }
        var /= cols;

        float scale = 1.0f / sqrt(var + eps);
        #pragma omp simd
        for (int j = 0; j < cols; j++) {
            normalized[i][j] = (input[i][j] - mean) * scale;
        }
    }
    return normalized;
}

vector<vector<float>> self_attention(const vector<vector<float>> &input) {
    vector<vector<vector<float>>> head_outputs(NUM_HEADS);
    float scale_factor = 1.0f / sqrt(static_cast<float>(HEAD_DIM));
    
    vector<vector<vector<float>>> all_Wq(NUM_HEADS), all_Wk(NUM_HEADS), all_Wv(NUM_HEADS);
    #pragma omp parallel for
    for (int h = 0; h < NUM_HEADS; h++) {
        all_Wq[h] = generate_random_matrix(EMBED_DIM, HEAD_DIM);
        all_Wk[h] = generate_random_matrix(EMBED_DIM, HEAD_DIM);
        all_Wv[h] = generate_random_matrix(EMBED_DIM, HEAD_DIM);
    }

    #pragma omp parallel for schedule(dynamic)
    for (int h = 0; h < NUM_HEADS; h++) {
        auto Q = matmul(input, all_Wq[h]);
        auto K = matmul(input, all_Wk[h]);
        auto V = matmul(input, all_Wv[h]);

        vector<vector<float>> scores(SEQ_LEN, vector<float>(SEQ_LEN));
        #pragma omp parallel for schedule(guided)
        for (int i = 0; i < SEQ_LEN; i++) {
            for (int j = 0; j < SEQ_LEN; j++) {
                float dot_product = 0.0f;
                #pragma omp simd reduction(+:dot_product)
                for (int k = 0; k < HEAD_DIM; k++) {
                    dot_product += Q[i][k] * K[j][k];
                }
                scores[i][j] = dot_product * scale_factor;
            }
        }

        #pragma omp parallel for schedule(guided)
        for (int i = 0; i < SEQ_LEN; i++) {
            float max_val = -INFINITY;
            #pragma omp simd reduction(max:max_val)
            for (int j = 0; j < SEQ_LEN; j++) {
                max_val = max(max_val, scores[i][j]);
            }

            float sum_exp = 0.0f;
            #pragma omp simd reduction(+:sum_exp)
            for (int j = 0; j < SEQ_LEN; j++) {
                scores[i][j] = exp(scores[i][j] - max_val);
                sum_exp += scores[i][j];
            }

            float scale = 1.0f / sum_exp;
            #pragma omp simd
            for (int j = 0; j < SEQ_LEN; j++) {
                scores[i][j] *= scale;
            }
        }

        head_outputs[h] = matmul(scores, V);
    }

    vector<vector<float>> concatenated(SEQ_LEN, vector<float>(EMBED_DIM));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int h = 0; h < NUM_HEADS; h++) {
            #pragma omp simd
            for (int j = 0; j < HEAD_DIM; j++) {
                concatenated[i][h * HEAD_DIM + j] = head_outputs[h][i][j];
            }
        }
    }


    auto Wo = generate_random_matrix(EMBED_DIM, EMBED_DIM);
    return layer_norm(matmul(concatenated, Wo));
}


vector<vector<float>> feedforward_layer(const vector<vector<float>> &input) {
    auto W1 = generate_random_matrix(EMBED_DIM, HIDDEN_DIM);
    auto W2 = generate_random_matrix(HIDDEN_DIM, EMBED_DIM);


    auto hidden = matmul(input, W1);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < hidden.size(); i++) {
        for (int j = 0; j < hidden[0].size(); j++) {
            hidden[i][j] = max(0.0f, hidden[i][j]);
        }
    }

    return layer_norm(matmul(hidden, W2));
}

vector<float> softmax(const vector<float> &logits) {
    vector<float> result(logits.size());

    float max_val = -INFINITY;
    #pragma omp simd reduction(max:max_val)
    for (size_t i = 0; i < logits.size(); i++) {
        max_val = max(max_val, logits[i]);
    }
    
    float sum_exp = 0.0f;
    #pragma omp simd reduction(+:sum_exp)
    for (size_t i = 0; i < logits.size(); i++) {
        result[i] = exp(logits[i] - max_val);
        sum_exp += result[i];
    }

    float scale = 1.0f / sum_exp;
    #pragma omp simd
    for (size_t i = 0; i < logits.size(); i++) {
        result[i] *= scale;
    }
    
    return result;
}

int main() {
    ofstream perf_file("parallel_performance_results.csv");
    perf_file << "Threads,Time,Speedup,ParallelFraction\n";


    vector<int> thread_counts = {1, 2, 4, 6, 8, 10, 12, 16, 20, 32, 64};

    vector<vector<float>> token_embeddings = generate_random_matrix(SEQ_LEN, EMBED_DIM);

    omp_set_num_threads(1);
    auto serial_start = chrono::high_resolution_clock::now();
    vector<vector<float>> attention_output = self_attention(token_embeddings);
    vector<vector<float>> final_output = feedforward_layer(attention_output);
    vector<float> softmax_output = softmax(final_output[SEQ_LEN - 1]);
    auto serial_end = chrono::high_resolution_clock::now();
    double serial_time = chrono::duration<double>(serial_end - serial_start).count();
    cout << "Serial execution time: " << serial_time << " seconds." << endl;

    for (int threads : thread_counts) {
        omp_set_num_threads(threads);
        auto start_time = chrono::high_resolution_clock::now();

        vector<vector<float>> att_out = self_attention(token_embeddings);
        vector<vector<float>> ff_out = feedforward_layer(att_out);
        vector<float> sm_out = softmax(ff_out[SEQ_LEN - 1]);

        auto end_time = chrono::high_resolution_clock::now();
        double elapsed = chrono::duration<double>(end_time - start_time).count();
        double speedup = serial_time / elapsed;
        double parallel_fraction = (threads > 1) ? 
            (1.0 - (1.0 / speedup)) / (1.0 - (1.0 / threads)) : 0.0;

        cout << "Threads: " << threads 
             << " | Time: " << elapsed << " s"
             << " | Speedup: " << speedup 
             << " | Parallel Fraction: " << parallel_fraction << endl;

        perf_file << threads << "," << elapsed << "," << speedup << "," << parallel_fraction << "\n";
    }
    
    perf_file.close();
    cout << "Performance data written to parallel_performance_results.csv" << endl;
    return 0;
}