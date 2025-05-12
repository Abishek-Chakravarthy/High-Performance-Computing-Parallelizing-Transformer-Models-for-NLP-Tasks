#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include<likwid.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
using namespace std;

const int SEQ_LEN = 1024;
const int EMBED_DIM = 1024;
const int HIDDEN_DIM = 2048;
const int NUM_HEADS = 8;
const int HEAD_DIM = EMBED_DIM / NUM_HEADS;

//const int SEQ_LEN = 100000;
//const int EMBED_DIM = 512;
//const int HIDDEN_DIM = 2048;
//const int NUM_HEADS = 16;
//const int HEAD_DIM = EMBED_DIM / NUM_HEADS;

//LIKWID_MARKER_START("PrintVector");
void print_vector(const vector<float> &vec, const string &name)
{
    cout << name << ": [ ";
    for (float val : vec)
    {
        cout << fixed << setprecision(4) << val << " ";
    }
    cout << "]\n";
}
//LIKWID_MARKER_STOP("PrintVector");

//LIKWID_MARKER_START("PrintMatrix");

void print_matrix(const vector<vector<float>> &matrix, const string &name, int rows = 10, int cols = 10)
{
    cout << name << " (Showing first " << min(rows, (int)matrix.size()) << "x" << min(cols, (int)matrix[0].size()) << " values):\n";
    for (int i = 0; i < min(rows, (int)matrix.size()); i++)
    {
        cout << "[ ";
        for (int j = 0; j < min(cols, (int)matrix[i].size()); j++)
        {
            cout << fixed << setprecision(6) << matrix[i][j] << " ";
        }
        if (cols < (int)matrix[i].size())
            cout << "... ]\n";
        else
            cout << "]\n";
    }
    cout << "...\n"; 
}
//LIKWID_MARKER_STOP("PrintMatrix");

//LIKWID_MARKER_START("GenerateRandomMatrix");

// Function to generate a random matrix
vector<vector<float>> generate_random_matrix(int rows, int cols)
{
    vector<vector<float>> matrix(rows, vector<float>(cols));
    random_device rd;
    mt19937 gen(rd());
    float scale = sqrt(2.0 / (rows + cols));
    normal_distribution<float> dist(0.0, scale);

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            matrix[i][j] = dist(gen);

    return matrix;
}

//LIKWID_MARKER_STOP("GenerateRandomMatrix");

//LIKWID_MARKER_START("LoadEmbedding");

// Function to load real BERT embeddings from a CSV file
vector<vector<float>> load_bert_embeddings(const string &filename)
{
    vector<vector<float>> embeddings;
    ifstream file(filename);
    string line;

    while (getline(file, line))
    {
        stringstream ss(line);
        string value;
        vector<float> row;

        while (getline(ss, value, ','))
        {
            row.push_back(stof(value));
        }

        embeddings.push_back(row);
    }

    return embeddings;
}
//LIKWID_MARKER_STOP("LoadEmbedding");

//LIKWID_MARKER_START("MatrixMultiplication");

// Matrix Multiplication Function
vector<vector<float>> matmul(const vector<vector<float>> &A, const vector<vector<float>> &B)
{
    int M = A.size();
    int K = A[0].size();
    int N = B[0].size();

    if (K != B.size())
    {
        throw runtime_error("Matrix dimension mismatch in multiplication");
    }

    vector<vector<float>> C(M, vector<float>(N, 0.0f));
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
            {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    return C;
}
//LIKWID_MARKER_STOP("MatrixMultiplication");

//LIKWID_MARKER_START("LayerNorm");

vector<vector<float>> layer_norm(const vector<vector<float>> &input, float eps = 1e-5)
{
    int rows = input.size();
    int cols = input[0].size();
    vector<vector<float>> normalized(rows, vector<float>(cols));

    for (int i = 0; i < rows; i++)
    {
        float mean = 0.0f;
        for (int j = 0; j < cols; j++)
        {
            mean += input[i][j];
        }
        mean /= cols;

        float var = 0.0f;
        for (int j = 0; j < cols; j++)
        {
            var += (input[i][j] - mean) * (input[i][j] - mean);
        }
        var /= cols;

        for (int j = 0; j < cols; j++)
        {
            normalized[i][j] = (input[i][j] - mean) / sqrt(var + eps);
        }
    }
    return normalized;
}
//LIKWID_MARKER_STOP("LayerNorm");

//LIKWID_MARKER_START("SelfAttention");

// Self-Attention Mechanism
vector<vector<float>> self_attention(const vector<vector<float>> &input)
{
   
    vector<vector<vector<float>>> head_outputs(NUM_HEADS);
    float scale_factor = 1.0f / sqrt(static_cast<float>(HEAD_DIM));

    for (int h = 0; h < NUM_HEADS; h++)
    {
        // Generate weights for this head
        auto Wq = generate_random_matrix(EMBED_DIM, HEAD_DIM);
        auto Wk = generate_random_matrix(EMBED_DIM, HEAD_DIM);
        auto Wv = generate_random_matrix(EMBED_DIM, HEAD_DIM);

        // Project input to Q, K, V for this head
        auto Q = matmul(input, Wq);
        auto K = matmul(input, Wk);
        auto V = matmul(input, Wv);

        // Calculate attention scores
        vector<vector<float>> scores(SEQ_LEN, vector<float>(SEQ_LEN, 0.0f));
        for (int i = 0; i < SEQ_LEN; i++)
        {
            for (int j = 0; j < SEQ_LEN; j++)
            {
                float dot_product = 0.0f;
                for (int k = 0; k < HEAD_DIM; k++)
                {
                    dot_product += Q[i][k] * K[j][k];
                }
                scores[i][j] = dot_product * scale_factor;
            }
        }

        // Apply softmax to attention scores
        for (int i = 0; i < SEQ_LEN; i++)
        {
            // Find max for numerical stability
            float max_val = scores[i][0];
            for (int j = 1; j < SEQ_LEN; j++)
            {
                max_val = max(max_val, scores[i][j]);
            }

            // Calculate exponentials and sum
            vector<float> exp_scores(SEQ_LEN);
            float sum_exp = 0.0f;
            for (int j = 0; j < SEQ_LEN; j++)
            {
                exp_scores[j] = exp(scores[i][j] - max_val);
                sum_exp += exp_scores[j];
            }

            // Normalize
            for (int j = 0; j < SEQ_LEN; j++)
            {
                scores[i][j] = exp_scores[j] / sum_exp;
            }
        }

        // Apply attention scores to values
        head_outputs[h] = matmul(scores, V);
    }

    // Concatenate all head outputs
    vector<vector<float>> concatenated(SEQ_LEN, vector<float>(EMBED_DIM));
    for (int i = 0; i < SEQ_LEN; i++)
    {
        for (int h = 0; h < NUM_HEADS; h++)
        {
            for (int j = 0; j < HEAD_DIM; j++)
            {
                concatenated[i][h * HEAD_DIM + j] = head_outputs[h][i][j];
            }
        }
    }

    // Apply output projection
    auto Wo = generate_random_matrix(EMBED_DIM, EMBED_DIM);
    auto output = matmul(concatenated, Wo);

    // Layer normalization
    return layer_norm(output);
}
//LIKWID_MARKER_STOP("SelfAttention");

//LIKWID_MARKER_START("FeedForward");
// Feedforward Neural Network
vector<vector<float>> feedforward_layer(const vector<vector<float>> &input)
{
    auto W1 = generate_random_matrix(EMBED_DIM, HIDDEN_DIM);
    auto W2 = generate_random_matrix(HIDDEN_DIM, EMBED_DIM);

    // First layer with ReLU
    auto hidden = matmul(input, W1);
    for (auto &row : hidden)
    {
        for (auto &val : row)
        {
            val = max(0.0f, val);
        }
    }

    // Second layer
    auto output = matmul(hidden, W2);

    // Layer normalization
    return layer_norm(output);
}
//LIKWID_MARKER_STOP("FeedForward");

//LIKWID_MARKER_START("Softmax");
// Softmax Function
vector<float> softmax(const vector<float> &logits)
{
    vector<float> result(logits.size());
    float max_val = *max_element(logits.begin(), logits.end()); // Prevent large exponentials
    float sum_exp = 0.0;

    for (float val : logits)
        sum_exp += exp(val - max_val);

    for (size_t i = 0; i < logits.size(); i++)
        result[i] = exp(logits[i] - max_val) / sum_exp;

    return result;
}
//LIKWID_MARKER_STOP("Softmax");

int main()
{
    //likwid_markerInit();
    
    //LIKWID_MARKER_START("Main")

    // Generate Random Token Embeddings or Load Real BERT Embeddings
    vector<vector<float>> token_embeddings = generate_random_matrix(SEQ_LEN, EMBED_DIM);
    // vector<vector<float>> token_embeddings = load_bert_embeddings("bert_embeddings.csv");

    // Print initial embeddings
    print_matrix(token_embeddings, "Token Embeddings");

    // Apply Self-Attention
    vector<vector<float>> attention_output = self_attention(token_embeddings);
    print_matrix(attention_output, "Self-Attention Output");

    // Apply Feedforward Network
    vector<vector<float>> final_output = feedforward_layer(attention_output);
    print_matrix(final_output, "Feedforward Layer Output");

    // Softmax over the last token's output
    vector<float> softmax_output = softmax(final_output[SEQ_LEN - 1]);
    float sum = 0;
    for (float i : softmax_output)
        sum += i;

    print_vector(softmax_output, "Softmax Output");

    cout << "Self-Attention + Feedforward + Softmax applied successfully!" << endl;
    cout << "Sum of all values in the softmax output " << sum << endl;

    //LIKWID_MARKER_STOP("Main");
    
    //likwid_markerClose();
    return 0;
}
