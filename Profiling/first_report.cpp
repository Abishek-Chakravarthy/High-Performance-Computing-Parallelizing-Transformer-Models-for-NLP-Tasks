#include <iostream>
#include <vector>
#include <cmath>
#include <random>

using namespace std;

// Simulate a token embedding
vector<vector<float>> generateTokenEmbeddings(int numTokens, int embeddingSize) {
    vector<vector<float>> embeddings(numTokens, vector<float>(embeddingSize, 0.0));
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dist(0.0, 1.0);

    for (auto& token : embeddings)
        for (auto& val : token)
            val = dist(gen);

    return embeddings;
}

// Simulate scaled dot-product attention
vector<vector<float>> selfAttention(vector<vector<float>>& Q, vector<vector<float>>& K, vector<vector<float>>& V) {
    int seqLen = Q.size();
    int depth = Q[0].size();

    // Compute QK^T
    vector<vector<float>> attention(seqLen, vector<float>(seqLen, 0.0));
    for (int i = 0; i < seqLen; ++i) {
        for (int j = 0; j < seqLen; ++j) {
            float dotProduct = 0.0;
            for (int k = 0; k < depth; ++k)
                dotProduct += Q[i][k] * K[j][k];
            attention[i][j] = dotProduct / sqrt(depth);
        }
    }

    // Apply softmax
    for (int i = 0; i < seqLen; ++i) {
        float sumExp = 0.0;
        for (int j = 0; j < seqLen; ++j)
            sumExp += exp(attention[i][j]);
        for (int j = 0; j < seqLen; ++j)
            attention[i][j] = exp(attention[i][j]) / sumExp;
    }

    // Compute attention-weighted values
    vector<vector<float>> output(seqLen, vector<float>(depth, 0.0));
    for (int i = 0; i < seqLen; ++i) {
        for (int j = 0; j < seqLen; ++j) {
            for (int k = 0; k < depth; ++k) {
                output[i][k] += attention[i][j] * V[j][k];
            }
        }
    }

    return output;
}

// Feed-forward network
vector<vector<float>> feedForward(vector<vector<float>>& input, int hiddenSize) {
    int seqLen = input.size();
    int depth = input[0].size();

    // Initialize weights and biases
    vector<vector<float>> W1(depth, vector<float>(hiddenSize, 0.1));
    vector<vector<float>> W2(hiddenSize, vector<float>(depth, 0.1));
    vector<float> b1(hiddenSize, 0.1);
    vector<float> b2(depth, 0.1);

    // First layer (ReLU activation)
    vector<vector<float>> hidden(seqLen, vector<float>(hiddenSize, 0.0));
    for (int i = 0; i < seqLen; ++i) {
        for (int j = 0; j < hiddenSize; ++j) {
            for (int k = 0; k < depth; ++k)
                hidden[i][j] += input[i][k] * W1[k][j];
            hidden[i][j] += b1[j];
            hidden[i][j] = max(0.0f, hidden[i][j]); // ReLU
        }
    }

    // Second layer
    vector<vector<float>> output(seqLen, vector<float>(depth, 0.0));
    for (int i = 0; i < seqLen; ++i) {
        for (int j = 0; j < depth; ++j) {
            for (int k = 0; k < hiddenSize; ++k)
                output[i][j] += hidden[i][k] * W2[k][j];
            output[i][j] += b2[j];
        }
    }

    return output;
}

int main() {
    // Parameters
    const int seqLen = 10;       // Number of tokens
    const int embeddingSize = 64; // Embedding size
    const int hiddenSize = 128;  // Hidden layer size

    // Generate token embeddings
    auto embeddings = generateTokenEmbeddings(seqLen, embeddingSize);

    // Initialize Q, K, V matrices (for simplicity, using embeddings as all three)
    auto Q = embeddings;
    auto K = embeddings;
    auto V = embeddings;

    // Self-attention
    auto attentionOutput = selfAttention(Q, K, V);

    // Feed-forward network
    auto finalOutput = feedForward(attentionOutput, hiddenSize);

    // Output result
    cout << "Final output dimensions: " << finalOutput.size() << "x" << finalOutput[0].size() << endl;

    return 0;
}
