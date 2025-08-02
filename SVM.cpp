#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

using namespace std;

/*
    Computes the hinge loss and regularization term for the SVM model.

    Formula:
    - Hinge Loss: $ \text{Loss} = \frac{1}{m} \sum_{i=1}^{m} \max(0, 1 - y_i (w \cdot x_i + b)) $
    - Regularization: $ \text{Regularization} = \frac{\lambda}{2} ||w||^2 $

    Parameters:
    - X: A vector<vector<double>> representing the feature matrix.
    - y: A vector<int> representing the binary labels (-1 or +1).
    - weights: A vector<double> representing the weight vector.
    - bias: A double representing the bias term.
    - lambda: A double representing the regularization parameter.

    Returns:
    - A double representing the total cost (hinge loss + regularization).
*/
double computeCost(
    const vector<vector<double>>& X,
    const vector<int>& y,
    const vector<double>& weights,
    double bias,
    double lambda
) {
    double cost = 0.0;
    int m = X.size();

    for (int i = 0; i < m; ++i) {
        double margin = y[i] * (bias + inner_product(X[i].begin(), X[i].end(), weights.begin(), 0.0));
        cost += max(0.0, 1 - margin);
    }

    double regularization = 0.5 * lambda * inner_product(weights.begin(), weights.end(), weights.begin(), 0.0);
    return cost / m + regularization;
}

/*
    Optimizes the weights and bias using Gradient Descent to minimize the hinge loss and regularization term.

    Steps:
    1. Compute gradients for weights and bias:
        - Weight Gradient: $$ \frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} \begin{cases} -y_i x_{ij} & \text{if } y_i (w \cdot x_i + b) < 1 \ 0 & \text{otherwise} \end{cases} $$
        - Bias Gradient: $$ \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \begin{cases} -y_i & \text{if } y_i (w \cdot x_i + b) < 1 \ 0 & \text{otherwise} \end{cases} $$
    2. Update weights and bias:
        - Weight Update: $$ w_j = w_j - \alpha \cdot \left( \frac{\partial J}{\partial w_j} + \lambda w_j \right) $$
        - Bias Update: $$ b = b - \alpha \cdot \frac{\partial J}{\partial b} $$
    
    Parameters:
    - X: A vector<vector<double>> representing the feature matrix.
    - y: A vector<int> representing the binary labels (-1 or +1).
    - weights: A vector<double> representing the bias term (updated in-place).
    - bias: A double representing the bias term (updated in-place).
    - learning_rate: A double representing the step size for gradient descent.
    - iterations: An int representing the number of iterations.
    - lambda: A double representing the regularization parameter.
*/
void gradientDescent(
    const vector<vector<double>>& X,
    const vector<int>& y,
    vector<double>& weights,
    double& bias,
    double learning_rate,
    int iterations,
    double lambda
) {
    int m = X.size();
    size_t n = X[0].size();

    for (int iter = 0; iter < iterations; ++iter) {
        vector<double> weight_gradients(n, 0.0);
        double bias_gradient = 0.0;

        for (int i = 0; i < m; ++i) {
            double margin = y[i] * (bias + inner_product(X[i].begin(), X[i].end(), weights.begin(), 0.0));
            if (margin < 1) {
                for (size_t j = 0; j < n; ++j) {
                    weight_gradients[j] += -y[i] * X[i][j];
                }
                bias_gradient += -y[i];
            }
        }

        for (size_t j = 0; j < n; ++j) {
            weights[j] -= learning_rate * (weight_gradients[j] / m + lambda + weights[j]);
        }
        bias -= learning_rate * bias_gradient / m;

        if (iter % 100 == 0) {
            cout << "Iteration " << iter << " | Cost: " << computeCost(X, y, weights, bias, lambda) << endl;
        }
    }
}

/*
    Predicts binary labels for the given feature matrix using the trained weights and bias.

    Steps:
    1. Compute the margin for each sample: $$ \text{margin} = b + w \cdot x_i $$
    2. Assign label based on margin: $$ \text{label} = \begin{cases} +1 & \text{if } \text{margin} \geq 0 \\ -1 & \text{otherwise} \end{cases} $$

    Parameters:
    - X: A vector<vector<double>> representing the feature matrix.
    - weights: A vector<double> representing the weight vector.
    - bias: A double representing the bias term.

    Returns:
    - A vector<int> containing the predicted labels (-1 or +1).
*/
vector<int> predict(
    const vector<vector<double>>& X,
    const vector<double>& weights,
    double bias
) {
    vector<int> predictions;
    for (const auto& sample : X) {
        double margin = bias + inner_product(sample.begin(), sample.end(), weights.begin(), 0.0);
        predictions.push_back(margin >= 0 ? 1 : -1);
    }
    return predictions;
}


int main() {
    // Example dataset
    vector<vector<double>> X = {
        {1.0, 2.0},
        {2.0, 3.0},
        {3.0, 4.0},
        {4.0, 5.0},
        {5.0, 6.0}
    };
    vector<int> y = {-1, -1, 1, 1, 1}; // Labels must be -1 or +1 for SVM

    // Initialize parameters
    vector<double> weights(X[0].size(), 0.0);
    double bias = 0.0;
    double learning_rate = 0.01;
    int iterations = 10000;
    double lambda = 0.1; // Regularization parameter

    // Train the model
    gradientDescent(X, y, weights, bias, learning_rate, iterations, lambda);

    // Predict on the training data
    vector<int> predictions = predict(X, weights, bias);

    // Output predictions
    cout << "Predictions: ";
    for (int pred : predictions) {
        cout << pred << " ";
    }
    cout << endl;

    return 0;
}
