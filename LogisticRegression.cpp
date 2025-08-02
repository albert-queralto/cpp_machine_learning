#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

using namespace std;

/*
    Sigmoid function that maps the input z to a probability between 0 and 1.

    Formula:
    - $$ \text{sigmoid}(z) = \frac{1}{1 + e^{-z}} $$

    Parameters:
    - z: A double representing the input value.

    Returns:
    - A double representing the probability.
*/
double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

/*
    Computes the log-loss cost function for logistic regression.

    Formula:
    - $$ J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(h(x_i)) + (1 - y_i) \log(1 - h(x_i)) \right] $$

    Parameters:
    - X: A vector<vector<double>> representing the feature matrix.
    - y: A vector<int> representing the binary labels (0 or 1).
    - weights: A vector<double> representing the weight vector.
    - bias: A double representing the bias term.

    Returns:
    - A double representing the average cost.
*/
double computeCost(
    const vector<vector<double>>& X, 
    const vector<int>& y, 
    const vector<double>& weights,
    double bias
) {
    double cost = 0.0;
    int m = X.size();

    for (int i = 0; i < m; ++i) {
        double z = bias;
        for (size_t j = 0; j < X[i].size(); ++j) {
            z += weights[j] * X[i][j];
        }
        double h = sigmoid(z);
        cost += -y[i] * log(h) - (1 - y[i]) * log(1 - h);
    }
    return cost / m;
}

/*
    Optimizes the weights and bias using Gradient Descent to minimize the cost function.

    Steos:
    1. Computes the gradients for weights and bias:
        $$ \frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (h(x_i) - y_i) x_{ij} $$
        $$ \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (h(x_i) - y_i) $$
    2. Update weights and bias:
        $$ w_j = w_j - \alpha \cdot \frac{\partial J}{\partial w_j} $$
        $$ b = b - \alpha \cdot \frac{\partial J}{\partial b} $$

    Parameters:
    - X: A vector<vector<double>> representing the feature matrix.
    - y: A vector<int> representing the binary labels (0 or 1).
    - weights: A vector<double> representing the weight vector (updated in-place).
    - bias: A double representing the bias term (updated in-place).
    - learning_rate: A double representing the step size for gradient descent.
    - iterations: An int representing the number of iterations.
*/
void gradientDescent(
    const vector<vector<double>>& X,
    const vector<int>& y,
    vector<double>& weights,
    double& bias,
    double learning_rate,
    int iterations
) {
    int m = X.size();
    size_t n = X[0].size();

    for (int iter = 0; iter < iterations; ++iter) {
        vector<double> weight_gradients(n, 0.0);
        double bias_gradient = 0.0;

        for (int i = 0; i < m; ++i) {
            double z = bias;
            for (size_t j = 0; j < n; ++j) {
                z += weights[j] * X[i][j];
            }
            double h = sigmoid(z);
            double error = h - y[i];

            for (size_t j = 0; j < n; ++j) {
                weight_gradients[j] += error * X[i][j];
            }
            bias_gradient += error;
        }

        for (size_t j = 0; j < n; ++j) {
            weights[j] -= (weight_gradients[j] / m) * learning_rate;
        }
        bias -= (bias_gradient / m) * learning_rate;

        if (iter % 100 == 0) {
            cout << "Iteration " << iter << " | Cost: " << computeCost(X, y, weights, bias) << endl;
        }
    }
}

/*
    Predicts binary labels for the given feature matrix using the trained weights and bias.

    Steps:
    1. Compute the probability for each sample: $$ h(x) = \text{sigmoid}(w \cdot x + b) $$
    2. Assign a label based on the threshold: $$ \text{label} = \begin{cases} 1 & \text{if } h(x) \geq 0.5 \\ 0 & \text{otherwise} \end{cases} $$

    Parameters:
    - X: A vector<vector<double>> representing the feature matrix.
    - weights: A vector<double> representing the trained weight vector.
    - bias: A double representing the trained bias term.

    Returns:
    - A vector<int> containing the predicted binary labels (0 or 1).
*/
vector<int> predict(
    const vector<vector<double>>& X,
    const vector<double>& weights,
    double bias
) {
    vector<int> predictions;
    for (const auto& sample : X) {
        double z = bias;
        for (size_t j = 0; j < sample.size(); ++j) {
            z += weights[j] * sample[j];
        }
        double h = sigmoid(z);
        predictions.push_back(h >= 0.5 ? 1 : 0);
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
    vector<int> y = {0, 0, 1, 1, 1};

    // Initialize parameters
    vector<double> weights(X[0].size(), 0.0);
    double bias = 0.0;
    double learning_rate = 0.01;
    int iterations = 10000;

    // Train the model
    gradientDescent(X, y, weights, bias, learning_rate, iterations);

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
