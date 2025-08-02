#include <iostream>
#include <vector>
#include <numeric>

using namespace std;

// Function to calculate the mean of a vector
double calculateMean(const vector<double>& values) {
    if (values.empty()) return 0.0;
    return accumulate(values.begin(), values.end(), 0.0) / values.size();
}

/* 
    The LinearRegression function calculates the weight (slope) and bias 
    (intercept) for a simple linear regression model using the Least Squares 
    Method. It determines the best-fit line for the given input data (x and y).

    Parameters:
    - x: A vector<double> representing the independent variable (input features).
    - y: A vector<double> representing the dependent variable (target values).
    - weight: A double reference where the calculated slope of the regression line will be stored.
    - bias: A double reference where the calculated intercept of the regression line will be stored.

*/
void LinearRegression(
    const vector<double>& x, 
    const vector<double>& y, 
    double& weight, 
    double& bias
) {
    double x_mean = calculateMean(x);
    double y_mean = calculateMean(y);

    double numerator = 0.0;
    double denominator = 0.0;

    // Calculation of the least square estimates (weight and bias)
    for (size_t i = 0; i < x.size(); ++i) {
        numerator += (x[i] - x_mean) * (y[i] - y_mean);
        denominator += (x[i] - x_mean) * (x[i] - x_mean);
    }

    weight = numerator / denominator;
    bias = y_mean - weight * x_mean;
}

int main() {
    vector<double> x = {1, 2, 3, 4, 5};
    vector<double> y = {2, 4, 6, 8, 10};

    double weight = 0.0;
    double bias = 0.0;

    LinearRegression(x, y, weight, bias);

    cout << "Weight (Slope): " << weight << endl;
    cout << "Bias (Intercept): " << bias << endl;

    return 0;
}
