# Machine Learning Algorithms in C++

This repository contains implementations of various machine learning algorithms written in C++. Each algorithm is implemented from scratch and includes detailed explanations of the underlying mathematical concepts.

## Algorithms Implemented

### 1. Decision Tree
A **Decision Tree** is a supervised learning algorithm used for classification and regression tasks. It splits the dataset into subsets based on feature values, creating a tree-like structure. The splits are chosen to minimize impurity (e.g., Gini Impurity or Entropy).

#### Key Concepts:
- **Gini Impurity**:

  $Gini = 1 - \sum_{i} P_i^2$
  
  where $P_i$ is the probability of each class.

- **Recursive Splitting**:
  The dataset is split recursively until all samples in a node belong to the same class or no further splits are possible.

#### Features:
- Handles both classification and regression tasks.
- Easy to interpret and visualize.

---

### 2. Linear Regression
**Linear Regression** is a supervised learning algorithm used for predicting continuous values. It models the relationship between the input features ($x$) and the target variable ($y$) using a linear equation.

#### Key Concepts:
- **Hypothesis Function**:

  $h(x) = w \cdot x + b$
  
  where:
  - $w$: Weight (slope).
  - $b$: Bias (intercept).

- **Least Squares Method**:
  The weights and bias are calculated to minimize the Mean Squared Error (MSE):
  
  $MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - h(x_i))^2$

#### Features:
- Simple and efficient for regression tasks.
- Assumes a linear relationship between features and the target variable.

---

### 3. Logistic Regression
**Logistic Regression** is a supervised learning algorithm used for binary classification tasks. It models the probability of a binary outcome using the sigmoid function.

#### Key Concepts:
- **Hypothesis Function**:
  
  $h(x) = \frac{1}{1 + e^{-(w \cdot x + b)}}$

- **Cost Function**:
  
  $J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(h(x_i)) + (1 - y_i) \log(1 - h(x_i)) \right]$

- **Gradient Descent**:
  Updates weights and bias iteratively to minimize the cost function:
  
  $w = w - \alpha \cdot \frac{\partial J}{\partial w}$
  
  $b = b - \alpha \cdot \frac{\partial J}{\partial b}$

#### Features:
- Suitable for binary classification tasks.
- Outputs probabilities for each class.

---

### 4. Random Forest
**Random Forest** is an ensemble learning algorithm that combines multiple decision trees to improve accuracy and reduce overfitting. It uses bootstrap sampling and random feature selection to create diverse trees.

#### Key Concepts:
- **Bootstrap Sampling**:
  Each tree is trained on a random subset of the data (sampling with replacement).

- **Feature Subset Selection**:
  At each split, a random subset of features is considered.

- **Aggregation**:
  - For classification: Majority voting.
  - For regression: Averaging predictions.

#### Features:
- Robust to overfitting.
- Handles large datasets and high-dimensional data effectively.

---

### 5. Support Vector Machine (SVM)
**Support Vector Machine (SVM)** is a supervised learning algorithm used for binary classification tasks. It finds the optimal hyperplane that separates data points of different classes with the maximum margin.

#### Key Concepts:
- **Hinge Loss**:
  
  $\text{Loss} = \frac{1}{m} \sum_{i=1}^{m} \max(0, 1 - y_i (w \cdot x_i + b))$

- **Regularization**:
  
  $\text{Regularization} = \frac{\lambda}{2} ||w||^2$

- **Gradient Descent**:
  Updates weights and bias iteratively to minimize the hinge loss and regularization term:
  
  $w_j = w_j - \alpha \cdot \left( \frac{\partial J}{\partial w_j} + \lambda w_j \right)$
  
  $b = b - \alpha \cdot \frac{\partial J}{\partial b}$

#### Features:
- Suitable for binary classification tasks.
- Maximizes the margin between classes.
- Can be extended to non-linear classification using kernel functions.
  
---

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/albert-queralto/cpp_machine_learning.git
   cd cpp_machine_learning
   ```
2. Compile the code:
  ```bash
  g++ -o <file_name> <file_name>.cpp
  ```
3. Run the executable:
  ```bash
  ./<file_name>
  ```
## License
This repository is licensed under the MIT License. Feel free to use and modify the code for your projects.

## Contributions
Contributions are welcome! If you have suggestions for improvements or additional algorithms, please open an issue or submit a pull request.

## Author
Created by Albert Queraltó. For any questions or feedback, feel free to reach out.
