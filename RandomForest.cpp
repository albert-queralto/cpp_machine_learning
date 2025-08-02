#include <iostream>
#include <vector>
#include <map>
#include <limits>
#include <cmath>
#include <algorithm>
#include <random>

using namespace std;

/*
    Represents a node in the decision tree.
    Contains:
    - feature_index: The feature used for splitting.
    - threshold: The threshold value for the split.
    - label: The class label for leaf nodes.
    - left and right: Pointers to child nodes.
*/
struct TreeNode {
    int feature_index; // Index of the feature used for splitting
    double threshold;  // Threshold value for the split
    string label;      // Label for leaf nodes
    TreeNode* left;    // Pointer to the left child
    TreeNode* right;   // Pointer to the right child

    TreeNode() : feature_index(-1), threshold(0.0), label(""), left(nullptr), right(nullptr) {}
};


/* 
    Encapsulates all functionality related to the decision tree.

    Attributes:
    - TreeNode* root: Pointer to the root node of the tree.
*/
class DecisionTree {
    public:
        TreeNode* root;

        DecisionTree() : root(nullptr) {}

        /*
            Calculates the Gini Impurity for a given set of labels.
            Formula: [ Gini = 1 - \sum_{i} P_{i²}] where P_i is the probability of each class.
        */
        double calculateGini(const vector<string>& labels) {
            map<string, int> label_count;
            for (const string& label : labels) {
                label_count[label]++;
            }

            double gini = 1.0;
            int total = labels.size();
            for (const auto& pair : label_count) {
                double prob = static_cast<double>(pair.second) / total;
                gini -= prob * prob;
            }
            return gini;
        }

        /*
            Splits the dataset into two subsets based on a feature and threshold.

            Parameters:
            - data: The feature matrix.
            - labels: The corresponding labels.
            - feature_index: The feature used for splitting.
            - threshold: The threshold value for the split.
            - left: Boolean indicating whether to split to the left (<= threshold) or right (> threshold).

            Returns:
            - A pair containing the split data and labels.
        */
        pair<vector<vector<double>>, vector<string>> splitDataset(
            const vector<vector<double>>& data,
            const vector<string>& labels,
            int feature_index,
            double threshold,
            bool left
        ) {
            vector<vector<double>> split_data;
            vector<string> split_labels;

            for (size_t i = 0; i < data.size(); ++i) {
                if (
                    (left && data[i][feature_index] <= threshold) || 
                    (!left && data[i][feature_index] > threshold)
                ) {
                        split_data.push_back(data[i]);
                        split_labels.push_back(labels[i]);
                }
            }
            return {split_data, split_labels};
        }

        /* 
            Recursive function to build the decision tree.

            Steps:
            - If all labels are the same, create a leaf node.
            - Otherwise, find the best feature and threshold to split the data using Gini impurity.
            - Create a node and recursively build child nodes.
        */
        TreeNode* buildTree(
            const vector<vector<double>>& data,
            const vector<string>& labels
        ) {
            if (labels.empty()) return nullptr;

            // Check if all labels are the same
            bool all_same = true;
            for (size_t i = 1; i < labels.size(); ++i) {
                if (labels[i] != labels[0]) {
                    all_same = false;
                    break;
                }
            }

            if (all_same) {
                TreeNode* leaf = new TreeNode();
                leaf -> label = labels[0];
                return leaf;
            }

            // Find the best split
            int best_feature = -1;
            double best_threshold = 0.0;
            double best_gini = numeric_limits<double>::max();

            for (size_t feature_index = 0; feature_index < data[0].size(); ++feature_index) {
                for (const auto& row: data) {
                    double threshold = row[feature_index];
                    auto left_split = splitDataset(data, labels, feature_index, threshold, true);
                    auto right_split = splitDataset(data, labels, feature_index, threshold, false);

                    double gini_left = calculateGini(left_split.second);
                    double gini_right = calculateGini(right_split.second);

                    double weighted_gini = (
                        left_split.second.size() * gini_left + 
                        right_split.second.size() * gini_right
                    ) / labels.size();

                    if (weighted_gini < best_gini) {
                        best_gini = weighted_gini;
                        best_feature = feature_index;
                        best_threshold = threshold;
                    }
                }
            }

            // Create the current node
            TreeNode* node = new TreeNode();
            node -> feature_index = best_feature;
            node -> threshold = best_threshold;

            // Split the dataset and build child nodes
            auto left_split = splitDataset(data, labels, best_feature, best_threshold, true);
            auto right_split = splitDataset(data, labels, best_feature, best_threshold, false);

            node -> left = buildTree(left_split.first, left_split.second);
            node -> right = buildTree(right_split.first, right_split.second);

            return node;
        }

        /*
            Builds the decision tree using the buildTree method.

            Parameters:
            - data: The feature matrix.
            - labels: The corresponding labels.
        */
        void train(const vector<vector<double>>& data, const vector<string>& labels) {
            root = buildTree(data, labels);
        }

        /*
            Traverses the tree to predict the label for a given sample.

            Parameters:
            - node: The current node being traversed.
            - sample: The feature vector of the sample.

            Returns:
            - The predicted label.
        */
        string predict(TreeNode* node, const vector<double>& sample) const {
            if (!node -> left && !node -> right) {
                return node -> label;
            }
            if (sample[node -> feature_index] <= node -> threshold) {
                return predict(node -> left, sample);
            } else {
                return predict(node -> right, sample);
            }
        }

        string predict(const vector<double>& sample) const {
            return predict(root, sample);
        }
};


/*
    Encapsulates the functionality of the Random Forest algorithm.

    Attributes:
    - trees: A vector of DecisionTree objects representing the individual trees in the forest.
    - num_trees: The number of trees in the forest.
    - sample_fraction: The fraction of the dataset used for training each tree.
*/
class RandomForest {
    private:
        vector<DecisionTree> trees;
        int num_trees;
        double sample_fraction;
    public:
        RandomForest(int num_trees, double sample_fraction) : num_trees(num_trees), sample_fraction(sample_fraction) {}

        /*
            Trains the Random Forest by creating multiple decision trees.

            Steps:
            1. Perform boostrap sampling to create subsets of the data for each tree.
            2. Train a DecisionTree on each subset.
            3. Store the trained trees in the trees vector.
        */
        void train(const vector<vector<double>>& data, const vector<string>& labels) {
            random_device rd;
            mt19937 gen(rd());
            uniform_int_distribution<> dist(0, data.size() - 1);

            for (int i = 0; i < num_trees; ++i) {
                vector<vector<double>> sampled_data;
                vector<string> sampled_labels;

                int sample_size = static_cast<int>(data.size() * sample_fraction);
                for (int j = 0; j < sample_size; ++j) {
                    int index = dist(gen);
                    sampled_data.push_back(data[index]);
                    sampled_labels.push_back(labels[index]);
                }

                DecisionTree tree;
                tree.train(sampled_data, sampled_labels);
                trees.push_back(tree);

            }
        }

        /*
            Aggregates predictions from all trees to make a final prediction.

            Steps:
            1. Each tree predicts the label for the input sample.
            2. Count votes for each label.
            3.  Return the label with the highest votes (majority voting).
        */
        string predict(const vector<double>& sample) {
            map<string, int> votes;

            for (const auto& tree : trees) {
                string prediction = tree.predict(sample);
                votes[prediction]++;
            }

            string majority_label;
            int max_votes = 0;
            for (const auto& vote : votes) {
                if (vote.second > max_votes) {
                    max_votes = vote.second;
                    majority_label = vote.first;
                }
            }

            return majority_label;
        }
};  


int main() {
    // Example dataset
    vector<vector<double>> data = {
        {2.0, 3.0},
        {1.0, 5.0},
        {3.0, 2.0},
        {5.0, 4.0},
        {4.0, 1.0}
    };
    vector<string> labels = {"A", "A", "B", "B", "B"};

    // Create and train the random forest
    RandomForest forest(5, 0.8); // 5 trees, 80% of data sampled for each tree
    forest.train(data, labels);

    // Predict for a new sample
    vector<double> sample = {3.0, 3.0};
    cout << "Predicted Label: " << forest.predict(sample) << endl;

    return 0;
}
