## Decision tree

### Packages used:

- `Numpy`
- `Pandas`

Decision tree classifier from scratch and includes functions for entropy calculation, information gain, building the decision tree, and making predictions.

Here's a breakdown of the key components:

1. **Entropy Calculation (`entropy` function)**:

   - Entropy is a measure of impurity or disorder in a set of data.
   - The `entropy` function calculates the entropy of a given set `s` using the formula: \( \text{Entropy}(S) = - \sum\_{i=1}^{c} p_i \log_2(p_i) \), where \( p_i \) is the proportion of examples in class \( i \) in set \( S \).

2. **Information Gain Calculation (`information_gain` function)**:

   - Information gain measures the reduction in entropy or impurity achieved by splitting a dataset based on a given attribute.
   - The `information_gain` function calculates the information gain of splitting a parent dataset into two child datasets.

3. **Node Class (`Node`)**:

   - Represents a node in the decision tree.
   - Contains attributes such as feature index, threshold, data for left and right child nodes, gain, and value.

4. **Decision Tree Class (`DecisionTree`)**:

   - Implements the decision tree classifier algorithm.
   - Includes methods for entropy calculation, information gain, best split determination, tree building, and prediction.
   - Uses a recursive approach to build the decision tree.
   - Also includes methods for model fitting and prediction.

5. **Model Evaluation**:
   - The code evaluates the decision tree model using the Iris dataset from scikit-learn.
   - It splits the dataset into training and testing sets.
   - Fits the decision tree model to the training data.
   - Makes predictions on the testing data.
   - Calculates the R² score and accuracy of the model using custom functions (`r2_score` and `accuracy`).
