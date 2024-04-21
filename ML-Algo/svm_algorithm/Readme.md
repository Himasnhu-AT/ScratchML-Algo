# Support Vector Machine (SVM)

### Package Used:

- `Numpy`
- `Matplotlib`
- `from sklearn.datasets import make_classification`

## Brief explanation:

This code implements a Support Vector Machine (SVM) for binary classification from scratch. Here's a breakdown of what it does:

1. **Data Generation and Visualization**:

   - It generates a synthetic dataset using `make_classification` from scikit-learn, consisting of two classes with two informative features.
   - The data is visualized using a scatter plot.

2. **SVM Class**:

   - The `Svm` class implements the SVM algorithm.
   - It has methods for fitting the model (`fit`), calculating the hinge loss (`hinge_loss`), updating weights and biases, and plotting the hyperplane.
   - The `fit` method trains the SVM model using gradient descent. It iterates over the dataset for a specified number of epochs, updating the model parameters (weights and bias) to minimize the hinge loss.
   - The `hinge_loss` method calculates the hinge loss, which is used as the optimization objective in SVMs.
   - The `plotHyperplane` function plots the decision boundary (hyperplane) learned by the SVM model along with the data points.

3. **Training and Visualization**:

   - An instance of the `Svm` class is created, and the `fit` method is called to train the model.
   - The loss curve is plotted to visualize the convergence of the training process.
   - The hyperplane learned by the SVM model is visualized along with the data points.

4. **Effect of Increasing Penalty (Cost)**:
   - The effect of increasing the penalty parameter `c` on the loss function and the decision boundary is demonstrated.
   - Another SVM model is trained with a higher penalty parameter (`c=1000`), and the loss curve and decision boundary are visualized.

Overall, this code provides a basic implementation of an SVM for binary classification and demonstrates its usage on a synthetic dataset. It also illustrates the impact of the penalty parameter on the model's performance and decision boundary.
