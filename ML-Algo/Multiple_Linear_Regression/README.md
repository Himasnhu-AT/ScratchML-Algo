## Multiple Linear Regression

### Packages used:

`Numpy` `Pandas`

### About Linear regression:

1. Linear Regression is a machine learning algorithm.

2. It attempts to model the relationship between TWO variables by fitting
   a ”best-fit” line to the observed data points where the ”best-fit” line has
   the minimum sum of the squares of the vertical distance from each data
   point to the ”best-fit” line.

3. The method of minimizing the sum of the squares of the vertical distance
   from each data point to the line is known as the method of least-squares.

4. The variables in Linear Regression is known as dependent variable and
   independent variable. The idea is to derive the independent variable using
   the dependent variable.

5. In Multiple Linear Regression, there are more than one dependent variable
   and exactly one independent variable.

### What we are doing:

batch gradient descent algorithm for linear regression. Here's a step-by-step explanation:

- It first sets up some hyperparameters such as batch size, learning rate, number of epochs, and the path to the dataset.

- The dataset is loaded into a pandas DataFrame. The features (total_X) and target variable (total_y) are extracted.

- A column of ones is added to the features matrix to account for the bias term in the linear regression equation.

- The dataset is split into a training set and a testing set. The split is done such that 20% of the data is reserved for testing.

- The parameters of the model (theta) are initialized randomly.

- Two functions are defined for evaluating the model: find_y_hat which predicts the target variable given the features and the model parameters, and testing_with_MSE and testing_with_R2 which calculate the Mean Squared Error and R-squared score of the model respectively.

- The model is then trained using batch gradient descent. For each epoch, it calculates the difference between the predicted and actual target variable, computes the loss, and updates the model parameters using the gradient of the loss with respect to the parameters. The loss is printed every 1000 epochs.

- After training, the final model parameters are printed, and the model is evaluated on the testing set using both MSE and R-squared score.

- Finally, it prints the difference between the predicted and actual target variable for the testing set.
