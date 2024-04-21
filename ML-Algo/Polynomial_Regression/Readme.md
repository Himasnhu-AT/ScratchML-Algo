## Polynomial_Regression

## package Used

- `numpy`
- `matplotLib`

Implemention of a polynomial regression model from scratch. Here's a step-by-step explanation:

- It first generates a dataset of 1000 points, where the target variable y is a quadratic function of the input variable X, with some added noise.

- It defines a loss function, which is the mean squared error between the predicted and actual target variable.

- It defines a function to calculate the gradients of the loss with respect to the model parameters (weights and bias).

- It defines a function to transform the input data by adding features of higher degrees. This is done by appending columns to the input matrix X where each column is X raised to a power specified in the degrees list.

- It defines a training function, which trains the model using batch gradient descent. For each epoch, it calculates the predicted target variable, computes the gradients of the loss, and updates the model parameters. It also calculates the loss for each epoch and appends it to a list.

- It defines a prediction function, which predicts the target variable for given input data using the trained model parameters.

- It trains the model using a batch size of 100, a single degree of 2 (for quadratic regression), 1000 epochs, and a learning rate of 0.01. It then plots the original data and the model's predictions.

- It defines a function to calculate the R-squared score, which is a statistical measure of how close the data are to the fitted regression line. It then calculates the R-squared score for the model's predictions.
