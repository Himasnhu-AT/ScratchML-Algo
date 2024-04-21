# Simple Linear Regression

### Packages used:

- `Numpy` : 1.19.5
- `Pandas` : 1.2.4

### Brief explanation:

Creating a Simple Linear Regression to model from scratch to make prediction.

### Steps:

```python
class SimpleLinearRegression:
    def fit(self, X, y):
        #Preprocessing
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)
        data = pd.concat([X,y], axis=1)

        #Calculating mean for both Target and Feature Variable
        meanX = float(X.mean())
        meanY = float(y.mean())

        #Calculating `x-mean` and `y-mean` for each data point
        data['x-mean'] = X - meanX
        data['y-mean'] = y - meanY

        #Also calculating product(`x-mean, y-mean`) and square(x-mean)
        data['mul'] = data['x-mean'] * data['y-mean']
        data['sq'] = np.power(data['x-mean'], 2)

        #Summation of data['mul'] and data['sq']
        sumXY = data['mul'].sum()
        sumX = data['sq'].sum()

        #Calculating the coefficients or weights
        global b1, b0
        b1 = sumXY/sumX
        b0 = meanY - b1 * meanX

        #We will be returning the coefficients/weights from this function
        return b0, b1

    def predict(self, X):
        predList, y = [], []
        #Creating a list for Feature Variable
        for i in range(0, len(X)):
            predList.append(X[i])
        #Making predictions over Feature Variable
        for i in predList:
            itemY = b0 + b1 * i
            y.append(itemY)
        #Returning the predictions list
        return list(y)

    def r2_score(self, y_pred, y_test):
        #Calculating r2 using formula
        r2 = ((1 - np.sum((y_test - y_pred) * 2) / np.sum((y_test - np.mean(y_test)) * 2)) * 100)
        return r2
```

This `SimpleLinearRegression` class is designed to perform linear regression with a fit, predict, and r2_score method. Here's a breakdown:

1. **fit**:

   - This method is used to train the linear regression model.
   - It takes two arguments, `X` (features) and `y` (target).
   - The features `X` and target `y` are converted into pandas DataFrames for easier manipulation.
   - The mean of both the feature variable (`meanX`) and the target variable (`meanY`) is calculated.
   - Deviations of each data point from their respective means (`x-mean` and `y-mean`) are computed.
   - The product of deviations (`mul`) and the square of deviations (`sq`) are calculated.
   - The sum of `mul` and `sq` are computed to obtain `sumXY` and `sumX`, respectively.
   - The coefficients or weights `b0` (intercept) and `b1` (slope) are calculated using the formulas of linear regression (`y = b0 + b1 * x`). These coefficients are stored in global variables `b0` and `b1`.

2. **predict**:

   - This method is used to make predictions using the trained linear regression model.
   - It takes a single argument `X`, which represents the feature variable for which predictions are to be made.
   - It iterates through each value of `X`, calculates the predicted `y` values using the formula `y = b0 + b1 * x`, and appends them to a list `y`.
   - Finally, it returns the list of predicted `y` values.

3. **r2_score**:
   - This method calculates the R² score, which is a measure of how well the linear regression model fits the data.
   - It takes two arguments, `y_pred` (predicted `y` values) and `y_test` (actual `y` values).
   - It calculates the R² score using the formula \(R^2 = 1 - \frac{\sum{(y*{test} - y*{pred})^2}}{\sum{(y*{test} - \bar{y}*{test})^2}} \times 100\).
   - The result is returned as a percentage.
