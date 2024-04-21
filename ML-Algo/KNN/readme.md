# Name: K-Nearest Neighbors

Packages used:

- `Numpy`
- `Pandas`
- `sklearn`

#### 📌 The k-nearest neighbors algorithm, also known as KNN or k-NN, is a non-parametric, supervised learning classifier, which uses proximity to make classifications or predictions about the grouping of an individual data point.

#### 📌 Goal of this project is to predict if the customer will purchase an iPhone or not given their gender, age and salary.

## Steps:

### 1: Euclidian distance

```python
from math import sqrt
def euclidean_distance(x_test, x_train):
    distance = 0
    for i in range(len(x_test)-1):
        distance += (x_test[i]-x_train[i])**2
    return sqrt(distance)
```

This code defines a function called `euclidean_distance` that calculates the Euclidean distance between two points represented by `x_test` and `x_train`.

To calculate the Euclidean distance, the code first imports the `sqrt` function from the `math` module. The `sqrt` function is used to calculate the square root of a number.

The function initializes a variable called distance to 0. It then iterates over the indices of the `x_test` and `x_train` lists using a for loop.

Inside the loop, the code calculates the squared difference between the corresponding elements of `x_test` and `x_train` using the expression `(x_test[i]-x_train[i])\*\*2`. The squared differences are then added to the distance variable.

After the loop, the code returns the square root of the distance using the sqrt function. This gives us the Euclidean distance between the two points.

#### What is Euclidean distance?

The Euclidean distance is a measure of the straight-line distance between two points in a Euclidean space. It is commonly used in various fields, including mathematics, physics, and computer science.

In computer science and data analysis, the Euclidean distance is often used as a similarity metric to quantify the dissimilarity between two data points. It is particularly useful in clustering algorithms, such as k-means, where it helps determine the distance between data points and cluster centroids.

For example, let's say we have a dataset of points in a two-dimensional space, and we want to find the distance between two points A(x1, y1) and B(x2, y2). The Euclidean distance between A and B can be calculated using the following formula:

`distance = sqrt((x2 - x1)^2 + (y2 - y1)^2)`
By calculating the Euclidean distance, we can determine how similar or dissimilar two points are in terms of their spatial location. This information can be used for various purposes, such as clustering, classification, and anomaly detection.

### Step 2. Getting the nearest neighbours

```python
def get_neighbors(x_test, x_train, num_neighbors):
    distances = []
    data = []
    for i in x_train:
        distances.append(euclidean_distance(x_test,i))
        data.append(i)
    distances = np.array(distances)
    data = np.array(data)
    sort_indexes = distances.argsort()             #argsort() function returns indices by sorting distances data in ascending order
    data = data[sort_indexes]                      #modifying our data based on sorted indices, so that we can get the nearest neightbours
    return data[:num_neighbors]
```

This Python function get_neighbors is used to find the nearest neighbors for a given test instance from the training dataset. Here's how it works:

- It takes three parameters: x_test (the test instance), x_train (the training dataset), and num_neighbors (the number of nearest neighbors to return).

- It initializes two empty lists, distances and data.

- It then iterates over each instance i in the training dataset x_train.

- For each instance i, it calculates the Euclidean distance between x_test and i using the euclidean_distance function, and appends this distance to the distances list. It also appends the instance i to the data list.

- It converts distances and data to numpy arrays for efficient computation.

- It then sorts the distances array in ascending order and gets the indices of the sorted elements using the argsort function. This gives the indices of the instances in data in the order of increasing distance from x_test.

- It reorders data based on these sorted indices, so that the instances in data are now in the order of increasing distance from x_test.

- Finally, it returns the first num_neighbors instances from data, which are the num_neighbors nearest neighbors of x_test in the training dataset.

### Step 3. Predicting the classifier of which our new data point belongs to.

```python
def prediction(x_test, x_train, num_neighbors):
    classes = []
    neighbors = get_neighbors(x_test, x_train, num_neighbors)
    for i in neighbors:
        classes.append(i[-1])
    predicted = max(classes, key=classes.count)              #taking the most repeated class
    return predicted
```

This Python function prediction is used to predict the class of a given test instance based on the classes of its nearest neighbors in the training dataset. Here's how it works:

- It takes three parameters: x_test (the test instance), x_train (the training dataset), and num_neighbors (the number of nearest neighbors to consider).

- It initializes an empty list classes.

- It calls the get_neighbors function to get the num_neighbors nearest neighbors of x_test in x_train.

- It then iterates over each neighbor.

- For each neighbor, it appends the class of the neighbor (assumed to be the last element of the neighbor instance) to the classes list.

- It then finds the class that occurs most frequently in classes using the max function with classes.count as the key function. This is the predicted class of x_test.

Finally, it returns the predicted class.
