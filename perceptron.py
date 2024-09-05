"""
Binary Classification using the Perceptron with 1D Data

Training Algorithm Steps:
    
1. Get data - features (X), special column/target column (y)

2. Select algorithm --> perceptron

3. Pick m and b randomly

4. Select learning rate

5. Compute the output for a row of data (starting with the first):
    
    z = mx + b
    if z >= 0 then the output (out) is 1 - otherwise -1
    
6. Compare predicted output with the real output
    If the real and predicted outputs are not the same, update m and b using gradient descent

7. Go back to step 5 for the next row of data - exit when m and b stop changing by a certain threshold
"""


import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification


### DATA ###
X,y=make_classification(n_features = 1, n_redundant = 0, n_informative = 1, n_clusters_per_class = 1, class_sep=1.5) # binary classification, returns labels 0, 1
y = np.where(y == 0, -1, 1) # changing class label 0 to -1 because the perceptron requires it
X = X+np.random.randint(-10,10) # shifting the data so it is not centered around the y-axis

### ALGORITHM VARIABLES ###
m = np.random.randint(-10, 10) # setting a random m
b = np.random.randint(-10, 10) # setting a random b

### TRAINING ###
lr = 0.025 # learning rate (smaller takes longer but is higher accuracy)
epochs = 2000 # number of times to loop over the data
m_list = [m] # recording the values of m to graph later
b_list = [b] # recording the values of b to graph later

for n_epochs in range(epochs):
    for i in range(len(X)):
        z = m*X[i]+b
        if z >= 0: # predicting the class of the data point
            out = 1
        else:
            out = -1
        
        
        if out != y[i]: # if there is an error, update m and b
            m = m+lr*X[i]*y[i]
            b = b+lr*y[i]
    m_list.append(m)
    b_list.append(b)


### GRAPHING ###
plt.grid()

# plotting the x-intercept as an X
x_intercept = (-b)/m # in the equation z=mx+b, setting z to 0 gives the equation x = -b/m
plt.scatter(x_intercept, 0, c="black", marker='x', s = 150) # x-intercept

# plotting the x and y axes as black lines
plt.axhline(0, c="black") # x-axis
plt.axvline(0, c='black') # y-axis

# defining the limits of the plot as slightly past the furthest points
plt.xlim(min(X)-1, max(X)+1)
plt.ylim(-2, 2)

# generating the decision boundary
plt.plot([min(X), x_intercept, x_intercept, max(X)], [-1,-1,1,1], c="blue") # drawing the decision boundary
plt.text(x_intercept, 1.1, "Decision Boundary", c="blue") # writing "decision boundary"

# showing the data points with the colors corresponding to the class
plt.scatter(X,np.zeros(len(X)), c=y, cmap="cool", edgecolor="k")

# labelling x-axis
plt.xlabel("Data")

# plotting the intermediate variable "z"
z_plot = [m*min(X) + b, m*max(X) + b]
plt.plot([min(X), max(X)], z_plot)

# plotting the progression of m and b during training
plt.figure()
plt.plot(m_list) # m plot
plt.xlabel("epoch")
plt.title("")
plt.figure()
plt.plot(b_list) # b plot
plt.xlabel("epoch")






