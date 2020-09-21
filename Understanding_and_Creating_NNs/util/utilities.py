"""
Created by : Rafay Khan
Twitter: @RafayAK

Contains a bunch of helper functions
"""


import numpy as np
import matplotlib.pyplot as plt  # import matplotlib for plotting and visualization
import matplotlib


def compute_cost(Y, Y_hat):
    """
    This function computes and returns the Cost and its derivative.
    The is function uses the Squared Error Cost function -> (1/2m)*sum(Y - Y_hat)^.2

    Args:
        Y: labels of data
        Y_hat: Predictions(activations) from a last layer, the output layer

    Returns:
        cost: The Squared Error Cost result
        dY_hat: gradient of Cost w.r.t the Y_hat

    """
    m = Y.shape[1]

    cost = (1 / (2 * m)) * np.sum(np.square(Y - Y_hat))
    cost = np.squeeze(cost)  # remove extraneous dimensions to give just a scalar

    dY_hat = -1 / m * (Y - Y_hat)  # derivative of the squared error cost function

    return cost, dY_hat


def predict(X, Y, Zs, As):
    """
    helper function to predict on data using a neural net model layers

    Args:
        X: Data in shape (features x num_of_examples)
        Y: labels in shape ( label x num_of_examples)
        Zs: All linear layers in form of a list e.g [Z1,Z2,...,Zn]
        As: All Activation layers in form of a list e.g [A1,A2,...,An]
    Returns::
        p: predicted labels
        probas : raw probabilities
        accuracy: the number of correct predictions from total predictions
    """
    m = X.shape[1]
    n = len(Zs)  # number of layers in the neural network
    p = np.zeros((1, m))

    # Forward propagation
    Zs[0].forward(X)
    As[0].forward(Zs[0].Z)
    for i in range(1, n):
        Zs[i].forward(As[i-1].A)
        As[i].forward(Zs[i].Z)
    probas = As[n-1].A

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:  # 0.5 is threshold
            p[0, i] = 1
        else:
            p[0, i] = 0

    # print results
    # print ("predictions: " + str(p))
    # print ("true labels: " + str(y))
    accuracy = np.sum((p == Y) / m)

    return p, probas, accuracy*100


def plot_learning_curve(costs, learning_rate, total_epochs, save=False):
    """
    This function plots the Learning Curve of the model

    Args:
        costs: list of costs recorded during training
        learning_rate: the learning rate during training
        total_epochs: number of epochs the model was trained for
        save: bool flag to save the image or not. Default False
    """
    # plot the cost
    plt.figure()

    steps = int(total_epochs / len(costs))  # the steps at with costs were recorded
    plt.ylabel('Cost')
    plt.xlabel('Iterations ')
    plt.title("Learning rate =" + str(learning_rate))
    plt.plot(np.squeeze(costs))
    locs, labels = plt.xticks()
    plt.xticks(locs[1:-1], tuple(np.array(locs[1:-1], dtype='int')*steps))  # change x labels of the plot
    plt.xticks()
    if save:
        plt.savefig('Cost_Curve.png', bbox_inches='tight')
    plt.show()


def predict_dec(Zs, As, X):
    """
    Used for plotting decision boundary.

    Args:
        Zs: All linear layers in form of a list e.g [Z1,Z2,...,Zn]
        As: All Activation layers in form of a list e.g [A1,A2,...,An]
        X: Data in shape (features x num_of_examples) i.e (K x m), where 'm'=> number of examples
           and "K"=> number of features

    Returns:
        predictions: vector of predictions of our model (red: 0 / green: 1)
    """

    # Predict using forward propagation and a classification threshold of 0.5
    m = X.shape[1]
    n = len(Zs)  # number of layers in the neural network

    # Forward propagation
    Zs[0].forward(X)
    As[0].forward(Zs[0].Z)
    for i in range(1, n):
        Zs[i].forward(As[i - 1].A)
        As[i].forward(Zs[i].Z)
    probas = As[n - 1].A   # output probabilities

    predictions = (probas > 0.5)  # if probability of example > 0.5 => output 1, vice versa
    return predictions


def plot_decision_boundary(model, X, Y, feat_crosses=None, save=False):
    """
    Plots decision boundary

    Args:
        model: neural network layer and activations in lambda function
        X: Data in shape (num_of_examples x features)
        feat_crosses: list of tuples showing which features to cross
        save: flag to save plot image
    """
    # Generate a grid of points between -0.5 and 1.5 with 1000 points in between
    xs = np.linspace(-0.5, 1.5, 1000)
    ys = np.linspace(1.5, -0.5, 1000)
    xx, yy = np.meshgrid(xs, ys) # create data points
    # Predict the function value for the whole grid

    # Z = model(np.c_[xx.ravel(), yy.ravel()])  # => add this for feature cross eg "xx.ravel()*yy.ravel()"

    prediction_data = np.c_[xx.ravel(), yy.ravel()]
    # add feat_crosses if provided
    if feat_crosses:
        for feature in feat_crosses:
            prediction_data = np.c_[prediction_data, prediction_data[:, feature[0]]*prediction_data[:, feature[1]]]

    Z = model(prediction_data)
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.style.use('seaborn-whitegrid')
    plt.contour(xx, yy, Z, cmap='Blues')  # draw a blue colored decision boundary
    plt.title('Decision boundary', size=18)
    plt.xlabel('$x_1$', size=20)
    plt.ylabel('$x_2$', size=20)
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')

    # color map 'cmap' maps 0 labeled data points to red and 1 labeled points to green
    cmap = matplotlib.colors.ListedColormap(["red", "green"], name='from_list', N=None)
    plt.scatter(X[:, 0], X[:, 1], s=200, c=np.squeeze(Y), marker='x', cmap=cmap)  # s-> size of marker

    if save:
        plt.savefig('decision_boundary.png', bbox_inches='tight')

    plt.show()


def plot_decision_boundary_shaded(model, X, Y, feat_crosses=None, save=False):
    """
        Plots shaded decision boundary

        Args:
            model: neural network layer and activations in lambda function
            X: Data in shape (num_of_examples x features)
            feat_crosses: list of tuples showing which features to cross
            save: flag to save plot image
    """

    # Generate a grid of points between -0.5 and 1.5 with 1000 points in between
    xs = np.linspace(-0.5, 1.5, 1000)
    ys = np.linspace(1.5, -0.5, 1000)
    xx, yy = np.meshgrid(xs, ys)
    # Predict the function value for the whole grid
    # Z = model(np.c_[xx.ravel(), yy.ravel()]) # => add this for feature cross eg "xx.ravel()*yy.ravel()"

    prediction_data = np.c_[xx.ravel(), yy.ravel()]
    # add feat_crosses if provided
    if feat_crosses:
        for feature in feat_crosses:
            prediction_data = np.c_[prediction_data, prediction_data[:, feature[0]] * prediction_data[:, feature[1]]]

    Z = model(prediction_data)
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    cmap = matplotlib.colors.ListedColormap(["red","green"], name='from_list', N=None)
    plt.style.use('seaborn-whitegrid')

    # 'contourf'-> filled contours (red('#EABDBD'): 0 / green('#C8EDD6'): 1)
    plt.contourf(xx, yy, Z, cmap=matplotlib.colors.ListedColormap(['#EABDBD', '#C8EDD6'], name='from_list', N=None))
    plt.title('Decision boundary', size=18)
    plt.xlabel('$x_1$', size=20)
    plt.ylabel('$x_2$', size=20)
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.scatter(X[:, 0], X[:, 1], s=200, c=np.squeeze(Y), marker='x', cmap=cmap)  # s-> size of marker

    if save:
        plt.savefig('decision_boundary_shaded.png', bbox_inches='tight')

    plt.show()

