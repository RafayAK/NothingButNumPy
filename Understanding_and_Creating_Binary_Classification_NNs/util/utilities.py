"""
Created by : Rafay Khan
Twitter: @RafayAK
"""


import numpy as np
import matplotlib.pyplot as plt  # import matplotlib for plotting and visualization
import matplotlib

# only required for inline labels in `plot_decision_boundary_distances` function
try:
    from labellines import labelLine
except ImportError:
    print("""Caution! `matplotlib-label-lines` package is not available, 'plot_decision_boundary_distances`
    will not work. Try installing the package through `pip install matplotlib-label-lines`""")


"""
Contains a bunch of helper functions
"""


def predict(X, Y, Zs, As, thresh=0.5):
    """
    helper function to predict on data using a neural net model layers

    Args:
        X: Data in shape (features x num_of_examples)
        Y: labels in shape ( label x num_of_examples)
        Zs: All linear layers in form of a list e.g [Z1,Z2,...,Zn]
        As: All Activation layers in form of a list e.g [A1,A2,...,An]
        thresh: is the classification threshold. All values >= threshold belong to positive class(1)
                and the rest to the negative class(0).Default threshold value is 0.5
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
        if probas[0, i] >= thresh:  # 0.5  the default threshold
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


def predict_dec(Zs, As, X, thresh=0.5):
    """
    Used for plotting decision boundary.

    Args:
        Zs: All linear layers in form of a list e.g [Z1,Z2,...,Zn]
        As: All Activation layers in form of a list e.g [A1,A2,...,An]
        X: Data in shape (features x num_of_examples) i.e (K x m), where 'm'=> number of examples
           and "K"=> number of features
        thresh: is the classification threshold. All values >= threshold belong to positive class(1)
                and the rest to the negative class(0).Default threshold value is 0.5
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

    # if probability of example >= thresh => output 1, vice versa
    predictions = (probas >= thresh)
    return predictions


def plot_decision_boundary(model, X, Y, feat_crosses=None, axis_lines=False,save=False):
    """
    Plots decision boundary

    Args:
        model: neural network layer and activations in lambda function
        X: Data in shape (num_of_examples x features)
        feat_crosses: list of tuples showing which features to cross
        axis_lines: Draw axis lines at x=0 and y=0(bool, default False)
        save: flag to save plot image
    """

    # first plot the data to see what is the size of the plot
    plt.scatter(X[:, 0], X[:, 1], s=200, c=np.squeeze(Y))  # s-> size of marker

    # get the x and y range of the plot
    x_ticks = plt.xticks()[0]
    y_ticks = plt.yticks()[0]

    plt.clf()  # clear figure after getting size

    # Generate a grid of points between min_x_point-0.5 and max_x_point+0.5 with 1000 points in between,
    # similarly, for y points
    xs = np.linspace(min(x_ticks) - 0.5, max(x_ticks) + 0.5, 1000)
    ys = np.linspace(max(y_ticks) + 0.5, min(y_ticks) - 0.5, 1000)

    xx, yy = np.meshgrid(xs, ys)  # create data points
    # Predict the function value for the whole grid

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
    plt.title('Decision Boundary', size=18)
    plt.xlabel('$x_1$', size=20)
    plt.ylabel('$x_2$', size=20)
    if axis_lines:
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')

    # color map 'cmap' maps 0 labeled data points to red and 1 labeled points to green
    cmap = matplotlib.colors.ListedColormap(["red", "green"], name='from_list', N=None)
    plt.scatter(X[:, 0], X[:, 1], s=200, c=np.squeeze(Y), marker='x', cmap=cmap)  # s-> size of marker

    if save:
        plt.savefig('decision_boundary.png', bbox_inches='tight')

    plt.show()


def plot_decision_boundary_shaded(model, X, Y, feat_crosses=None, axis_lines=False,save=False):
    """
        Plots shaded decision boundary

        Args:
            model: neural network layer and activations in lambda function
            X: Data in shape (num_of_examples x features)
            feat_crosses: list of tuples showing which features to cross
            axis_lines: Draw axis lines at x=0 and y=0(bool, default False)
            save: flag to save plot image
    """

    # first plot the data to see what is the size of the plot
    plt.scatter(X[:, 0], X[:, 1], s=200, c=np.squeeze(Y))  # s-> size of marker

    # get the x and y range of the plot
    x_ticks = plt.xticks()[0]
    y_ticks = plt.yticks()[0]

    plt.clf()  # clear figure after getting size

    # Generate a grid of points between min_x_point-0.5 and max_x_point+0.5 with 1000 points in between,
    # similarly, for y points
    xs = np.linspace(min(x_ticks)-0.5, max(x_ticks)+0.5, 1000)
    ys = np.linspace(max(y_ticks)+0.5, min(y_ticks)-0.5, 1000)
    xx, yy = np.meshgrid(xs, ys)

    # Predict the function value for the whole grid

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
    plt.title('Shaded Decision Boundary', size=18)
    plt.xlabel('$x_1$', size=20)
    plt.ylabel('$x_2$', size=20)
    if axis_lines:
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
    plt.scatter(X[:, 0], X[:, 1], s=200, c=np.squeeze(Y), marker='x', cmap=cmap)  # s-> size of marker

    if save:
        plt.savefig('decision_boundary_shaded.png', bbox_inches='tight')

    plt.show()


def point_on_line(P1, P2, points):
    """
    Helper function for `plot_decision_boundary_distances`.
    This function calculates the perpendicular point(closes point) on the decision boundary line from another point

    Logic for finding intersection points:
        -https://stackoverflow.com/questions/10301001/perpendicular-on-a-line-segment-from-a-given-point

    Logic for finding distances:
        -https://stackoverflow.com/questions/39840030/distance-between-point-and-a-line-from-two-points/39840218

    Args:
        P1: a point on the line
        P2: another point on the line
        points: list of points

    Returns:
        intersection_points, distances_to_intersection points

    """
    distances = np.abs(np.cross(P2 - P1, P1 - points) / np.linalg.norm(P2 - P1))

    x = np.dot(points - P1, (P2 - P1).T) / np.dot(P2 - P1, (P2 - P1).T)
    return P1 + x * (P2 - P1), distances


def plot_decision_boundary_distances(model, X, Y, feat_crosses=None, axis_lines=False, save=False):
    """
    Plots decision boundary

    Args:
        model: neural network layer and activations in lambda function
        X: Data in shape (num_of_examples x features)
        feat_crosses: list of tuples showing which features to cross
        axis_lines: Draw axis lines at x=0 and y=0(bool, default False)
        save: flag to save plot image
    """

    # first plot the data to see what is the size of the plot
    plt.scatter(X[:, 0], X[:, 1], s=200, c=np.squeeze(Y))

    # get the x and y range of the plot
    x_ticks = plt.xticks()[0]
    y_ticks = plt.yticks()[0]

    plt.clf()  # clear figure after getting size

    # Generate a grid of points between min_x_point-0.5 and max_x_point+0.5 with 1000 points in between,
    # similarly, for y points
    xs = np.linspace(min(x_ticks) - 0.5, max(x_ticks) + 0.5, 1000)
    ys = np.linspace(max(y_ticks) + 0.5, min(y_ticks) - 0.5, 1000)
    xx, yy = np.meshgrid(xs, ys)  # create data points
    # Predict the function value for the whole grid

    prediction_data = np.c_[xx.ravel(), yy.ravel()]
    # add feat_crosses if provided
    if feat_crosses:
        for feature in feat_crosses:
            prediction_data = np.c_[prediction_data, prediction_data[:, feature[0]] * prediction_data[:, feature[1]]]

    Z = model(prediction_data)
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.style.use('seaborn-whitegrid')
    c = plt.contour(xx, yy, Z, cmap='Blues')  # draw a blue colored decision boundary
    plt.title('Distances from Decision Boundary', size=18)
    plt.xlabel('$x_1$', size=20)
    plt.ylabel('$x_2$', size=20)
    if axis_lines:
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')

    # color map 'cmap' maps 0 labeled data points to red and 1 labeled points to green
    cmap = matplotlib.colors.ListedColormap(["red", "green"], name='from_list', N=None)
    plt.scatter(X[:, 0], X[:, 1], s=200, c=np.squeeze(Y), marker='x', cmap=cmap)  # s-> size of marker

    points = X  # data points from to which perpendicular lines are drawn
    v = c.collections[0].get_paths()[0].vertices  # returns two points from the decision line(visible start & end point)
    P1 = np.expand_dims(np.asarray((v[0, 0], v[0, 1])), axis=0)  # the visible start point of the line
    P2 = np.expand_dims(np.asarray((v[-1, 0], v[-1, 1])), axis=0)  # the visible end point of the line

    inter_points, distances = point_on_line(P1, P2, points)

    # combine the intersection points so that they're in the format required by `plt.plot` so
    # each list item is:
    # [(x_1,x_2), (y_1, y_2), len_of_line]
    perpendicular_line_points = [list(zip(a, b))+[c] for a, b, c in zip(points, inter_points, distances)]

    # plot and label perpendicular lines to the decision boundary one by one
    # labelLine function comes from https://github.com/cphyc/matplotlib-label-lines/tree/master/labellines/baseline
    for line in perpendicular_line_points:
        x_points = np.clip(line[0], a_min=-0.5, a_max=1.5)  # clip lines going out of bounds of visible area
        y_points = np.clip(line[1], a_min=-0.5, a_max=1.5)
        len = line[2]
        plt.plot(x_points, y_points, 'm--', label='{:.2f}'.format(len))  # print label to 2 decimal places
        labelLine(plt.gca().get_lines()[-1], x= sum(x_points)/2)  # label of the line should be centered, so (x_1+x_2)/2

    if save:
        plt.savefig('decision_boundary_with_distances.png', bbox_inches='tight')

    plt.tight_layout()
    plt.show()
