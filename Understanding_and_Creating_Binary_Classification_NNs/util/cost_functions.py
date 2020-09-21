"""
Created by : Rafay Khan
Twitter: @RafayAK
"""


import numpy as np

"""
Contains a bunch of Cost functions.
This file implementations of :
    - Binary Cross_entropy Cost function
        * compute_binary_cost(Y, P_hat) -> "unstable"
        * compute_stable_bce_cost(Y, Z) -> "stable" 
        * computer_keras_like_bce_cost(Y, P_hat, from_logits=False) -> stable
    - Mean Squared Error Cost function
"""


def compute_bce_cost(Y, P_hat):
    """
    This function computes Binary Cross-Entropy(bce) Cost and returns the Cost and its
    derivative.
    This function uses the following Binary Cross-Entropy Cost defined as:
    => (1/m) * np.sum(-Y*np.log(P_hat) - (1-Y)*np.log(1-P_hat))

    Args:
        Y: labels of data
        P_hat: Estimated output probabilities from the last layer, the output layer

    Returns:
        cost: The Binary Cross-Entropy Cost result
        dP_hat: gradient of Cost w.r.t P_hat

    """
    m = Y.shape[1]  # m -> number of examples in the batch

    cost = (1/m) * np.sum(-Y*np.log(P_hat) - (1-Y)*np.log(1-P_hat))
    cost = np.squeeze(cost)  # remove extraneous dimensions to give just a scalar (e.g. this turns [[17]] into 17)

    dP_hat = (1/m) * (-(Y/P_hat) + ((1-Y)/(1-P_hat)))

    return cost, dP_hat


def compute_stable_bce_cost(Y, Z):
    """
    This function computes the "Stable" Binary Cross-Entropy(stable_bce) Cost and returns the Cost and its
    derivative w.r.t Z_last(the last linear node) .
    The Stable Binary Cross-Entropy Cost is defined as:
    => (1/m) * np.sum(max(Z,0) - ZY + log(1+exp(-|Z|)))
    Args:
        Y: labels of data
        Z: Values from the last linear node

    Returns:
        cost: The "Stable" Binary Cross-Entropy Cost result
        dZ_last: gradient of Cost w.r.t Z_last
    """
    m = Y.shape[1]

    cost = (1/m) * np.sum(np.maximum(Z, 0) - Z*Y + np.log(1+ np.exp(- np.abs(Z))))
    dZ_last = (1/m) * ((1/(1+np.exp(- Z))) - Y)  # from Z computes the Sigmoid so P_hat - Y, where P_hat = sigma(Z)

    return cost, dZ_last


def compute_keras_like_bce_cost(Y, P_hat, from_logits=False):
    """
    This function computes the Binary Cross-Entropy(stable_bce) Cost function the way Keras
    implements it. Accepting either probabilities(P_hat) from the sigmoid neuron or values direct
    from the linear node(Z)

    Args:
        Y: labels of data
        P_hat: Probabilities from sigmoid function
        from_logits: flag to check if logits are being provided or not(Default: False)

    Returns:
        cost: The "Stable" Binary Cross-Entropy Cost result
        dZ_last: gradient of Cost w.r.t Z_last

    """
    if from_logits:
        # assume that P_hat contains logits and not probabilities
        return compute_stable_bce_cost(Y, Z=P_hat)

    else:
        # Assume P_hat contains probabilities. So make logits out of them

        # First clip probabilities to stable range
        EPSILON = 1e-07
        P_MAX = 1- EPSILON  # 0.9999999

        P_hat = np.clip(P_hat, a_min=EPSILON, a_max=P_MAX)

        # Second, Convert stable probabilities to logits(Z)
        Z = np.log(P_hat/(1-P_hat))

        # now call compute_stable_bce_cost
        return compute_stable_bce_cost(Y, Z)



def compute_mse_cost(Y, Y_hat):
    """
    This function computes Mean Squared Error(mse) Cost and returns the Cost and its derivative.
    This function uses the Squared Error Cost defined as follows:
    => (1/2m)*sum(Y - Y_hat)^.2

    Args:
        Y: labels of data
        Y_hat: Predictions(activations) from a last layer, the output layer

    Returns:
        cost: The Squared Error Cost result
        dY_hat: gradient of Cost w.r.t Y_hat

    """
    m = Y.shape[1]  # m -> number of examples in the batch

    cost = (1 / (2 * m)) * np.sum(np.square(Y - Y_hat))
    cost = np.squeeze(cost)  # remove extraneous dimensions to give just a scalar (e.g. this turns [[17]] into 17)

    dY_hat = -1 / m * (Y - Y_hat)  # derivative of the squared error cost function

    return cost, dY_hat
