"""
    The calculation to be performed at each point (modified model), evaluating
    the loss value, accuracy and eigen values of the hessian matrix
"""
import tensorflow.keras as keras
import time

def eval_loss(model, loader):
    """
    Evaluate the loss value for a given 'net' on the dataset provided by the loader.

    Args:
        net: the neural net model
        criterion: loss function
        loader: dataloader
        use_cuda: use cuda or not
    Returns:
        loss value and accuracy
    """
    x_test = loader['x']
    y_test = loader['y']
    loss_output, acc_output = model.evaluate(x_test, y_test, verbose=0)

    return loss_output, acc_output # this should just be loss and accuracy
