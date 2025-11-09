def mse_loss(prediction, target):
    return ((prediction - target) ** 2).mean()