from .loss_functional import *

loss_ = {
    "ce_loss": ce_loss,
    "mse_loss": mes_loss
}


def get_loss_function(name):
    assert name in loss_.keys()
    return loss_[name]
