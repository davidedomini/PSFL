import copy
import torch

def local_training(model):
    pass


def model_evaluation(model_params):
    pass


def average_weights(models_params, weights):
    w_avg = copy.deepcopy(models_params[0])
    for key in w_avg.keys():
        w_avg[key] = torch.mul(w_avg[key], 0.0)
    sum_weights = sum(weights)
    for key in w_avg.keys():
        for i in range(0, len(models_params)):
            w_avg[key] += models_params[i][key] * weights[i]
        w_avg[key] = torch.div(w_avg[key], sum_weights)
    return w_avg