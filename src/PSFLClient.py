from learning.model import MLP
from phyelds.data import Field
from phyelds.libraries.collect import collect_with
from phyelds.libraries.device import local_id, store
from phyelds.calculus import aggregate, neighbors, remember
from phyelds.libraries.leader_election import elect_leaders
from phyelds.libraries.spreading import distance_to, broadcast
from learning import local_training, model_evaluation, average_weights


impulsesEvery = 5


@aggregate
def psfl_client(initial_model_params, data, threshold, sparsity_level):

    data = remember((initial_model_params, 0)) # Stores local model and current global round
    local_model_weights, tick = data.value
    local_model = load_from_weights(local_model_weights)

    # Local training
    evolved_model, train_loss = local_training(local_model, 2, data, 128)
    validation_loss, validation_accuracy = model_evaluation(evolved_model, data, 128)

    log(train_loss, validation_loss, validation_accuracy) # Metrics logging

    distances = loss_based_distances(evolved_model.state_dict())
    leader = elect_leaders(threshold, distances) # If leader is true, then I'm an aggregator
    store('is_aggregator', leader)
    potential = distance_to(leader, distances)

    models = collect_with(potential, set(evolved_model.state_dict()), lambda x, y: x.union(y))
    aggregated_model = average_weights(models, [1.0 for _ in models])
    area_model = broadcast(leader, aggregated_model, distances)

    if tick % impulsesEvery == 0:
        average_weights({evolved_model, area_model}, [0.1, 0.9])

    data.update((evolved_model, tick + 1))


@aggregate
def loss_based_distances(model_weights):
    models_weights = neighbors(model_weights)
    neighbors_models = Field(models_weights.exclude_self(), local_id())
    evaluations = neighbors_models.map(model_evaluation)
    neighbors_evaluations = neighbors(evaluations.data)
    loss_field = compute_loss_metric(evaluations, neighbors_evaluations.data)
    return loss_field


@aggregate
def compute_loss_metric(evaluations, neighbors_evaluations):
    mid = evaluations.node_id
    data = evaluations.data
    loss_metric = dict()
    for neighbor_id, evaluation in data.items():
        neighbor_evaluation_of_myself = neighbors_evaluations[neighbor_id].get(mid, float('inf'))
        loss_metric[neighbor_id] = neighbor_evaluation_of_myself + evaluation
    return Field(loss_metric, mid)


def log(train_loss, validation_loss, validation_accuracy):
    store('TrainLoss', train_loss)
    store('ValidationLoss', validation_loss)
    store('ValidationAccuracy', validation_accuracy)


def load_from_weights(weights):
    model = MLP()
    model.load_state_dict(weights)
    return model