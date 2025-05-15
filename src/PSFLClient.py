from phyelds.data import Field
from phyelds.libraries.device import local_id
from phyelds.calculus import aggregate, neighbors
from phyelds.libraries.collect import count_nodes
from phyelds.libraries.leader_election import elect_leaders
from phyelds.libraries.distances import neighbors_distances
from phyelds.libraries.spreading import distance_to, broadcast

@aggregate
def psfl_client():
    """
    Example to use the phyelds library to create a simple simulation
    :return:
    """
    distances = neighbors_distances() # TODO --> loss_based_distances()
    leader = elect_leaders(4, distances)
    potential = distance_to(leader, distances)
    nodes = count_nodes(potential)
    area_value = broadcast(leader, nodes, distances)
    return area_value

@aggregate
def loss_based_distances():
    local_model = local_id() # TODO - this should be the neural network
    models = neighbors(local_model)
    neighbors_models = Field(models.exclude_self(), local_id())
    evaluations = neighbors_models.map(evaluate)
    neighbors_evaluations = neighbors(evaluations.data)
    loss_field = compute_loss_metric(evaluations, neighbors_evaluations.data)
    return loss_field

def evaluate(model) -> float:
    return 2.0 #TODO - implement neural network evaluation

@aggregate
def compute_loss_metric(evaluations, neighbors_evaluations):
    mid = evaluations.node_id
    data = evaluations.data
    loss_metric = dict()
    for neighbor_id, evaluation in data.items():
        neighbor_evaluation_of_myself = neighbors_evaluations[neighbor_id].get(mid, float('inf'))
        loss_metric[neighbor_id] = neighbor_evaluation_of_myself + evaluation
    return Field(loss_metric, mid)