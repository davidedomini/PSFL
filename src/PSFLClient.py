from phyelds.libraries.device import local_id
from phyelds.calculus import aggregate, neighbors
from phyelds.libraries.collect import count_nodes
from phyelds.libraries.leader_election import elect_leader
from phyelds.libraries.distances import neighbors_distances
from phyelds.libraries.spreading import distance_to, broadcast

@aggregate
def psfl_client():
    """
    Example to use the phyelds library to create a simple simulation
    :return:
    """
    loss_based_distance()
    distances = neighbors_distances()
    leader = elect_leader(4, distances)
    potential = distance_to(leader, distances)
    nodes = count_nodes(potential)
    area_value = broadcast(leader, nodes, distances)
    return area_value

def loss_based_distance():
    mid = local_id()
    models = neighbors(mid)
