from phyelds.calculus import aggregate
from phyelds.libraries.collect import count_nodes
from phyelds.libraries.leader_election import elect_leader
from phyelds.libraries.distances import neighbors_distances
from phyelds.libraries.spreading import distance_to, broadcast

@aggregate
def PSFLClient():
    """
    Example to use the phyelds library to create a simple simulation
    :return:
    """
    distances = neighbors_distances()
    leader = elect_leader(4, distances)
    potential = distance_to(leader, distances)
    nodes = count_nodes(potential)
    area_value = broadcast(leader, nodes, distances)
    return area_value
