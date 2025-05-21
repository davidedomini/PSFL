import random
from learning.model import MLP
from PSFLClient import psfl_client
from phyelds.simulator import Simulator
from utils import divide_nodes_spatially
from ProFed.partitionings import Partitioner
from phyelds.simulator.render import render_sync
from phyelds.simulator.deployments import deformed_lattice
from phyelds.simulator.runner import aggregate_program_runner
from src.custom_exporter import federations_count_csv_exporter
from phyelds.simulator.neighborhood import radius_neighborhood
from phyelds.simulator.exporter import csv_exporter, ExporterConfig


from dummy_client import dummy_client

random.seed(42)


def run_simulation(threshold, sparsity_level, areas):

    simulator = Simulator()

    # deformed lattice
    simulator.environment.set_neighborhood_function(radius_neighborhood(1.15))
    deformed_lattice(simulator, 7, 7, 1, 0.01)

    initial_model_params = MLP().state_dict()

    devices = len(simulator.environment.nodes.values())
    mapping_devices_area = divide_nodes_spatially(devices, areas)

    print(f'Number of devices: {devices}')

    partitioner = Partitioner()
    dataset = partitioner.download_dataset('EMNIST')
    training_set, validation_set = partitioner.train_validation_split(dataset, 0.8)
    training_partitioning = partitioner.partition('Hard', training_set, areas)
    validation_partitioning = partitioner.partition('Hard', validation_set, areas)
    devices_training_data = partitioner.subregions_distributions_to_devices_distributions(training_partitioning, mapping_devices_area, training_set)
    devices_validation_data = partitioner.subregions_distributions_to_devices_distributions(validation_partitioning, mapping_devices_area, validation_set)

    # schedule the main function
    for node in simulator.environment.nodes.values():
        # simulator.schedule_event(
        #     0.0,
        #     aggregate_program_runner,
        #     simulator,
        #     0.1,
        #     node,
        #     psfl_client,
        #     data=..., # TODO
        #     initial_model_params=initial_model_params,
        #     threshold=threshold,
        #     sparsity_level=sparsity_level)
        simulator.schedule_event(
                0.0,
                aggregate_program_runner,
                simulator,
                0.1,
                node,
                dummy_client,
                data=(devices_training_data[node.id], devices_validation_data[node.id]))

            # render
    simulator.schedule_event(0.95, render_sync, simulator, "result")
    # config = ExporterConfig('data/', 'federations', [], [], 3)
    # simulator.schedule_event(0.96, federations_count_csv_exporter, simulator, 1.0, config)
    # config = ExporterConfig('data/', 'experiment', ['TrainLoss', 'ValidationLoss', 'ValidationAccuracy'], ['mean', 'std', 'min', 'max'], 3)
    # simulator.schedule_event(1.0, csv_exporter, simulator, 1.0, config)
    simulator.run(100)


# Hyper-parameters configuration
thresholds = [20.0]
sparsity_levels = [0.0] # TODO - update this
areas = [5] # TODO - update this

for a in areas:
    for threshold in thresholds:
        for sparsity_level in sparsity_levels:
            run_simulation(threshold, sparsity_level, a)
