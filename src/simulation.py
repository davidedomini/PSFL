import random
from learning.model import MLP
from PSFLClient import psfl_client
from dummy_client import dummy_client
from phyelds.simulator import Simulator
from utils import distribute_nodes_spatially
from phyelds.simulator.render import render_sync
from phyelds.simulator.deployments import deformed_lattice
from phyelds.simulator.runner import aggregate_program_runner
from src.custom_exporter import federations_count_csv_exporter
from phyelds.simulator.neighborhood import radius_neighborhood
from phyelds.simulator.exporter import csv_exporter, ExporterConfig
from ProFed.partitioner import Environment, Region, download_dataset, split_train_validation, partition_to_subregions


def run_simulation(threshold, sparsity_level, number_subregions, seed):

    simulator = Simulator()

    # deformed lattice
    simulator.environment.set_neighborhood_function(radius_neighborhood(1.15))
    deformed_lattice(simulator, 7, 7, 1, 0.01)

    initial_model_params = MLP().state_dict()

    devices = len(simulator.environment.nodes.values())
    mapping_devices_area = distribute_nodes_spatially(devices, number_subregions)

    print(f'Number of devices: {devices}')
    print(mapping_devices_area)

    train_data, test_data = download_dataset('EMNIST')

    train_data, validation_data = split_train_validation(train_data, 0.8)
    print(f'Number of training samples: {len(train_data)}')
    environment = partition_to_subregions(train_data, validation_data, 'Hard', number_subregions, seed)

    mapping = {}

    for region_id, devices in mapping_devices_area.items():
        mapping_devices_data = environment.from_subregion_to_devices(region_id, len(devices))
        for device_index, data in mapping_devices_data.items():
            device_id = devices[device_index]
            mapping[device_id] = data


    # schedule the main function
    for node in simulator.environment.nodes.values():
        simulator.schedule_event(
            0.0,
            aggregate_program_runner,
            simulator,
            0.1,
            node,
            psfl_client,
            data=mapping[node.id],
            initial_model_params=initial_model_params,
            threshold=threshold,
            sparsity_level=sparsity_level)
        # simulator.schedule_event(
        #         0.0,
        #         aggregate_program_runner,
        #         simulator,
        #         0.1,
        #         node,
        #         dummy_client,
        #         data=mapping[node.id]
        # )
    # render
    simulator.schedule_event(0.95, render_sync, simulator, "result")
    config = ExporterConfig('data/', 'federations', [], [], 3)
    simulator.schedule_event(0.96, federations_count_csv_exporter, simulator, 1.0, config)
    config = ExporterConfig('data/', 'experiment', ['TrainLoss', 'ValidationLoss', 'ValidationAccuracy'], ['mean', 'std', 'min', 'max'], 3)
    simulator.schedule_event(1.0, csv_exporter, simulator, 1.0, config)
    simulator.run(100)


#TODO - update hyperparams
# Hyper-parameters configuration
thresholds = [20.0]
sparsity_levels = [0.0]
areas = [5]
seeds = [42]

for seed in seeds:
    random.seed(seed)
    for a in areas:
        for threshold in thresholds:
            for sparsity_level in sparsity_levels:
                run_simulation(threshold, sparsity_level, a, seed)
