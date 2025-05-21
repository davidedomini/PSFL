import random
from learning.model import MLP
from PSFLClient import psfl_client
from phyelds.simulator import Simulator
from phyelds.simulator.render import render_sync
from phyelds.simulator.deployments import deformed_lattice
from phyelds.simulator.runner import aggregate_program_runner
from src.custom_exporter import federations_count_csv_exporter
from phyelds.simulator.neighborhood import radius_neighborhood
from phyelds.simulator.exporter import csv_exporter, ExporterConfig


random.seed(42)


def run_simulation(threshold, sparsity_level):
    simulator = Simulator()
    # deformed lattice
    simulator.environment.set_neighborhood_function(radius_neighborhood(1.15))
    deformed_lattice(simulator, 10, 10, 1, 0.01)

    initial_model_weights = MLP().state_dict()

    # schedule the main function
    for node in simulator.environment.nodes.values():
        simulator.schedule_event(
            0.0,
            aggregate_program_runner,
            simulator,
            0.1,
            node,
            psfl_client,
            initial_model_weights=initial_model_weights,
            threshold=threshold,
            sparsity_level=sparsity_level)

    # render
    simulator.schedule_event(0.95, render_sync, simulator, "result")
    config = ExporterConfig('data/', 'federations', [], [], 3)
    simulator.schedule_event(0.96, federations_count_csv_exporter, simulator, 1.0, config)
    config = ExporterConfig('data/', 'experiment', ['TrainLoss', 'ValidationLoss', 'ValidationAccuracy'], ['mean', 'std', 'min', 'max'], 3)
    simulator.schedule_event(1.0, csv_exporter, simulator, 1.0, config)
    simulator.run(100)


# Hyper-parameters configuration
thresholds = [20.0]
sparsity_levels = [0.0] # TODO - update this

for threshold in thresholds:
    for sparsity_level in sparsity_levels:
        run_simulation(threshold, sparsity_level)
