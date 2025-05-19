import random
from PSFLClient import psfl_client
from phyelds.simulator import Simulator
from phyelds.simulator.render import render_sync
from phyelds.simulator.deployments import deformed_lattice
from phyelds.simulator.runner import aggregate_program_runner
from phyelds.simulator.neighborhood import radius_neighborhood
from phyelds.simulator.exporter import csv_exporter, ExporterConfig

random.seed(42)

simulator = Simulator()
# deformed lattice
simulator.environment.set_neighborhood_function(radius_neighborhood(1.15))
deformed_lattice(simulator, 10, 10, 1, 0.01)
# schedule the main function
for node in simulator.environment.nodes.values():
    simulator.schedule_event(0.0, aggregate_program_runner, simulator, 0.1, node, psfl_client)
# render
simulator.schedule_event(0.95, render_sync, simulator, "result")
config = ExporterConfig('data/', 'experiment', ['TrainLoss', 'ValidationLoss', 'ValidationAccuracy'], ['mean', 'std', 'min', 'max'], 3)
simulator.schedule_event(1.0, csv_exporter, simulator, 1.0, config)
simulator.run(100)