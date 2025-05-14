import random
from PSFLClient import PSFLClient
from phyelds.simulator import Simulator
from phyelds.simulator.render import render_sync
from phyelds.simulator.deployments import deformed_lattice
from phyelds.simulator.runner import aggregate_program_runner
from phyelds.simulator.neighborhood import radius_neighborhood

random.seed(42)

simulator = Simulator()
# deformed lattice
simulator.environment.set_neighborhood_function(radius_neighborhood(1.15))
deformed_lattice(simulator, 10, 10, 1, 0.01)
# schedule the main function
for node in simulator.environment.nodes.values():
    simulator.schedule_event(0.0, aggregate_program_runner, simulator, 0.1, node, PSFLClient)
# render
simulator.schedule_event(1.0, render_sync, simulator, "result")
simulator.run(100)