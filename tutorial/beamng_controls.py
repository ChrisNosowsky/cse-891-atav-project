import time
from beamngpy import BeamNGpy, Vehicle, Scenario

STEAM_GAME_PATH_DIR = "E:/Program Files (x86)/Steam/steamapps/common/BeamNG.drive"

# Connect to the BeamNG.drive simulator
beamng = BeamNGpy('localhost', 64256, home=STEAM_GAME_PATH_DIR)

beamng.open(launch=True)

scenario = Scenario('east_coast_usa', 'freeroam')
vehicle = Vehicle('vehicle', model='pickup', licence='PYTHON')
scenario.restart()
# Add the vehicle to the scenario
scenario.add_vehicle(vehicle, pos=(-426.68, -43.59, 31.11), rot_quat=(0, 0, 1, 0))
scenario.make(beamng)

beamng.scenario.load(scenario)
beamng.scenario.start()

# Control inputs for the vehicle
steering = 0.0  # Example value for steering input (range: [-1, 1])
throttle = 0.5  # Example value for throttle input (range: [0, 1])

# Control the vehicle's inputs for a certain duration
duration = 10  # Example duration in seconds
start_time = time.time()
while time.time() - start_time < duration:
    # Apply control inputs
    vehicle.control(steering=steering, throttle=throttle)

    # Step the simulation
    beamng.step(1)

# Cleanup
beamng.close()
