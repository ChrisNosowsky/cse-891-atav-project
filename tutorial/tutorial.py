from beamngpy import BeamNGpy, Scenario, Vehicle

STEAM_GAME_PATH_DIR = "E:/Program Files (x86)/Steam/steamapps/common/BeamNG.drive"

bng = BeamNGpy('localhost', 64256, home=STEAM_GAME_PATH_DIR)

bng.open()

scenario = Scenario('east_coast_usa', 'tech_test', description='Random driving for research')
vehicle = Vehicle('ego_vehicle', model='etk800', license='PYTHON')

scenario.add_vehicle(vehicle, pos=(-426.68, -43.59, 31.11), rot_quat=(0, 0, 1, 0))

scenario.make(bng)

bng.scenario.load(scenario)
bng.scenario.start()

vehicle.ai.set_mode('span')
input('Hit enter when done...')
