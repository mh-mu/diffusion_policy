import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import time, math
import numpy as np

from diffusion_policy.real_world.T42_controller import T42_controller

gripper_port = '/dev/ttyUSB0'
finger_offset_positions_rigid = [0.13, 0.13]

gripper = T42_controller(finger_offset_positions_rigid, finger_type='rigid', port=gripper_port, data_collection_mode=False)
# gripper = T42_controller(finger_offset_positions_compliant, finger_type='compliant', port=gripper_port, data_collection_mode=False)
print(gripper.read_motor_positions())
# gripper.release()
time.sleep(1)
gripper.move_to_zero_positions()
time.sleep(1)
# ic(gripper.read_motor_positions())
# time.sleep(1)
# gripper.release()
# time.sleep(1)

gripper.close()
time.sleep(1)