from dm_control import suite
from dm_control import viewer
from dm_control.rl.control import Environment
from dm_control.suite.base import Task
from dm_control import mujoco
import numpy as np


class SbTreadmillTask(Task):

    def __init__(self):
        return
    
    def get_observation(self):
        return None
    
    def get_reward(self):
        return None

if __name__ == '__main__':

    sb_physics = mujoco.Physics.from_xml_path("rimlesstreadmill.xml")
    sb_task = SbTreadmillTask()

    sb_env = Environment(physics=sb_physics,
                        task=sb_task)

    # env = suite.load(domain_name="humanoid", task_name="stand")
    action_spec = sb_env.action_spec()

    # Launch the viewer application.
    viewer.launch(sb_env)


