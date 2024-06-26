from dm_control import suite
from dm_control import viewer
from dm_control.rl.control import Environment
from dm_control.suite.base import Task
from dm_control.mujoco import Physics
from dm_control import mujoco
from dm_control.utils import xml_tools
from dm_control import mjcf


import os

from lxml import etree
import pandas as pd
import numpy as np
from moviepy.editor import ImageSequenceClip
from PIL import Image

import common
from sbtdata import ConstructData, findTime, AverageSteadyVelocity


class SbTreadmillTask(Task):

    def __init__(self):
        super().__init__()
    
    def get_observation(self, physics):
        return None
    
    def get_reward(self, physics):
        return None
    
    def initialize_episode(self, physics):
        physics.named.data.qvel['redconveyor'] = 1
        physics.named.data.qvel['blueconveyor'] = 1.5
        physics.named.data.qvel['axleYAxis'] = 0  # Talk about this

def merge_models():

    world = mjcf.RootElement()
    world.worldbody.add('geom', name='ground',  type='plane', size=[10, 10, 1])

    wheel = mjcf.from_xml_file('wheel.xml')
    world.attach(wheel)

def render_video(frames):
    clip = ImageSequenceClip(frames, fps=100)
    clip.write_videofile('my_video.mp4')



def simulate_treadmill(belt_diff, i, sb_physics, render_video_enable, print_data):
    
    frames = []
    fprdat_raw, fpbdat_raw, measured_velocity_data, timedata = [], [], [], []

    while i < 1700:

        sb_physics.step()
        sb_physics.named.data.qvel['redconveyor'] = 1 # Constant value red (slow) belt
        sb_physics.named.data.qvel['blueconveyor'] = sb_physics.named.data.qvel['redconveyor'] + belt_diff # Blue (fast) belt velocity = Red belt velocity + belt_diff

        if (sb_physics.data.qpos[0] > 10) or (sb_physics.data.qpos[1] > 10):
            sb_physics.data.qpos[0] = 0 # Red belt
            sb_physics.data.qpos[1] = 0 # Blue Belt
    
        fprdat_raw.append(sb_physics.named.data.sensordata[2].copy())
        fpbdat_raw.append(sb_physics.named.data.sensordata[5].copy())
        measured_velocity_data.append(abs(sb_physics.named.data.sensordata['axle_velocimeter'][2].copy()))
        timedata.append(sb_physics.data.time)
        
        if render_video_enable is True:
            img = sb_physics.render(width=640, height=480,camera_id=0)
            frames.append(img)

        i+=1
    
    if render_video_enable is True:
            
        render_video(frames)

    datList = ConstructData(timedata, fprdat_raw, fpbdat_raw, measured_velocity_data)
    df, fprdat, fprdat_shift, fpbdat, fpbdat_shift, measured_velocity = datList
    
    time_slow = findTime(df, fprdat, fprdat_shift, print_data)
    time_fast = findTime(df, fpbdat, fpbdat_shift, print_data)

    alpha = np.deg2rad(4.5)
    beta = np.deg2rad(15.5)
    length = 2
    
    measured_velocity_avg, simulated_velocity_avg = AverageSteadyVelocity(time_slow, time_fast, alpha, beta, length, belt_diff, measured_velocity)

    return measured_velocity_avg, simulated_velocity_avg



def loop_simulate_treadmill(belt_diff, i):

    while belt_diff < 1.8:

        measured_velocity_avg, simulated_velocity_avg = simulate_treadmill(belt_diff, i, sb_physics, render_video_enable=False, print_data=False)

        MVAlist.append(measured_velocity_avg)
        SVAlist.append(simulated_velocity_avg)
        BDlist.append(belt_diff)

        belt_diff += 0.005
        sb_env.reset()

    graphpd = {

    'BD': BDlist,
    'SVA': SVAlist,
    'MVA': MVAlist,

    }
    
    gdf = pd.DataFrame(graphpd)

    gdf.to_csv('Velocities_and BeltDiffs.csv')


        
if __name__ == '__main__':

    merge_models()

    sb_physics = mujoco.Physics.from_xml_path("rimlesstreadmill.xml")
    sb_task = SbTreadmillTask()
    sb_env = Environment(physics=sb_physics, task=sb_task)
    sb_task.initialize_episode(sb_physics)

    MVAlist, SVAlist, BDlist = [], [], []

    render_video_enable = True
    print_data = False

    belt_diff = 0.5
    i = 0

    simulate_treadmill(belt_diff, i, sb_physics, render_video_enable, print_data) # Simulates one instance of the treadmill Env

    # loop_simulate_treadmill(belt_diff, i) # Simulates belt_diff/interval instances of treadmill Env

    
    
