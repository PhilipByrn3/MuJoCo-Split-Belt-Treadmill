from dm_control import suite
from dm_control import viewer
from dm_control.rl.control import Environment
from dm_control.suite.base import Task
from dm_control.mujoco import Physics
from dm_control import mujoco
from dm_control.utils import xml_tools
from dm_control import mjcf
from dm_control.mjcf.base import Element

import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
from lxml import etree
import pandas as pd
import numpy as np
from moviepy.editor import ImageSequenceClip
from PIL import Image


import common
from sbtdata import construct_data, find_time, average_steady_velocity
from sbtgraph import convert_dataframe, generate_graph


class SbTreadmillTask(Task):

    def __init__(self):
        super().__init__()
    
    def get_observation(self, physics):
        return None
    
    def get_reward(self, physics):
        return None
    
    def initialize_episode(self, physics, give_boost=True, boost_value=0):
        # Gives the wheel an initial boost
        if give_boost is True:

            physics.named.data.qvel['axleYAxis'] = boost_value
        
        else:

            pass


def merge_models():

    world_xml_string = common.read_model('split_belt_treadmill.xml')
    model_xml_string = common.read_model('wheel.xml')

    parser = etree.XMLParser(remove_blank_text=True)

    world_mjcf = etree.XML(world_xml_string, parser)
    model_mjcf = etree.XML(model_xml_string, parser)

    # Find XML elements to combine
    world_worldbody = world_mjcf.find('worldbody')
    model_worldbody = model_mjcf.find('worldbody')

    world_sensor = world_mjcf.find('sensor')
    model_sensor = model_mjcf.find('sensor')

    # Combine chosen XML elements
    for body in model_worldbody:
        world_worldbody.append(body)
    
    for sensor in model_sensor:
        world_sensor.append(sensor)

    return etree.tostring(world_mjcf, pretty_print=True).decode()



def render_video(frames):
    # Set FPS to frames gathered in simulate_treadmill() accordingly
    clip = ImageSequenceClip(frames, fps=100)
    clip.write_videofile('sbt_video.mp4')



def simulate_treadmill(belt_diff, i, sb_physics, render_video_enable, print_data):
    
    slow_belt_force_raw, fast_belt_force_raw, measured_velocity_data, time_data, frames = [], [], [], [], []

    while i < 10000:

        sb_physics.step()

        # Belt velocities and positions are set at the beginning of each step
        sb_physics.named.data.qvel['slow_belt_conveyor'] = 1 # Constant value red (slow) belt
        sb_physics.named.data.qvel['fast_belt_conveyor'] = sb_physics.named.data.qvel['slow_belt_conveyor'] + belt_diff # Blue (fast) belt velocity = Red belt velocity + belt_diff

        sb_physics.named.data.qpos['fast_belt_conveyor'] = 0 # Slow belt position set to 0
        sb_physics.named.data.qpos['slow_belt_conveyor'] = 0 # Fast belt position set to 0
    
        # Gather data from sensors and append into list
        slow_belt_force_raw.append(sb_physics.named.data.sensordata['slow_belt_force_sensor'][2].copy())
        fast_belt_force_raw.append(sb_physics.named.data.sensordata['fast_belt_force_sensor'][2].copy())
        measured_velocity_data.append(abs(sb_physics.named.data.sensordata['axle_velocimeter'][2].copy()))
        time_data.append(sb_physics.data.time)
        
        if render_video_enable is True:

            # MUJOCO_GL=egl python test.py <-----------USE ME TO RUN IN TERMINAL IF RENDERING VIDEO in VM
            img = sb_physics.render(width=640, height=480, camera_id=0)
            frames.append(img)

        i+=1
    
    if render_video_enable is True:
            
        render_video(frames)

    # Create Dataframe for collected data
    dat_list = construct_data(time_data, slow_belt_force_raw, fast_belt_force_raw, measured_velocity_data)
    df, slow_belt_force, slow_belt_force_shift, fast_belt_force, fast_belt_force_shift, measured_velocity = dat_list
    
    # Calculate the average time of contact for spokes of wheel hitting the sensors on the belts
    time_slow = find_time(df, slow_belt_force, slow_belt_force_shift, print_data)
    time_fast = find_time(df, fast_belt_force, fast_belt_force_shift, print_data)

    # Angle offset between spokes of opposite sides and length of spokes
    alpha = np.deg2rad(4.5)
    beta = np.deg2rad(15.5)
    length = 2.54 # <-- NEED TO FIX SO THAT VALUE IS NOT CONSTANT, BUT UPDATES WITH MODEL
    
    # Find velocity using eq from Butterfield paper
    measured_velocity_avg, simulated_velocity_avg = average_steady_velocity(time_slow, time_fast, alpha, beta, length, belt_diff, measured_velocity)

    return measured_velocity_avg, simulated_velocity_avg



def loop_simulate_treadmill(belt_diff, i):

    while belt_diff <= 4.55:

        measured_velocity_avg, simulated_velocity_avg = simulate_treadmill(belt_diff, i, sb_physics, render_video_enable, print_data)

        #MVAlist.append(measured_velocity_avg)
        SVAlist.append(simulated_velocity_avg)
        BDlist.append(belt_diff)

        belt_diff += 0.005
        sb_env.reset()

    graphpd = {

    'BD': BDlist,
    'SVA': SVAlist,
    #'MVA': MVAlist,

    }
    
    gdf = pd.DataFrame(graphpd)
    return gdf


def create_graph(dataframe):
    belt_diff_axis, steady_velocity_avg = convert_dataframe(dataframe)
    generate_graph(belt_diff_axis, steady_velocity_avg)


        
if __name__ == '__main__':

    # Initialize Phyics, Task, Env, and initial conditions for SBT and wheel
    sb_physics = mujoco.Physics.from_xml_string(merge_models())
    sb_task = SbTreadmillTask()
    sb_env = Environment(physics=sb_physics, 
                         task=sb_task)
    
    # If you want to give your object some initial velocity, set give_boost to true and configure the function
    sb_task.initialize_episode(sb_physics)

    MVAlist, SVAlist, BDlist = [], [], []

    simulate_treadmill(belt_diff=0.55, i=0, sb_physics, render_video_enable=False, print_data=False) # Simulates one instance of the treadmill Env

    # gdf = loop_simulate_treadmill(belt_diff, i) # Simulates belt_diff/interval instances of treadmill Env

    # create_graph(gdf)
    

    
    
