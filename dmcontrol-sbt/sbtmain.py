import cv2
from lxml import etree
import numpy as np
import time

import dm_control
from dm_control import suite, viewer, mujoco
from dm_control.rl.control import Environment
from dm_control.suite.base import Task
from dm_control.mujoco import Physics
from configparser import ConfigParser

import common



class SplitBeltTreadmillTask(Task):

    def __init__(self):
        super().__init__()
    
    def get_observation(self, physics):
        return None
    
    def get_reward(self, physics):
        return None
    
    def initialize_episode(self, physics):
        return None

def sbt_config_load():
    config = ConfigParser()
    config.read('config.ini')
    return config

def str_to_bool(value):
    if value.lower() in ("yes", "true", "t", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise ValueError("Invalid boolean value: {}".format(value))

def config_entry_to_list(config_file, section, key):
    config = ConfigParser()
    config.read(config_file)
    
    value = config.get(section, key)
    
    return [float(item.strip()) for item in value.split(',')]

def sbt_merge_models(model):
    sbt_xml_string = common.read_model('models/split_belt_treadmill.xml')
    model_xml_string = common.read_model(model)

    parser = etree.XMLParser(remove_blank_text=True)

    sbt_mjcf = etree.XML(sbt_xml_string, parser)
    model_mjcf = etree.XML(model_xml_string, parser)

    # Find XML elements to combine
    sbt_worldbody = sbt_mjcf.find('worldbody')
    model_worldbody = model_mjcf.find('worldbody')

    sbt_sensor = sbt_mjcf.find('sensor')
    model_sensor = model_mjcf.find('sensor')

    # Combine chosen XML elements
    for body in model_worldbody:
        sbt_worldbody.append(body)
    
    for sensor in model_sensor:
        sbt_sensor.append(sensor)

    return etree.tostring(sbt_mjcf, pretty_print=True).decode()

def sbt_render_video(frames):
    if frames and frames[0] is not None:
        height, width, _ = frames[0].shape
        fps = 24 
        out = cv2.VideoWriter('media/sbt_video.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

        if not out.isOpened():
            raise Exception("VideoWriter could not be opened. Check codec and file path.")

        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr) 

        out.release()

    else:
        raise ValueError("The frames list is empty or frames are invalid.")

def sbt_initialize_dmcontrol():
    sbt_world = sbt_merge_models(model=config.get('SimulationConfig', 'model_xml_path'))
    sbt_physics = mujoco.Physics.from_xml_string(sbt_world)
    sbt_task = SplitBeltTreadmillTask()
    sbt_env = Environment(physics=sbt_physics, 
                          task=sbt_task)
    return [sbt_world, sbt_physics, sbt_task, sbt_env]

def sbt_simulate_treadmill(base_belt_diff, set_belt_diff, timesteps, render_video, sbt_physics):
    i = 0
    frames = []
    while i < timesteps:
        sbt_physics.step()
        sbt_physics.named.data.qvel['slow_belt_conveyor'] = base_belt_diff 
        sbt_physics.named.data.qvel['fast_belt_conveyor'] = sbt_physics.named.data.qvel['slow_belt_conveyor'] + set_belt_diff

        sbt_physics.named.data.qpos['fast_belt_conveyor'] = 0 
        sbt_physics.named.data.qpos['slow_belt_conveyor'] = 0 

        if render_video is True:
            img = sbt_physics.render(width=640, height=480, camera_id='axle_cam')
            frames.append(img)

        i+=1

    if render_video is True:   
        sbt_render_video(frames)
    
def sbt_loop_simulate_treadmill(base_belt_diff, belt_speed_difference_range, belt_difference_increment):
    i = base_belt_diff
    loop_start_time = time.time()

    while i <= belt_speed_difference_range:
        render_video_overwrite = False
        sim_start_time = time.time()

        sbt_simulate_treadmill(float(config.get('SimulationConfig', 'sbt_belt_speed_baseline_difference')),
                               float(config.get('SimulationConfig', 'sbt_belt_speed_set_difference')),
                               int(config.get('SimulationConfig', 'number_of_timesteps_per_simulation')),
                               render_video_overwrite,
                               sbt_physics
                               )
        
        sim_end_time = time.time()

        print('Simulation complete with difference of: ', round(i,4),  '\nTime Elapsed: ', sim_end_time-sim_start_time,'\n----------')
        i += belt_difference_increment

    loop_end_time = time.time()

    print('Loop Simulation complete!',
          '\nTotal time elapsed: ', loop_end_time-loop_start_time,
          '\nTotal amount of simulations: ', int(belt_speed_difference_range/belt_difference_increment),
          '\nBelt speed difference range: ', belt_speed_difference_range,
          '\nBelt speed difference increment: ', belt_difference_increment
          )

def sbt_run_simulation():
    sbt_run_single_sim = str_to_bool(config.get('SimulationConfig', 'sbt_run_single_sim'))
    sbt_run_loop_sim = str_to_bool(config.get('LoopSettings', 'sbt_run_loop_sim'))

    if sbt_run_single_sim == True and sbt_run_loop_sim == False:
        sbt_simulate_treadmill(float(config.get('SimulationConfig', 'sbt_belt_speed_baseline_difference')),
                               float(config.get('SimulationConfig', 'sbt_belt_speed_set_difference')),
                               int(config.get('SimulationConfig', 'number_of_timesteps_per_simulation')),
                               str_to_bool(config.get('SimulationConfig', 'render_video_iff_no_loop')),
                               sbt_physics
                               )      
    elif sbt_run_loop_sim == True and sbt_run_single_sim == False:
        sbt_loop_simulate_treadmill(float(config.get('SimulationConfig', 'sbt_belt_speed_baseline_difference')),
                                    float(config.get('LoopSettings', 'sbt_loop_belt_speed_difference_range')),
                                    float(config.get('LoopSettings', 'sbt_loop_belt_speed_difference_increment'))
                                    )       
    else:
        raise Exception('sbt_run_single_sim and sbt_run_loop_sim must be different boolean values in config.ini')

if __name__ == '__main__':

    config = sbt_config_load()
    sbt_world, sbt_physics, sbt_task, sbt_env = sbt_initialize_dmcontrol()
    sbt_task.initialize_episode(sbt_physics)

    sbt_run_simulation()
