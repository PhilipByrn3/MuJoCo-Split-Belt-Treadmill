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

def sbt_find_average_belt_time(single_belt_force_sim, time_data):
    #Finds the average steady velocity of the Rimless Wheel in refrence to the slow belt for each simulation
    start_time_list, end_time_list, time_pairs = [], [], []

    for z in range(len(single_belt_force_sim)):
        if single_belt_force_sim[z] == 490.5:
            single_belt_force_sim[z] = 0
        else:
            single_belt_force_sim[z] = 1

    #print(single_belt_force_sim)

    for k in range(len(single_belt_force_sim)):

        if single_belt_force_sim[k] == 1: 
            #print(k, single_belt_force_sim[k], '*****')
            pass

        elif single_belt_force_sim[k] == 0:
            #print(k, single_belt_force_sim[k])

            if sum(single_belt_force_sim[k:k+16]) != 0 and sum(single_belt_force_sim[k-16:k]) == 0 and single_belt_force_sim[k+1]==1:
                start_time_list.append(time_data[k+1])
                start_time = time_data[k+1]
                #print('stime=',start_time, k+1)

            if sum(single_belt_force_sim[k:k+16]) == 0 and sum(single_belt_force_sim[k-16:k]) != 0 and single_belt_force_sim[k-1]==1:
                end_time_list.append(time_data[k-1])
                end_time = time_data[k-1]
                #print('etime=', end_time, k-1)

    #print(start_time_list, end_time_list)


    if start_time_list[0] > end_time_list[0]:
        end_time_list.remove(end_time_list[0])
        
    
    for y in range(len(end_time_list)):
        time_pairs.append((start_time_list[y], end_time_list[y]))
    
    #print(time_pairs)
    time_difference = sum(end-start for start, end in time_pairs)
    time_average = time_difference/len(time_pairs)

    return time_average

def sbt_average_wheel_velocity(sbt_slow_belt_time, sbt_fast_belt_time, belt_diff):
    alpha = np.deg2rad(4.5)
    beta = np.deg2rad(15.5)
    length = 2.54 

    sinsum = np.sin(alpha) + np.sin(beta)

    timesum = sbt_slow_belt_time + sbt_fast_belt_time
    sbt_rimless_walk_speed = ((2 * length * sinsum) - (sbt_fast_belt_time * belt_diff)) / timesum

    # print(timesum, 2 * length * sinsum, sbt_fast_belt_time * belt_diff)

    return sbt_rimless_walk_speed 

def sbt_simulate_treadmill(base_belt_diff, set_belt_diff, timesteps, render_video, sbt_physics, single_slow_belt_force_sim ,single_fast_belt_force_sim, time_data):
    i = 0

    frames = []

    while i < timesteps:
        sbt_physics.step()
        sbt_physics.named.data.qvel['slow_belt_conveyor'] = base_belt_diff 
        sbt_physics.named.data.qvel['fast_belt_conveyor'] = sbt_physics.named.data.qvel['slow_belt_conveyor'] + set_belt_diff

        sbt_physics.named.data.qpos['fast_belt_conveyor'] = 0 
        sbt_physics.named.data.qpos['slow_belt_conveyor'] = 0 

        single_slow_belt_force_sim.append(float(sbt_physics.named.data.sensordata['slow_belt_force_sensor'][2].copy()))
        single_fast_belt_force_sim.append(float(sbt_physics.named.data.sensordata['fast_belt_force_sensor'][2].copy()))
        time_data.append(sbt_physics.data.time)

        if render_video is True:
            img = sbt_physics.render(width=640, height=480, camera_id='axle_cam')
            frames.append(img)

        i+=1

    if render_video is True:   
        sbt_render_video(frames)

    sbt_slow_belt_time = sbt_find_average_belt_time(single_slow_belt_force_sim, time_data)
    sbt_fast_belt_time = sbt_find_average_belt_time(single_fast_belt_force_sim, time_data)
   
    sbt_rimless_walk_speed = sbt_average_wheel_velocity(sbt_slow_belt_time, sbt_fast_belt_time, set_belt_diff)

    return sbt_rimless_walk_speed
    
def sbt_loop_simulate_treadmill(base_belt_diff, belt_speed_difference_range, belt_difference_increment, single_slow_belt_force_list, single_fast_belt_force_list,       sbt_belt_speed_baseline_difference):
    i = base_belt_diff
    loop_start_time = time.time()
    total_slow_belt_force, total_fast_belt_force, time_data = [], [], []

    while i <= belt_speed_difference_range:
        render_video_overwrite = False
        sim_start_time = time.time()
        sbt_rimless_walk_speed = sbt_simulate_treadmill(
                               float(config.get('SimulationConfig', 'sbt_belt_speed_baseline_difference')),
                               float(config.get('SimulationConfig', 'sbt_belt_speed_set_difference')),
                               int(config.get('SimulationConfig', 'number_of_timesteps_per_simulation')),
                               render_video_overwrite,
                               sbt_physics,
                               single_slow_belt_force_list,
                               single_fast_belt_force_list,
                               time_data                          
                               )

        sim_end_time = time.time()

        print('Simulation complete with difference of: ', round(i,4),  
              '\nTime Elapsed: ', sim_end_time-sim_start_time,
              '\nAverage Walk Speed: ', sbt_rimless_walk_speed,
              '\n----------')
        i += belt_difference_increment

    loop_end_time = time.time()

    print('Loop Simulation complete!',
          '\nTotal time elapsed: ', loop_end_time-loop_start_time,
          '\nTotal amount of simulations: ', float((belt_speed_difference_range-sbt_belt_speed_baseline_difference)/belt_difference_increment),
          '\nBelt speed difference range: ', belt_speed_difference_range,
          '\nBelt speed difference increment: ', belt_difference_increment
          )
    

    return total_slow_belt_force, total_fast_belt_force
    
    

def sbt_run_simulation():

    sbt_run_single_sim = str_to_bool(config.get('SimulationConfig', 'sbt_run_single_sim'))
    sbt_run_loop_sim = str_to_bool(config.get('LoopSettings', 'sbt_run_loop_sim'))

    single_slow_belt_force_list = []
    single_fast_belt_force_list = []
    time_data = []

    if sbt_run_single_sim == True and sbt_run_loop_sim == False:
        sbt_simulate_treadmill(float(config.get('SimulationConfig', 'sbt_belt_speed_baseline_difference')),
                               float(config.get('SimulationConfig', 'sbt_belt_speed_set_difference')),
                               int(config.get('SimulationConfig', 'number_of_timesteps_per_simulation')),
                               str_to_bool(config.get('SimulationConfig', 'render_video_iff_no_loop')),
                               sbt_physics,
                               single_slow_belt_force_list,
                               single_fast_belt_force_list,
                               time_data
                               )      
        
    elif sbt_run_loop_sim == True and sbt_run_single_sim == False:
        total_slow_belt_force, total_fast_belt_force = sbt_loop_simulate_treadmill(float(config.get('SimulationConfig', 'sbt_belt_speed_baseline_difference')),
                                    float(config.get('LoopSettings', 'sbt_loop_belt_speed_difference_range')),
                                    float(config.get('LoopSettings', 'sbt_loop_belt_speed_difference_increment')),
                                    single_slow_belt_force_list,
                                    single_fast_belt_force_list,
                                    float(config.get('SimulationConfig', 'sbt_belt_speed_baseline_difference'))
                                    )
       
    else:
        raise Exception('sbt_run_single_sim and sbt_run_loop_sim must be different boolean values in config.ini')
    
    # return total_slow_belt_force, total_fast_belt_force

if __name__ == '__main__':

    config = sbt_config_load()
    sbt_world, sbt_physics, sbt_task, sbt_env = sbt_initialize_dmcontrol()
    sbt_task.initialize_episode(sbt_physics)

    # total_slow_belt_force, total_fast_belt_force = sbt_run_simulation()
    sbt_run_simulation()

   