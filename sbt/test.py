from dm_control import suite
from dm_control import viewer
from dm_control.rl.control import Environment
from dm_control.suite.base import Task
from dm_control.mujoco import Physics
from dm_control import mujoco


import numpy as np
from moviepy.editor import ImageSequenceClip
from PIL import Image

from sbtdata import ConstructData, findTime, AverageSteadyVelocity



class SbTreadmillTask(Task):

    def __init__(self):
        super().__init__()
    
    def get_observation(self, physics):
        return None
    
    def get_reward(self, physics):
        return None
    
    def action_spec(self, physics):
        physics.named.data.qvel['redconveyor'] = 1
        physics.named.data.qvel['blueconveyor'] = 1.5



def render_video(frames):
    clip = ImageSequenceClip(frames, fps=100)
    clip.write_videofile('my_video.mp4')



if __name__ == '__main__':

    sb_physics = mujoco.Physics.from_xml_path("rimlesstreadmill.xml")
    sb_task = SbTreadmillTask()
    sb_env = Environment(physics=sb_physics, task=sb_task)
    action_spec = sb_env.action_spec()

    frames = []
    fprdat_raw = []
    fpbdat_raw = []
    timedata = []

    i = 0
    belt_diff = 0.5

    while i < 1700:
        sb_physics.step()
        sb_physics.named.data.qvel['redconveyor'] = 1 # Constant value red (slow) belt
        sb_physics.named.data.qvel['blueconveyor'] = sb_physics.named.data.qvel['redconveyor'] + belt_diff # Blue (fast) belt velocity = Red belt velocity + belt_diff

        if (sb_physics.data.qpos[0] > 10) or (sb_physics.data.qpos[1] > 10):
            sb_physics.data.qpos[0] = 0 # Red belt
            sb_physics.data.qpos[1] = 0 # Blue Belt
    
        fprdat_raw.append(sb_physics.named.data.sensordata[0].copy())
        fpbdat_raw.append(sb_physics.named.data.sensordata[1].copy())
        timedata.append(sb_physics.data.time)

        img = sb_physics.render(width=640, height=480,camera_id=0)
        frames.append(img)
        i+=1

    datList = ConstructData(timedata, fprdat_raw, fpbdat_raw)
    fprdat, fprdat_shift, fpbdat, fpbdat_shift = datList

    print(fprdat, fpbdat)

    # time_slow = findTime(fprdat, fprdat_shift)
    # time_fast = findTime(fpbdat, fpbdat_shift)

    render_video(frames)
