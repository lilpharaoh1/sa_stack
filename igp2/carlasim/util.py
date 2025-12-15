# Parts of this work is licensed under:
# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import carla
import re
import numpy as np


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []


def draw_waypoints(world, waypoints, z=0.5):
    """
    Draw a list of waypoints at a certain height given in z.

        :param world: carla.world object
        :param waypoints: list or iterable container with the waypoints to draw
        :param z: height in meters
    """
    for wpt in waypoints:
        wpt_t = wpt.transform
        begin = wpt_t.location + carla.Location(z=z)
        angle = np.radians(wpt_t.rotation.yaw)
        end = begin + carla.Location(x=np.cos(angle), y=np.sin(angle))
        world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=1.0)


def get_speed(vehicle: carla.Actor, ignore_z: bool = True):
    """
    Compute speed of a vehicle in Km/h.

    Args:
        vehicle: the vehicle for which speed is calculated
        ignore_z: Whether to ignore the velocity component in the z-axis.

    Returns:
        speed as a float in Km/h
    """
    vel = vehicle.get_velocity()
    if ignore_z:
        return 3.6 * np.sqrt(vel.x ** 2 + vel.y ** 2)
    else:
        return 3.6 * np.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

def world_to_ego_batch(xs, ys, agent_pose):
    cx, cy, heading = agent_pose
    dx = np.array(xs) - cx
    dy = np.array(ys) - cy
    theta = -(heading - np.pi/2)
    cos_h = np.cos(theta)
    sin_h = np.sin(theta)
    x_ego = cos_h * dx - sin_h * dy
    y_ego = sin_h * dx + cos_h * dy
    return x_ego, y_ego

def ego_to_world_batch(xs, ys, agent_pose):
    cx, cy, heading = agent_pose
    
    # Rotation by +heading (ego → world)
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    
    # Apply rotation
    x_rot = cos_h * np.array(xs) - sin_h * np.array(ys)
    y_rot = sin_h * np.array(xs) + cos_h * np.array(ys)
    
    # Apply translation
    x_world = x_rot + cx
    y_world = y_rot + cy
    
    return x_world, y_world
