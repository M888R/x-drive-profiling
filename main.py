from xdrive import XDriveKinematics, EKSDRIVE, XDriveRenderer
from bezier import Bezier
from math import sqrt, pi
import numpy as np
import matplotlib.pyplot as plt
import cv2

path = Bezier([np.array([0.0, 0.0]), np.array([-1.26, 0.96]), np.array([1.61, 4.21]), np.array([-1.8, 1.98])], 50)
path_reversed = Bezier([np.array([-1.8, 1.98]), np.array([1.61, 4.21]), np.array([-1.26, 0.96]), np.array([0.0, 0.0])], 50)
# path = Bezier([np.array([0.0, 0.0]), np.array([4.0, 0.0])])
# path_reversed = Bezier([np.array([4.0, 0.0]), np.array([0.0, 0.0])])

canvas_size = np.array([800, 800])

m_to_px = 130
dt = 0.01 # 10ms

drive = EKSDRIVE(1.05, 0.35, False)
drive_renderer = XDriveRenderer(drive, m_to_px, np.array([30, 30]), canvas_size, dt)

class ProfileStep:
    def __init__(self, pose, velocity, acceleration):
        self.pose = pose
        self.velocity = velocity
        self.acceleration = acceleration
        
max_accel = 5.0
max_vel = 1.05 * sqrt(2)
dong_rate = 1.0

total_distance = path.total_dist()

def calculate_deccel_dist():
    angle_cursor = (None, None)

    velocity_multiplier = 0.0
    velocity = 0
    linear_distance_covered = 0

    last_angle = 0.0

    # accel
    while velocity_multiplier < 1.0:
        angle, new_angle_cursor = path_reversed.get_angle_at_dist(linear_distance_covered, angle_cursor)
        angle_cursor = new_angle_cursor

        velocity_multiplier += max_accel * dt * (1.0 / max_vel)
        velocity = drive.kinematics.calc_vel_limit(angle + last_angle, dong_rate) * velocity_multiplier
        last_angle += dong_rate * dt
        linear_distance_covered += velocity * dt
    
    return linear_distance_covered

# Profile calculations
deccel_dist = total_distance - calculate_deccel_dist()

step_cursor = (None, None)
angle_cursor = (None, None)

def next():
    global step_cursor
    global angle_cursor

    point, new_step_cursor = path.get_point_at_dist(linear_distance_covered, step_cursor)
    angle, new_angle_cursor = path.get_angle_at_dist(linear_distance_covered, angle_cursor)

    step_cursor = new_step_cursor
    angle_cursor = new_angle_cursor

    return point, angle

profile_steps = []

velocity_multiplier = 0.0
velocity = 0
linear_distance_covered = 0.0

# accel
while velocity_multiplier < 1.0:
    point, angle = next()

    last_pose = profile_steps[len(profile_steps) - 1].pose if len(profile_steps) != 0 else np.array([0.0, 0.0, 0.0])

    pose = np.array([point[0], point[1], last_pose[2] + dong_rate * dt])
    profile_steps.append(ProfileStep(pose, velocity, max_accel))    

    velocity_multiplier += max_accel * dt * (1.0 / max_vel)
    velocity = drive.kinematics.calc_vel_limit(angle + last_pose[2], dong_rate) * velocity_multiplier
    linear_distance_covered += velocity * dt

velocity_multiplier = 1.0

# cruise
while linear_distance_covered < deccel_dist:
    point, angle = next()

    last_pose = profile_steps[len(profile_steps) - 1].pose if len(profile_steps) != 0 else np.array([0.0, 0.0, 0.0])
    pose = np.array([point[0], point[1], last_pose[2] + dong_rate * dt])

    profile_steps.append(ProfileStep(pose, velocity, max_accel))    

    velocity = drive.kinematics.calc_vel_limit(angle + last_pose[2], dong_rate) * velocity_multiplier
    linear_distance_covered += velocity * dt

while velocity >= 0:
    point, angle = next()

    last_pose = profile_steps[len(profile_steps) - 1].pose if len(profile_steps) != 0 else np.array([0.0, 0.0, 0.0])
    pose = np.array([point[0], point[1], last_pose[2] + dong_rate * dt])

    profile_steps.append(ProfileStep(pose, velocity, max_accel))    

    velocity_multiplier -= max_accel * (1.0 / max_vel) * dt
    velocity = drive.kinematics.calc_vel_limit(angle + last_pose[2], dong_rate) * velocity_multiplier
    linear_distance_covered += velocity * dt

velocity = 0
linear_distance_covered = total_distance

# point, new_step_cursor = path.get_point_at_dist(linear_distance_covered, step_cursor)
# pose = np.array([point[0], point[1], 0.0])
# profile_steps.append(ProfileStep(pose, velocity, -max_accel))   

img = np.zeros((canvas_size[0], canvas_size[1], 3), np.uint8)

for step in profile_steps:
    scaled_x = step.pose[0] * m_to_px
    scaled_y = step.pose[1] * m_to_px
    center_x, center_y = canvas_size / 2
    img[int(center_y - scaled_y), int(center_x + scaled_x)] = (0, 255, 255)

def draw():
    tmp = img.copy()

    drive_renderer.render(tmp)

    cv2.imshow('Simu-dingler', tmp)
    cv2.waitKey(int(1000 * dt))

last_pose = np.array([0.0, 0.0, 0.0])

for step in profile_steps:
    pose = step.pose

    pose_dot = pose - last_pose
    last_pose = pose

    u = drive.ik(pose_dot / dt)
    drive.step(u * dt)
    draw()

cv2.waitKey(1500)