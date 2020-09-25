from math import sin, cos, sqrt, atan2, inf, pi
import numpy as np
import cv2

class XDriveKinematics:
    def __init__(self, v_max, b, cap_velocity = True):
        self.v_max = v_max
        self.b = b
        self.cap_velocity = cap_velocity

    # todo: incorporate theta_dot, shouldn't be that hard, right?
    def calc_vel_limit(self, theta, yaw_rate = 0):
        base = abs(self.v_max * sqrt(2) / (abs(sin(theta)) + abs(cos(theta))))
        turn_scaled = base * abs((self.max_yaw_rate() - yaw_rate) / self.max_yaw_rate())
        return turn_scaled

    def max_yaw_rate(self):
        v_max = self.v_max
        return self.fk([0.0, 0.0, 0.0], [v_max, v_max, v_max, v_max])[2]

    # state vector (x, y, theta), control vector (v1, v2, v3, v4)
    def fk(self, x, u):
        if self.cap_velocity:
            u = np.clip(u, -self.v_max, self.v_max)

        v1, v2, v3, v4 = u

        theta = x[2]

        A = np.array([
            [cos(theta), -sin(theta), 0],
            [sin(theta), cos(theta), 0],
            [0, 0, 1]
        ])

        ootsqt = 1.0 / (2.0 * sqrt(2))

        B = np.array([
            [ootsqt, ootsqt, 0],
            [-ootsqt, ootsqt, 0],
            [0, 0, 1.0 / self.b]
        ])

        C = np.array([
            [1, 0, -1, 0],
            [0, -1, 0, 1],
            [0.25, 0.25, 0.25, 0.25]
        ])

        return np.linalg.multi_dot([A, B, C, u])

    # state vector (x, y, theta), control vector (x_dot, y_dot, theta_dot)
    def ik(self, x, u):
        x_dot, y_dot, theta_dot = u
        theta = x[2]

        A = np.array([
            [0.5, 0, 1],
            [0, -0.5, 1],
            [-0.5, 0, 1],
            [0, 0.5, 1]
        ])

        B = np.array([
            [cos(theta), sin(theta), 0],
            [-sin(theta), cos(theta), 0],
            [0, 0, 1]
        ])

        C = np.array([
            [sqrt(2), -sqrt(2), 0],
            [sqrt(2), sqrt(2), 0],
            [0, 0, self.b]
        ])
        
        output = np.linalg.multi_dot([A, B, C, u])
        return np.clip(output, -self.v_max, self.v_max) if self.cap_velocity else output
        
class EKSDRIVE:
    # x, y, theta
    X = np.array([0.0, 0.0, 0.0])
    X_dot = np.array([0.0, 0.0, 0.0])

    def __init__(self, v_max, b, cap_velocity):
        self.kinematics = XDriveKinematics(v_max, b, cap_velocity)

    # v1, v2, v3, v4
    def step(self, u):
        self.X_dot = self.kinematics.fk(self.X, u)
        # don't change this to +=. ye hath been warned
        self.X = self.X + self.X_dot

        return self.X

    # v1, v2, v3, v4
    def fk(self, u):
        return self.kinematics.fk(self.X, u)

    # x_dot, y_dot, theta_dot
    def ik(self, u):
        return self.kinematics.ik(self.X, u)

class XDriveRenderer:
    def __init__(self, xdrive, m_to_px, size = np.array([50, 50]), canvas_size = np.array([500, 500]), dt = 0.01):
        self.xdrive = xdrive
        self.size = size
        self.m_to_px = m_to_px
        self.canvas_size = canvas_size
        self.dt = dt

    def render(self, img):
        x, y, theta = self.xdrive.X
        pos = np.array([x, -y])

        scaled_pos = pos * self.m_to_px
        center = self.canvas_size / 2
        # left_corner = center - self.size / 2 + scaled_pos
        # bottom_right_corner = center + self.size / 2 + scaled_pos
        
        X_dot = self.xdrive.X_dot
        v = sqrt(X_dot[0] ** 2 + X_dot[1] ** 2) / self.dt
        
        cv2.putText(img, f'x: {x:.2f}, y: {y:.2f}, theta: {(theta * 180 / pi):.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        cv2.putText(img, f'v: {v:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        
        top_left_corner = np.array([
            center[0] + scaled_pos[0] + sin(-theta + pi / 4) * self.size[0],
            center[1] + scaled_pos[1] + cos(-theta + pi / 4) * self.size[1]
        ], dtype=np.int32)

        top_right_corner = np.array([
            center[0] + scaled_pos[0] + sin(-theta + pi * (3 / 4)) * self.size[0],
            center[1] + scaled_pos[1] + cos(-theta + pi * (3 / 4)) * self.size[1]
        ], dtype=np.int32)
        
        bottom_right_corner = np.array([
            center[0] + scaled_pos[0] + sin(-theta + pi * (5 / 4)) * self.size[0],
            center[1] + scaled_pos[1] + cos(-theta + pi * (5 / 4)) * self.size[1]
        ], dtype=np.int32)

        bottom_left_corner = np.array([
            center[0] + scaled_pos[0] + sin(-theta + pi * (7 / 4)) * self.size[0],
            center[1] + scaled_pos[1] + cos(-theta + pi * (7 / 4)) * self.size[1]
        ], dtype=np.int32)

        pts = np.array([top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner])

        cv2.polylines(img, np.int32([pts]), True, (255, 100, 0))

        # cv2.rectangle(img, tuple(left_corner.astype(int)), tuple(bottom_right_corner.astype(int)), (255, 0, 0))