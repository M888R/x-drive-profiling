from math import sin, cos, sqrt, atan2, inf, pi
import numpy as np
import scipy as sp
import scipy.special

def constrain_angle(theta):
    return atan2(sin(theta), cos(theta))

class Arc:
    def __init__(self, s = None, r = None, theta = None):
        if s is not None and r is not None:
            self.s = s
            self.r = r
            self.theta = s / r
        elif s is not None and theta is not None:
            self.s = s
            self.r = s / theta
            self.theta = theta
        elif r is not None and theta is not None:
            self.s = theta * r
            self.r = r
            self.theta = theta

class Bezier:
    def __init__(self, control_points, resolution = 1000):
        self.control_points = control_points
        self.resolution = resolution

    def point_at_t(self, t):
        t = np.clip(t, 0.0, 1.0)

        point = np.array([0.0, 0.0])

        n = len(self.control_points) - 1

        for i, ctrl_point in enumerate(self.control_points):
            point += (sp.special.comb(n, i) * pow(1.0 - t, n - i) * pow(t, i)) * ctrl_point

        return point

    def theta_at_t(self, t):
        t = np.clip(t, 0.0000001, 0.9999999)
        
        t_next = t + 0.00000001
        t_prev = t - 0.00000001

        point = self.point_at_t(t)
        point_next = self.point_at_t(t_next)
        point_prev = self.point_at_t(t_prev)

        dX1, dY1 = point_next - point
        dX2, dY2 = point - point_prev
        
        return constrain_angle((atan2(dY1, dX1) + atan2(dY2, dX1)) / 2.0)

    def arc_at_t(self, t):
        t = np.clip(t, 0.0000001, 0.9999999)
        step = 1.0 / self.resolution
        t_next = t + step

        point = self.point_at_t(t)
        point_next = self.point_at_t(t_next)

        dX, dY = point_next - point
        
        # Chord length
        d = sqrt(dX ** 2 + dY ** 2)
        d_theta = constrain_angle((self.theta_at_t(t_next) - self.theta_at_t(t)))

        # Arc radius
        r = inf if d_theta == 0 else d / (2.0 * sin(d_theta / 2.0))

        # arc length
        s = d if d_theta == 0 else d_theta * r

        return Arc(r = r, theta = d_theta, s = s)

    cached_dist = None

    def total_dist(self):
        if self.cached_dist is not None:
            return self.cached_dist

        dist = 0

        for i in np.linspace(0.0, 1.0, num = self.resolution, endpoint = False):
            dist += self.arc_at_t(i).s

        self.cached_dist = dist

        return dist

    def get_points(self):
        t = np.linspace(0.0, 1.0, num = self.resolution + 1)
        return np.array([self.point_at_t(x) for x in t])

    def get_arcs(self):
        t = np.linspace(0.0, 1.0, num = self.resolution, endpoint = False)
        return np.array([self.arc_at_t(x) for x in t])

    def get_curvature_at_dist(self, dist, saved_state = (None, None)):
        step = 1.0 / self.resolution
        
        initial_cursor, dist_explored = saved_state

        cursor = initial_cursor if initial_cursor is not None else 0.0
        dist_explored = dist_explored if dist_explored is not None else 0.0

        while dist_explored < dist:
            arc = self.arc_at_t(cursor * step)
            dist_explored += arc.s
            cursor += 1
            
        cursor -= 1
        arc = self.arc_at_t(cursor * step)

        saved_state = (cursor, dist_explored)

        return 1 / arc.r if arc.r != 0 else 0, saved_state

    def get_angle_at_dist(self, dist, saved_state = (None, None)):
        step = 1.0 / self.resolution
        
        initial_cursor, dist_explored = saved_state

        cursor = initial_cursor if initial_cursor is not None else 0.0
        dist_explored = dist_explored if dist_explored is not None else 0.0

        while dist_explored < dist:
            arc = self.arc_at_t(cursor * step)
            dist_explored += arc.s
            cursor += 1
            
        cursor -= 1
        arc = self.arc_at_t(cursor * step)
        dist_explored -= arc.s

        saved_state = (cursor, dist_explored)

        dist_explored -= arc.s
        desired_s = dist - dist_explored

        theta_1 = self.theta_at_t(cursor * step)

        if arc.theta == 0:
            return theta_1, saved_state

        return theta_1 + (desired_s / arc.s) * arc.theta, saved_state

    def get_point_at_dist(self, dist, saved_state = (None, None)):
        step = 1.0 / self.resolution
        
        initial_cursor, dist_explored = saved_state

        cursor = initial_cursor if initial_cursor is not None else 0.0
        dist_explored = dist_explored if dist_explored is not None else 0.0

        while dist_explored < dist:
            arc = self.arc_at_t(cursor * step)
            dist_explored += arc.s
            cursor += 1

        if dist == dist_explored:
            saved_state = (cursor, dist_explored)
            return self.point_at_t(cursor * step), saved_state
            
        cursor -= 1
        arc = self.arc_at_t(cursor * step)

        delta_pos = np.array([0.0, 0.0])

        dist_explored -= arc.s
        desired_s = dist - dist_explored

        if arc.theta == 0:
            # straight line
            # todo rotate this??
            diff = self.point_at_t(cursor * step) - self.point_at_t((cursor + 1) * step)
            saved_state = (cursor, dist_explored)
            return self.point_at_t(cursor * step) + diff * (desired_s / arc.s), saved_state

        
        arc_portion = Arc(s = desired_s, r = arc.r)

        delta_pos[0] = arc_portion.r - cos(-arc_portion.theta) * arc_portion.r
        delta_pos[1] = arc_portion.r * sin(-arc_portion.theta)

        dx, dy = self.point_at_t((cursor + 1) * step) - self.point_at_t(cursor * step)

        base_theta = constrain_angle(pi / 2 + atan2(dy, dx) - arc.theta / 2.0)

        rotation_matrix = np.array([
            [cos(base_theta), -sin(base_theta)],
            [sin(base_theta), cos(base_theta)]
        ])

        delta_pos = rotation_matrix.dot(delta_pos)

        saved_state = (cursor, dist_explored)
        return self.point_at_t(cursor * step) + delta_pos, saved_state
