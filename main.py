import jax
import jax.numpy as jnp

import numpy as np

from tgpr_cv import TGPR_CV

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


class PIController:
    def __init__(self, kp, ki, max_speed=None):
        self.kp = kp
        self.ki = ki
        self.integral = 0.0
        self.max_speed = max_speed

    def control(self, error, dt=0.1):
        self.integral += error * dt
        output = self.kp * error + self.ki * self.integral
        if self.max_speed is not None:
            output = np.clip(output, -self.max_speed, self.max_speed)
        return output

class Vehicle:
    def __init__(self, max_speed=1.0, x0=0.0, y0=0.0, vx0=0.0, vy0=0.0):
        self.x = x0
        self.y = y0
        self.vx = vx0
        self.vy = vy0
        self.max_speed = max_speed

    def update(self, vel, dt=0.1):
        self.vx = vel[0]
        self.vy = vel[1]
        self.x += self.vx * dt
        self.y += self.vy * dt

    def reachability_range(self, dt=0.1):
        return self.max_speed * dt


def simulate_forward(state, vehicle, PI_x, PI_y, next_point, measurement_noise, dt=0.1):
    x_meas = state[0] + np.random.normal(0, measurement_noise)
    y_meas = state[1] + np.random.normal(0, measurement_noise)

    obs = np.array([x_meas, y_meas])
    error_y = next_point[1] - state[1]
    error_x = next_point[0] - state[0]

    vel_x = PI_x.control(error_x, dt=dt)
    vel_y = PI_y.control(error_y, dt=dt)
    vel = np.array([vel_x, vel_y])

    vehicle.update(vel, dt=dt)
    state[0] = vehicle.x
    state[1] = vehicle.y
    state[2] = vehicle.vx
    state[3] = vehicle.vy

    return obs, state


def circle_trajectory(num_points, radius=1.0):
    t = np.linspace(0, 2 * np.pi, num_points)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    return x, y

def compute_ring(pos, heading, radius, num_viewpoints=8):
    # Generate viewpoints with with first one along heading direction
    viewpoints = []
    for i in range(num_viewpoints):
        angle = heading + i * (2 * np.pi / num_viewpoints)
        x_vp = pos[0] + radius * np.cos(angle)
        y_vp = pos[1] + radius * np.sin(angle)
        viewpoints.append([x_vp, y_vp, i])
    viewpoints = np.array(viewpoints)
    # sort viewpoints by angle
    viewpoints = viewpoints[np.argsort(np.arctan2(viewpoints[:,1] - pos[1], viewpoints[:,0] - pos[0]))]
    return viewpoints


def prediction_step(dt, steps_ahead, tgpr, pursuer, visited_indexes, r_off, num_viewpoints, reachability_range):
    predicted_traj, predicted_cov = tgpr.predict_trajectory(dt=dt, pred_horizon=steps_ahead)

    predicted_traj = predicted_traj[steps_ahead, :]
    predicted_cov = predicted_cov[steps_ahead, :, :]

    cov_k = predicted_cov[:2, :2]

    eigvals, eigvecs = np.linalg.eigh(cov_k)  # symmetric PSD
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * np.sqrt(eigvals)  # 1-sigma

    ellipse = Ellipse(
        xy=(predicted_traj[0], predicted_traj[1]),
        width=width,
        height=height,
        angle=angle,
        alpha=0.15,
        color="red",
        zorder=1
    )

    r_unc = max(width, height) / 2.0
    r_capture = r_off + r_unc

    heading = np.arctan2(predicted_traj[3], predicted_traj[2])

    viewpoints = compute_ring(predicted_traj[:2], heading, r_capture, num_viewpoints=num_viewpoints)

    # reachable viewpoints
    reachable_viewpoints = []
    for point in viewpoints:
        dist_pursuer = np.linalg.norm(point[:2] - np.array([pursuer.x, pursuer.y]))
        if dist_pursuer <= reachability_range:
            reachable_viewpoints.append(point)
    reachable_viewpoints = np.array(reachable_viewpoints)

    # Randomly pick one among reachable viewpoints not yet visited and store visited indexes
    mask = ~np.isin(reachable_viewpoints[:, 2], visited_indexes)
    feasible_viewpoints = reachable_viewpoints[mask]

    selected_index = 0

    if len(feasible_viewpoints) > 0:
        selected_index = np.random.randint(len(feasible_viewpoints))
        target_point = feasible_viewpoints[selected_index]
        visited_indexes.append(target_point[2])
    else:
        print('All viewpoints visited, resetting visited list.')
        target_point = None

    return ellipse, predicted_traj, predicted_cov, viewpoints, feasible_viewpoints, target_point

# Initiale
memory = 10
dt = 0.1
seconds_ahead = 0.5
steps_ahead = int(seconds_ahead / dt)  # 10
"""
meas_noise: high = more noisy measurements, trust less the measurements
sigma_a: how much acceleration changes, high = more agile target
"""
meas_noise = 0.05
sigma_p = 2 * meas_noise
sigma_v = meas_noise / dt
sigma_a = 0.5
K0 = jnp.diag(jnp.array([sigma_p**2, sigma_p**2, sigma_v**2, sigma_v**2]))
tgpr = TGPR_CV(dataset_history=memory, sigma_a=sigma_a, dt=dt, R=jnp.eye(2)*(meas_noise**2), K0=K0)

limits = [0, 10, 0, 10]
center = np.array([5.0, 5.0])

# Generate circular trajectory data
num_points = 100
x, y = circle_trajectory(num_points, radius=3.0)
x = x + center[0]
y = y + center[1]
trajectory = (x, y)
traj_counter = 0
next_point = (trajectory[0][traj_counter], trajectory[1][traj_counter])

state = np.array([0.0, 0.0, 0.0, 0.0])  # x, y, vx, vy
state_pursuer = np.array([3.0, 2.0, 0.0, 0.0])  # x, y, vx, vy
dataset = np.zeros((0, 2))

steps = 500

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

max_speed = 1.0
max_speed_pursuer = 5.0
vehicle = Vehicle(max_speed=max_speed, x0=state[0], y0=state[1], vx0=state[2], vy0=state[3])
pursuer = Vehicle(max_speed=max_speed_pursuer, x0=state_pursuer[0], y0=state_pursuer[1], vx0=state_pursuer[2], vy0=state_pursuer[3]) 
PI_x = PIController(2.0, 0.0, max_speed=max_speed)
PI_y = PIController(2.0, 0.0, max_speed=max_speed)
PI_x_p = PIController(7.0, 0.0, max_speed=max_speed_pursuer)
PI_y_p = PIController(7.0, 0.0, max_speed=max_speed_pursuer)
r_off = 0.5

num_viewpoints = 8
visited_indexes = []

reachability_range = pursuer.reachability_range(dt=seconds_ahead)

filled = False
i = 0
est_i = 0
while i < steps:    
    print(f'Step {i+1}/{steps}')
    
    ax.clear()
    ax.set_aspect('equal')
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    
    obs, state = simulate_forward(state, vehicle, PI_x, PI_y, next_point, meas_noise, dt=dt)
    dataset = np.vstack((dataset, obs))

    if dataset.shape[0] > memory:
        dataset = dataset[1:, :]

    dist = np.linalg.norm(state[:2] - np.array(next_point))

    if dist < 0.5:
        if traj_counter >= len(trajectory[0]) - 1:
            traj_counter = 0
        traj_counter += 1
        next_point = (trajectory[0][traj_counter], trajectory[1][traj_counter])

    if dataset.shape[0] < memory:
        print('Filling dataset...')
        i = 0
        continue

    tgpr.measurements = dataset

    
    if est_i == 0 or i % steps_ahead == 0:
        est_i += 1
        ellipse, predicted_traj, predicted_cov, viewpoints, feasible_viewpoints, target_point = prediction_step(
            dt, steps_ahead, tgpr, pursuer, visited_indexes, r_off, num_viewpoints, reachability_range
        )
    if target_point is None:
        plt.show()
        break
    ax.plot(pursuer.x, pursuer.y, 'co', label='Pursuer Position', zorder=10)
    # Circle of reachability
    reach_circle = plt.Circle((pursuer.x, pursuer.y), reachability_range, color='cyan', fill=False, linestyle='--', label='Pursuer Reachability')
    ax.add_artist(reach_circle)
 
    # pursuer.x = target_point[0]
    # pursuer.y = target_point[1]
    
    dist = np.linalg.norm(state_pursuer[:2] - np.array(target_point[:2]))
    error_y_p = target_point[1] - pursuer.y
    error_x_p = target_point[0] - pursuer.x

    vel_x_p = PI_x_p.control(error_x_p, dt=dt)
    vel_y_p = PI_y_p.control(error_y_p, dt=dt)
    vel_p = np.array([vel_x_p, vel_y_p])

    pursuer.update(vel_p, dt=dt)
    state_pursuer = np.array([pursuer.x, pursuer.y, pursuer.vx, pursuer.vy])

    i += 1

    # Visualization
    if i % 1 == 0:
        ax.plot(trajectory[0], trajectory[1], color='gray', linestyle='--')
        ax.plot(dataset[:, 0], dataset[:, 1], 'bo', label='Measurements')
        ax.plot(state[0], state[1], 'ko', label='Current Position')
        ax.plot(next_point[0], next_point[1], 'go', label='Next Target Point')
        ax.plot(predicted_traj[0], predicted_traj[1], 'ro', label='Predicted Trajectory')
        ax.plot(viewpoints[:, 0], viewpoints[:, 1], 'mx', label='Capture Ring Viewpoints', zorder=1)
        ax.plot(feasible_viewpoints[:, 0], feasible_viewpoints[:, 1], 'cx', label='Reachable Viewpoints', zorder=2)
        ax.plot(target_point[0], target_point[1], 'r*', markersize=15, label='Selected Target Viewpoint', zorder=3)
        # Plot visited indexes
        for idx in visited_indexes:
            vp = viewpoints[viewpoints[:, 2] == idx][0]
            ax.plot(vp[0], vp[1], 'mo', markersize=12, label='Visited Viewpoint' if idx == visited_indexes[0] else "")

        ax.add_patch(ellipse)
        # ax.legend(loc='upper left')
        plt.pause(0.1)

