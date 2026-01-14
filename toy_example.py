import jax
import jax.numpy as jnp

import numpy as np

from tgpr_cv import TGPR_CV

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


class PIController:
    def __init__(self, kp, ki):
        self.kp = kp
        self.ki = ki
        self.integral = 0.0

    def control(self, error, dt=0.1, cap=None):
        self.integral += error * dt
        output = self.kp * error + self.ki * self.integral
        if cap is not None:
            output = np.clip(output, -cap, cap)
        return output


class Vehicle:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0

    def update(self, vel, dt=0.1):
        self.vx = vel[0]
        self.vy = vel[1]
        self.x += self.vx * dt
        self.y += self.vy * dt


def simulate_forward(state, vehicle, PI_x, PI_y, next_point, measurement_noise, dt=0.1):
    x_meas = state[0] + np.random.normal(0, measurement_noise)
    y_meas = state[1] + np.random.normal(0, measurement_noise)
    # if i > 0:
    #     theta_est = np.arctan2(y_meas - measurements[i-1][1], x_meas - measurements[i-1][0])

    obs = np.array([x_meas, y_meas])
    error_y = next_point[1] - state[1]
    error_x = next_point[0] - state[0]

    vel_x = PI_x.control(error_x, cap=1.0, dt=dt)
    vel_y = PI_y.control(error_y, cap=1.0, dt=dt)
    vel = np.array([vel_x, vel_y])
    print(f'Velocity command: {vel}')

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


# Initiale
memory = 10
dt = 0.1
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

dataset = np.zeros((0, 2))

steps = 500

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

vehicle = Vehicle()
PI_x = PIController(2.0, 0.0)
PI_y = PIController(2.0, 0.0)

for i in range(steps):
    print(f'Step {i+1}/{steps}')
    print(f'Dataset size: {dataset.shape[0]}')
    obs, state = simulate_forward(state, vehicle, PI_x, PI_y, next_point, meas_noise, dt=dt)
    dataset = np.vstack((dataset, obs))

    if dataset.shape[0] > memory:
        dataset = dataset[1:, :]

    dist = np.linalg.norm(state[:2] - np.array(next_point))
    print(f'Distance to next point: {dist}')
    if dist < 0.5:
        if traj_counter >= len(trajectory[0]) - 1:
            traj_counter = 0
        traj_counter += 1
        next_point = (trajectory[0][traj_counter], trajectory[1][traj_counter])

    if dataset.shape[0] < memory:
        continue

    tgpr.measurements = dataset

    predicted_traj, predicted_cov = tgpr.predict_trajectory(dt=0.1, pred_horizon=20)

    # Visualization
    if i % 1 == 0:
        ax.clear()
        ax.set_aspect('equal')
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.plot(trajectory[0], trajectory[1], color='gray', linestyle='--')
        ax.plot(dataset[:, 0], dataset[:, 1], 'bo', label='Measurements')
        ax.plot(state[0], state[1], 'ro', label='Current Position')
        ax.plot(next_point[0], next_point[1], 'go', label='Next Target Point')
        ax.plot(predicted_traj[:, 0], predicted_traj[:, 1], 'r-', label='Predicted Trajectory')
        # uncertainty ellipses
        for k in range(predicted_cov.shape[0]):
            # extract 2x2 covariance for position
            cov_k = predicted_cov[k, :2, :2]

            eigvals, eigvecs = np.linalg.eigh(cov_k)  # symmetric PSD
            order = eigvals.argsort()[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]

            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            width, height = 2 * np.sqrt(eigvals)  # 1-sigma
            
            ellipse = Ellipse(
                xy=(predicted_traj[k, 0], predicted_traj[k, 1]),
                width=width,
                height=height,
                angle=angle,
                alpha=0.15,
                color="red",
                zorder=1
            )
            ax.add_patch(ellipse)
        ax.legend()
        plt.pause(0.01)

