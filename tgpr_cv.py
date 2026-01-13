import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag, solve
from functools import partial

class TGPR_CV:
    """
    Trajectory Gaussian Process Regression with Constant Velocity Model

    State: [x, y, vx, vy] - position and velocity
    Measurements: [x, y] - position only

    No control inputs needed - velocities are part of the state.
    """

    def __init__(self,
                 dataset_history: int = 10,
                 sigma_a: float = 0.5,  # acceleration uncertainty
                 C_single: jnp.ndarray = None,
                 K0: jnp.ndarray = None,
                 R: jnp.ndarray = None,
                 dt: float = 0.1,
                 key: jax.random.PRNGKey = jax.random.PRNGKey(0)):
        """
        Args:
            dataset_history (int): Number of past timesteps in dataset
            sigma_a (float): Process noise intensity (acceleration std dev)
            C_single (jnp.ndarray): Measurement matrix (2x4), observes [x,y] from [x,y,vx,vy]
            K0 (jnp.ndarray): Initial state covariance (4x4)
            R (jnp.ndarray): Measurement noise covariance (2x2)
            dt (float): Time step
            key: JAX random key
        """
        self._max_hist = dataset_history
        self._measurements = jnp.empty((0, 2))  # Only x, y positions
        self._current_state = jnp.array([0.0, 0.0, 0.0, 0.0])  # [x, y, vx, vy]

        # Process noise parameter
        self.sigma_a = sigma_a

        # Default measurement matrix: observe position only
        if C_single is None:
            C_single = jnp.array([[1.0, 0.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0, 0.0]])
        self.C_single = C_single

        # Default initial covariance
        if K0 is None:
            K0 = jnp.diag(jnp.array([0.01, 0.01, 0.1, 0.1]))  # higher uncertainty on velocities
        self.K0 = K0

        # Default measurement noise
        if R is None:
            R = jnp.eye(2) * 0.01
        self.R = R

        self.dt = dt
        self.key = key

        # Trajectory storage
        self.x_bar_traj = jnp.empty((0, 4))
        self._x_traj = jnp.empty((0, 4))
        self._k_traj = jnp.empty((0, 4, 4))

        # Precompute big matrices
        R_inv = jnp.kron(jnp.eye(self._max_hist), jnp.linalg.inv(self.R))
        self.R_inv = R_inv
        C_big = jnp.kron(jnp.eye(self._max_hist), self.C_single)
        self.C_big = C_big

    @property
    def dataset_size(self) -> int:
        return self._measurements.shape[0]

    @property
    def measurements(self) -> jnp.ndarray:
        return self._measurements

    @measurements.setter
    def measurements(self, value: jnp.ndarray):
        """Set measurements. Expected shape: (N, 2) for [x, y] positions"""
        self._measurements = value

    @property
    def prior(self) -> jnp.ndarray:
        return self.x_bar_traj

    @property
    def predicted_trajectory(self) -> jnp.ndarray:
        """Returns predicted trajectory of shape (pred_horizon+1, 4)"""
        return self._x_traj

    @property
    def predicted_covariances(self) -> jnp.ndarray:
        """Returns covariance matrices of shape (pred_horizon+1, 4, 4)"""
        return self._k_traj

    @property
    def current_state(self) -> jnp.ndarray:
        """Current estimated state [x, y, vx, vy]"""
        return self._current_state

    @property
    def current_pose(self) -> jnp.ndarray:
        """For compatibility with original API - returns position [x, y]"""
        return self._current_state[:2]

    def predict_trajectory(self, dt: float, pred_horizon: int) -> jnp.ndarray:
        """
        Main prediction function using constant velocity model.

        Args:
            dt (float): Time step
            pred_horizon (int): Number of future timesteps to predict

        Returns:
            jnp.ndarray: Predicted state trajectory (pred_horizon+1, 4)
        """
        if self.dataset_size < 1:
            # No measurements yet, return zeros
            return jnp.zeros((pred_horizon + 1, 4))

        # Estimate initial state with velocity from position measurements
        x0_with_velocity = self._estimate_initial_state(self._measurements, dt)

        # Rollout the prior using CV model
        self.x_bar_traj = self.prior_rollout_cv(x0_with_velocity, dt)

        x_bar = self.x_bar_traj.reshape(-1)

        # Get state transition matrices (all identical for CV model)
        F = self._F_cv(dt)
        Phi_list = jnp.tile(F[None, :, :], (self.dataset_size - 1, 1, 1))

        Q = self._Q_cv(dt)
        Q_list = jnp.tile(Q[None, :, :], (self.dataset_size - 1, 1, 1))

        # Build lifted system
        A_lift = self._A_lift(Phi_list)
        Q_big = block_diag(self.K0, *Q_list)

        # Prior covariance
        K = A_lift @ Q_big @ A_lift.T + jnp.eye(A_lift.shape[0]) * 1e-8
        K_inv = jnp.linalg.inv(K)

        # Measurements
        y = self._measurements.reshape(-1)

        # Gaussian Process Regression
        x_est_flat, Sigma_post = self._gpr(K_inv, self.C_big, self.R_inv, x_bar, y)
        x_est = x_est_flat.reshape(-1, 4)

        # Update current state
        self._current_state = x_est[-1]
        self.K0 = Sigma_post[-4:, -4:]

        # Forward prediction
        x_future, K_future = self._predict(self._current_state, self.K0, dt, pred_horizon)

        # Include current state as first point
        self._x_traj = jnp.vstack([self._current_state[None, :], x_future])
        self._k_traj = jnp.concatenate([self.K0[None, :, :], K_future], axis=0)

        return self._x_traj

    def _estimate_initial_state(self, poses: jnp.ndarray, dt: float) -> jnp.ndarray:
        """
        Estimate initial state [x, y, vx, vy] from position measurements.
        Uses finite difference to estimate velocities.
        """
        if poses.shape[0] < 2:
            # Only one measurement, assume zero velocity
            return jnp.array([poses[0, 0], poses[0, 1], 0.0, 0.0])

        # Use multiple measurements to get better velocity estimate
        n_samples = min(3, poses.shape[0])
        vx_samples = []
        vy_samples = []

        for i in range(n_samples - 1):
            dx = poses[i + 1, 0] - poses[i, 0]
            dy = poses[i + 1, 1] - poses[i, 1]
            vx_samples.append(dx / dt)
            vy_samples.append(dy / dt)

        vx = jnp.mean(jnp.array(vx_samples))
        vy = jnp.mean(jnp.array(vy_samples))

        return jnp.array([poses[0, 0], poses[0, 1], vx, vy])

    @partial(jax.jit, static_argnums=(0, 4,))
    def _predict(self, x_last: jnp.ndarray, K_last: jnp.ndarray, 
                 dt: float, pred_horizon: int) -> tuple:
        """
        Roll out the CV model forward over prediction horizon.

        Args:
            x_last: Last estimated state (4,)
            K_last: Last covariance (4, 4)
            dt: Time step
            pred_horizon: Number of steps to predict

        Returns:
            x_traj: Predicted states (pred_horizon, 4)
            K_traj: Predicted covariances (pred_horizon, 4, 4)
        """
        def step(carry, _):
            x_k, K_k = carry # 

            F = self._F_cv(dt)
            Q = self._Q_cv(dt)

            # Mean propagation: x_{k+1} = F * x_k
            x_next = F @ x_k

            # Covariance propagation: K_{k+1} = F * K_k * F^T + Q
            K_next = F @ K_k @ F.T + Q

            new_carry = (x_next, K_next)
            outputs = (x_next, K_next)
            return new_carry, outputs

        init_carry = (x_last, K_last)
        (_, _), (x_traj, K_traj) = jax.lax.scan(
            step, init_carry, xs=None, length=pred_horizon
        )

        return x_traj, K_traj

    @partial(jax.jit, static_argnums=(0,))
    def _gpr(self, K_inv: jnp.ndarray, C_big: jnp.ndarray, 
             R_inv: jnp.ndarray, x_bar: jnp.ndarray, 
             measurements: jnp.ndarray) -> tuple:
        """
        Perform Gaussian Process Regression to refine trajectory predictions.

        Solves: (K^{-1} + C^T R^{-1} C) x = K^{-1} x_bar + C^T R^{-1} y
        """
        x_bar = x_bar.ravel()
        measurements = measurements.ravel()

        # Information form update
        H = K_inv + C_big.T @ R_inv @ C_big
        b = K_inv @ x_bar + C_big.T @ R_inv @ measurements

        x_est = solve(H, b)
        Sigma_post = jnp.linalg.inv(H)

        return x_est, Sigma_post

    def prior_rollout_cv(self, x0: jnp.ndarray, dt: float) -> jnp.ndarray:
        """
        Rollout constant velocity model from initial state.

        Args:
            x0: Initial state [x, y, vx, vy]
            dt: Time step

        Returns:
            State trajectory (N, 4) where N = dataset_size
        """
        N = self._measurements.shape[0]
        x_bar = jnp.zeros((N, 4))
        x_bar = x_bar.at[0, :].set(x0)

        F = self._F_cv(dt)

        def body_fun(i, x_bar):
            x_prev = x_bar[i - 1, :]
            x_next = F @ x_prev
            x_bar = x_bar.at[i, :].set(x_next)
            return x_bar

        x_bar = jax.lax.fori_loop(1, N, body_fun, x_bar)
        return x_bar

    def _F_cv(self, dt: float) -> jnp.ndarray:
        """
        State transition matrix for constant velocity model.

        x_{k+1} = F * x_k where x = [x, y, vx, vy]

        Returns:
            F: State transition matrix (4, 4)
        """
        return jnp.array([[1.0, 0.0, dt,  0.0],
                          [0.0, 1.0, 0.0, dt ],
                          [0.0, 0.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0]])

    def _Q_cv(self, dt: float) -> jnp.ndarray:
        q_c = self.sigma_a ** 2  # here interpret sigma_a^2 as continuous-time PSD
        Q1 = jnp.array([[dt**3/3, dt**2/2],
                        [dt**2/2, dt]]) * q_c
        Q = jnp.block([[Q1, jnp.zeros((2,2))],
                    [jnp.zeros((2,2)), Q1]])
        Q += 1e-9 * jnp.eye(4)
        return Q


    @partial(jax.jit, static_argnums=(0,))
    def _A_lift(self, Phi: jnp.ndarray) -> jnp.ndarray:
        """
        Construct lifting matrix from state transitions.

        Maps initial state and process noise to trajectory:
        [x_0, x_1, ..., x_M]^T = A_lift * [x_0, w_0, w_1, ..., w_{M-1}]^T

        Args:
            Phi: State transition matrices (M, N, N)

        Returns:
            A_lift: Lifting matrix ((M+1)*N, (M+1)*N)
        """
        M, N, _ = Phi.shape
        I = jnp.eye(N, dtype=Phi.dtype)

        # Initialize with identity blocks on diagonal
        A = jnp.zeros((M + 1, M + 1, N, N), dtype=Phi.dtype)
        A = A.at[jnp.arange(M + 1), jnp.arange(M + 1)].set(I)

        # Fill lower triangular part
        def outer(i, A_blocks):
            Phi_im1 = Phi[i - 1]
            def inner(j, A_blocks):
                # A[i,j] = Phi_{i-1} @ A[i-1,j]
                return A_blocks.at[i, j].set(Phi_im1 @ A_blocks[i - 1, j])
            return jax.lax.fori_loop(0, i, inner, A_blocks)

        A = jax.lax.fori_loop(1, M + 1, outer, A)

        # Reshape to matrix form
        return A.transpose(0, 2, 1, 3).reshape((M + 1) * N, (M + 1) * N)
