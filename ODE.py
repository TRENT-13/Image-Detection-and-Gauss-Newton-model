import cv2
import numpy as np
import pygame
import math

from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BALL_RADIUS = 10
TARGET_RADIUS = 20  # Increased to match detected target size
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
desired_fps = 60
new_h = 1.0 / desired_fps

class BallShootingSimulation:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Image '{image_path}' not found or unable to read.")
        self.screen_width, self.screen_height = self.image.shape[1], self.image.shape[0]
        self.all_targets = []
        self.hit_targets = set()
        self.trajectory_memory = []
        self.GRAVITY = 9.81
        self.AIR_RESISTANCE = 0.1
        self.COLORS = {
            'rk4': (0, 0, 255),  # Blue
            'euler': (255, 0, 0),  # Red
            'ab': (0, 255, 0),  # Green
            'am': (255, 255, 0)  # Yellow
        }

    def detect_objects(self, image):
        # Convert to grayscale
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        kernel_mid = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        copy = np.array(image)

        blurred = cv2.GaussianBlur(copy, (11, 11), 0)

        # sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=15)
        # sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=15)
        #
        # magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        # magnitude = np.uint8(magnitude * 255 / np.max(magnitude))

        canny = cv2.Canny(image=cv2.convertScaleAbs(blurred), threshold1=150, threshold2=220)
        cv2.imshow('canny',canny)
        cleaned = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel_small, iterations=8)
        cv2.imshow('cleaned',cleaned)
        # cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_mid, iterations=8)
        # cv2.imshow('cleaned1',cleaned)
        # cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=8)
        # cv2.imshow('cleaned2', cleaned)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area:  # Adjust based on your image size
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                objects.append((center, radius))

        # Clustering remains the same
        if len(objects) > 0:
            centers = np.array([obj[0] for obj in objects])
            db = DBSCAN(eps=50, min_samples=1).fit(centers)
            labels = db.labels_

            final_objects = []
            for label in set(labels):
                if label != -1:
                    cluster_indices = [i for i, l in enumerate(labels) if l == label]
                    cluster_centers = centers[cluster_indices]
                    cluster_radii = [objects[i][1] for i in cluster_indices]

                    weights = [objects[i][1] for i in cluster_indices]
                    avg_center = tuple(map(int, np.average(cluster_centers, weights=weights, axis=0)))
                    avg_radius = int(np.mean(cluster_radii))
                    final_objects.append((avg_center, avg_radius))

            return final_objects

        return objects



    def check_collision(self, ball_pos, target_pos, ball_radius, target_radius):
        dx = ball_pos[0] - target_pos[0]
        dy = ball_pos[1] - target_pos[1]
        distance = math.sqrt(dx * dx + dy * dy)
        return distance < (ball_radius)

    def ode_system(self, state, t, g):
        """Ball motion ODE system: dx/dt = vx, dy/dt = vy, dvx/dt = -k*vx, dvy/dt = -g -k*vy"""
        x, y, vx, vy = state
        dvx_dt = self.AIR_RESISTANCE * vx
        dvy_dt = g + self.AIR_RESISTANCE * vy
        return np.array([vx, vy, dvx_dt, dvy_dt])

    def rk4_step(self, func, state, t, h, g):
        """Fourth-order Runge-Kutta method for solving the ODE system"""
        k1 = func(state, t, g)
        k2 = func(state + 0.5 * h * k1, t + 0.5 * h, g)
        k3 = func(state + 0.5 * h * k2, t + 0.5 * h, g)
        k4 = func(state + h * k3, t + h, g)
        return state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def euler_step(self, func, state, t, h, g):
        """Euler method for solving the ODE system"""
        derivatives = func(state, t, g)
        return state + h * derivatives

    def adams_bashforth_step(self, func, states, t, h, g):
        """Fourth-order Adams-Bashforth method"""
        if len(states) < 4:
            # Use RK4 to bootstrap if we don't have enough previous points
            return self.rk4_step(func, states[-1], t, h, g)

        f_n = func(states[-1], t, g)
        f_nm1 = func(states[-2], t - h, g)
        f_nm2 = func(states[-3], t - 2 * h, g)
        f_nm3 = func(states[-4], t - 3 * h, g)

        coefficients = np.array([55 / 24, -59 / 24, 37 / 24, -9 / 24])
        derivatives = np.array([f_n, f_nm1, f_nm2, f_nm3])

        return states[-1] + h * sum(c * f for c, f in zip(coefficients, derivatives))

    def adams_moulton_step(self, func, states, t, h, g):
        """Fourth-order Adams-Moulton method"""
        if len(states) < 3:
            # Use RK4 to bootstrap if we don't have enough previous points
            return self.rk4_step(func, states[-1], t, h, g)

        # Predict step using Adams-Bashforth
        y_predicted = self.adams_bashforth_step(func, states, t, h, g)

        # Correct step using Adams-Moulton
        f_np1 = func(y_predicted, t + h, g)
        f_n = func(states[-1], t, g)
        f_nm1 = func(states[-2], t - h, g)
        f_nm2 = func(states[-3], t - 2 * h, g)

        coefficients = np.array([9 / 24, 19 / 24, -5 / 24, 1 / 24])
        derivatives = np.array([f_np1, f_n, f_nm1, f_nm2])

        return states[-1] + h * sum(c * f for c, f in zip(coefficients, derivatives))

    def residual(self, p, x0, y0, x_target, y_target):
        """Calculate position residuals for given initial velocity"""
        vx0, vy0 = p
        initial_state = np.array([x0, y0, vx0, vy0])
        current_state = initial_state.copy()
        t = 0
        h = 0.01
        T = 7.0  # Reasonable time limit

        while t < T:
            current_state = self.rk4_step(self.ode_system, current_state, t, h, self.GRAVITY)
            t += h

            if abs(current_state[1] - y_target) < 0.1:
                break

        return np.array([current_state[0] - x_target, current_state[1] - y_target])

    def numerical_jacobian(self, p, x0, y0, x_target, y_target, epsilon=1e-6):
        """Calculate Jacobian matrix for velocity parameters"""
        f = self.residual(p, x0, y0, x_target, y_target)
        J = np.zeros((2, 2))

        for i in range(2):
            h = np.zeros(2)
            h[i] = epsilon
            f_plus = self.residual(p + h, x0, y0, x_target, y_target)
            f_minus = self.residual(p - h, x0, y0, x_target, y_target)
            J[:, i] = (f_plus - f_minus) / (2 * epsilon)
        return J

    def calculate_initial_velocity(self, start_pos, target_pos):
        x0, y0 = start_pos
        x_target, y_target = target_pos

        # Initial guess for velocities
        dx = x_target - x0
        dy = y_target - y0
        T_estimate = 2.0 * np.sqrt(abs(dy) / self.GRAVITY)
        v0_mag = np.sqrt(dx**2 + dy**2) / T_estimate
        angle = math.atan2(dy, dx)
        v0 = np.array([v0_mag * math.cos(angle), v0_mag * math.sin(angle)])

        max_iter = 20
        tol = 1e-6
        lambda_reg = 1e-6  # Regularization parameter

        for _ in range(max_iter):
            # Define residual for current parameters
            error = self.residual(v0, x0, y0, x_target, y_target)

            # Numerical Jacobian calculation
            J = self.numerical_jacobian(v0, x0, y0, x_target, y_target)

            # Compute delta with regularization
            try:
                delta = np.linalg.solve(
                    J.T @ J + lambda_reg * np.eye(2),
                    -J.T @ error
                )
            except np.linalg.LinAlgError:
                # Handle singular matrix
                break

            # Update parameters
            v0_new = v0 + delta

            # Check for convergence
            if np.linalg.norm(delta) < tol or np.linalg.norm(error) < tol:
                break

            v0 = v0_new

        return v0

    def calculate_comparison_trajectories(self, start_pos, target_pos, v0, h=new_h):
        """Calculate trajectories using different numerical methods"""
        x0, y0 = start_pos
        T = 7.0  # simulation time

        # Initialize states
        state = np.array([x0, y0, v0[0], v0[1]])

        # Initialize trajectory storage
        trajectories = {
            'rk4': [(x0, y0)],
            'euler': [(x0, y0)],
            'ab': [(x0, y0)],
            'am': [(x0, y0)]
        }

        # Initialize states for each method
        state_rk4 = state.copy()
        state_euler = state.copy()
        states_ab = [state.copy()]
        states_am = [state.copy()]

        t = 0
        while t < T:
            # RK4
            state_rk4 = self.rk4_step(self.ode_system, state_rk4, t, h, self.GRAVITY)
            trajectories['rk4'].append((state_rk4[0], state_rk4[1]))

            # Euler
            state_euler = self.euler_step(self.ode_system, state_euler, t, h, self.GRAVITY)
            trajectories['euler'].append((state_euler[0], state_euler[1]))

            # Adams-Bashforth
            new_state_ab = self.adams_bashforth_step(self.ode_system, states_ab, t, h, self.GRAVITY)
            states_ab.append(new_state_ab)
            if len(states_ab) > 4:
                states_ab.pop(0)
            trajectories['ab'].append((new_state_ab[0], new_state_ab[1]))

            # Adams-Moulton
            new_state_am = self.adams_moulton_step(self.ode_system, states_am, t, h, self.GRAVITY)
            states_am.append(new_state_am)
            if len(states_am) > 4:
                states_am.pop(0)
            trajectories['am'].append((new_state_am[0], new_state_am[1]))

            t += h

            # Stop if any trajectory goes off screen
            if (any(traj[-1][1] > self.screen_height or
                    traj[-1][0] < 0 or
                    traj[-1][0] > self.screen_width
                    for traj in trajectories.values())):
                break

        return trajectories

    def calculate_trajectory_errors(self, trajectories):
        """Calculate errors between different numerical methods and RK4 (reference solution)"""
        # Initialize error dictionaries
        position_errors = {
            'euler': [],
            'ab': [],
            'am': []
        }

        # Calculate errors at each time step
        rk4_trajectory = trajectories['rk4']

        # Ensure all trajectories have the same length for comparison
        min_length = min(len(trajectories[method]) for method in trajectories)

        for i in range(min_length):
            rk4_pos = np.array(rk4_trajectory[i])

            for method in position_errors.keys():
                method_pos = np.array(trajectories[method][i])
                # Calculate Euclidean distance between positions
                error = np.linalg.norm(method_pos - rk4_pos)
                position_errors[method].append(error)

        # Calculate summary statistics
        error_stats = {
            method: {
                'max_error': max(errors),
                'mean_error': np.mean(errors),
                'rms_error': np.sqrt(np.mean(np.array(errors) ** 2))
            }
            for method, errors in position_errors.items()
        }

        return position_errors, error_stats

    def plot_trajectories(self, all_trajectories, shot_number):
        """Plot trajectories and errors from different numerical methods"""
        # Calculate errors
        position_errors, error_stats = self.calculate_trajectory_errors(all_trajectories)

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

        # Plot trajectories
        ax1.plot(*zip(*all_trajectories['rk4']), 'b-', label='RK4 (reference)', linewidth=2)
        ax1.plot(*zip(*all_trajectories['euler']), 'r--', label='Euler', linewidth=2)
        ax1.plot(*zip(*all_trajectories['ab']), 'g:', label='Adams-Bashforth', linewidth=2)
        ax1.plot(*zip(*all_trajectories['am']), 'y-.', label='Adams-Moulton', linewidth=2)

        ax1.set_title(f'Trajectory Comparison - Shot {shot_number}')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.legend()
        ax1.grid(True)
        ax1.invert_yaxis()
        ax1.set_xlim(0, self.screen_width)
        ax1.set_ylim(self.screen_height, 0)

        # Plot errors
        time_steps = np.arange(len(next(iter(position_errors.values())))) * new_h
        for method, errors in position_errors.items():
            ax2.plot(time_steps, errors, label=f'{method} (RMS: {error_stats[method]["rms_error"]:.2e})')

        ax2.set_title('Position Errors Relative to RK4')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Error (pixels)')
        ax2.legend()
        ax2.grid(True)

        # Add error statistics as text
        stats_text = "Error Statistics (relative to RK4):\n"
        for method, stats in error_stats.items():
            stats_text += f"\n{method}:\n"
            stats_text += f"  Max Error: {stats['max_error']:.2e}\n"
            stats_text += f"  Mean Error: {stats['mean_error']:.2e}\n"
            stats_text += f"  RMS Error: {stats['rms_error']:.2e}\n"

        plt.figtext(0.95, 0.5, stats_text, fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8),
                    verticalalignment='center',
                    horizontalalignment='right')

        plt.tight_layout()
        plt.show()

    def run_game(self):
        pygame.init()
        screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        clock = pygame.time.Clock()

        # Previous initialization code remains the same...
        background = pygame.Surface((self.screen_width, self.screen_height))
        background.fill((0, 0, 0))

        detected_objects = self.detect_objects(self.image)
        self.all_targets = [(pos, radius) for pos, radius in detected_objects]
        remaining_targets = self.all_targets.copy()

        waiting_for_click = True
        shooting_pos = None
        shooting_radius = None

        while waiting_for_click:
            # Previous click handling code remains the same...
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    shooting_pos = event.pos
                    if remaining_targets:
                        distances = [(((t[0][0] - shooting_pos[0]) ** 2 +
                                       (t[0][1] - shooting_pos[1]) ** 2) ** 0.5, t[1])
                                     for t in remaining_targets]
                        shooting_radius = min(distances, key=lambda x: x[0])[1]
                    waiting_for_click = False

            screen.blit(background, (0, 0))
            for target_pos, radius in remaining_targets:
                pygame.draw.circle(screen, WHITE, target_pos, radius)
            pygame.display.flip()
            clock.tick(60)

        if not shooting_pos or shooting_radius is None:
            pygame.quit()
            return

        # Store trajectories for each shot
        shot_trajectories = []
        current_pos = shooting_pos
        current_radius = shooting_radius

        for i, (target_pos, target_radius) in enumerate(remaining_targets[:]):
            v0 = self.calculate_initial_velocity(current_pos, target_pos)

            # Calculate trajectories for all methods
            all_trajectories = self.calculate_comparison_trajectories(current_pos, target_pos, v0)
            shot_trajectories.append(all_trajectories)

            # Use RK4 trajectory for the actual simulation
            trajectory = all_trajectories['rk4']

            for ball_pos in trajectory:
                screen.blit(background, (0, 0))

                # Draw remaining targets
                for t_pos, t_radius in remaining_targets:
                    pygame.draw.circle(screen, WHITE, t_pos, t_radius)

                # Draw current trajectory
                points = [(int(p[0]), int(p[1])) for p in trajectory]
                if len(points) > 1:
                    pygame.draw.lines(screen, BLUE, False, points, 2)

                # Draw current ball
                pygame.draw.circle(screen, GREEN,
                                   (int(ball_pos[0]), int(ball_pos[1])), 10)

                pygame.display.flip()
                clock.tick(60)

                if self.check_collision(ball_pos, target_pos, current_radius, target_radius):
                    remaining_targets.remove((target_pos, target_radius))
                    current_radius = target_radius
                    break

        pygame.quit()

        # After simulation ends, plot all trajectories
        print("\nPlotting trajectories for all shots...")
        for i, trajectories in enumerate(shot_trajectories):
            self.plot_trajectories(trajectories, i + 1)


if __name__ == "__main__":
    simulation = BallShootingSimulation("2.jpg")
    simulation.run_game()
