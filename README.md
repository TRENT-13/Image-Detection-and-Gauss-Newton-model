numpy version 1.26.4

# Image-Detection-and-Gauss-Newton-model
Detect balls in an image, map them to a simulation, and allow users to specify a shooting location to hit a target ball.
Simulation consistes image datection using the canny filter and thresholding the background using the various size of kernels(the larger more details are gone/added)
after that we use gauss newton mehtod with the L2 regularization to get the best results for the shooting.
Innovative part is the adding extra optimzitaionn in this algorithm, which works relatively better for the stiff ODE cases, this gives us the edge

![image](https://github.com/user-attachments/assets/7c5f696d-df43-42a6-8b4c-8cdf16d74018)
![image](https://github.com/user-attachments/assets/36a85fb8-bd05-49a7-ba0f-6e751df45478)


![image](https://github.com/user-attachments/assets/2c5acee8-0051-4695-9c02-b903c3e92824)


![image](https://github.com/user-attachments/assets/786da56e-c30e-4ecd-a0b4-29e90bf94990)


![image](https://github.com/user-attachments/assets/e61a28b5-8d0c-470c-8d68-6658b80cdd8b)


![image](https://github.com/user-attachments/assets/3513e6b2-2035-481e-b622-b6dc370b36c1)

![image](https://github.com/user-attachments/assets/c4cf2940-6f0f-4293-8a6a-c80a4e838fb7)

![image](https://github.com/user-attachments/assets/6cde71fa-10eb-488e-9427-0943e0862242)



![image](https://github.com/user-attachments/assets/4b7a7924-c963-42e4-b96c-65ed81647357)



i use canny edge detection, main problem was to optimize for the noise pictures, for that i use different size of kernels, this kernels are tailored fit for these images may not work for other images with the noise,
core part of the code is this one, gauss newton's method + L2 regularization for the  stiff cases, it is some a prevention for these kind of cases because, we are adding the residual error to jacobi matrix and therefore we reduce the condition number which itself prevents these case
```
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
```

besides that i use RK-4 as the main method, i also have the comparision of other methods(adam-moulton(linear multistep), adam-bashfort and euler) compared to RK4


