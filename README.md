![image](https://github.com/user-attachments/assets/c6d8301d-c592-4ca8-b24e-bf26f2799e80)# Image-Detection-and-Gauss-Newton-model
Detect balls in an image, map them to a simulation, and allow users to specify a shooting location to hit a target ball.
Simulation consistes image datection using the canny filter and thresholding the background using the various size of kernels(the larger more details are gone/added)
after that we use gauss newton mehtod with the L2 regularization to get the best results for the shooting.
Innovative part is the adding extra optimzitaionn in this algorithm, which works relatively better for the stiff ODE cases, this gives us the edge


![image](https://github.com/user-attachments/assets/2c5acee8-0051-4695-9c02-b903c3e92824)
