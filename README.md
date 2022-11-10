### **Proximal-Gradient**
The proximal gradient methods in julia. Credit to \<A Fast Iterative-Thresholding Algorithm for Linear Inverse Problem\> by Amir Beck and Marc Teboule. The implementations is based on their paper and the book by Amir \<First-Order method for Optimization\> is an essential reference for the theoretical background of this repo. 


---
### **How to Run**

Run [setup.jl](setup.jl) with julia. And then you can start running the examples we have. 

---
### **The Proximal Gradient**

Smooth and nonsmooth function with gradient and proximal oracles are defined as types. The Proximal Gradient algorithm with adaptive stepsize and Nesterov Accelerations is impelmented in one function in [this file](./src/proximal_gradient.jl). 

---
### **Inverse Linear**

We consider the LASSO Problem applied to image deblurring. The LASSO problem is simply: 

$$
\arg\min_{x}\left\lbrace
    \Vert Ax - b\Vert^2_2 + \lambda \Vert x\Vert_1
\right\rbrace, 
$$

in which we solve it using the FISTA algorithm with $A$ being the discretized Guassian Blurr matrix. The example is in [here](applications/Inverse_linear.jl). The sparse matrix multiplication is slow in Julia, I don't have mental strength to optimize the speed yet. It's proved in Beck book that the convergence is first order for both the accelerated and unaccelerated case. 

