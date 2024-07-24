

# Detection of block structures in Matrix Sparsity Patterns

## Repository Link

[GitHub repo "opencampus-preconditioner-ai-project"](https://github.com/24io/opencampus-preconditioner-ai-project)


## Background

Many real-world phenomena are continuous in nature and characterised by a multitude of variables that
interact in complex ways. Such phenomena are typically modelled using sets of partial differential equations
(PDEs) which often do not have closed-form solutions [^1]. A prime example of this complexity is found in
Computational Fluid Dynamics (CFD), where the behaviour of fluids is simulated accounting for factors such
as viscosity, turbulence, and pressure gradients [^2]. In CFD, the continuous flow of fluids is modelled using
the Navier-Stokes equations, a set of PDEs describing the motion of fluid substances [^3]. These equations,
while elegant in their continuous form, are extremely difficult to solve analytically for all but the simplest
cases [^4]. To address this challenge, numerical methods such as Finite Volume (FV), Finite Element (FE),
or Finite Difference (FD) are employed to discretise the continuous domain into a finite number of points or
cells, allowing for the computation of approximate solutions [^6] [^7] [^8].

The discretisation process transforms the continuous fluid flow problem into a large system of algebraic
equations, which can be represented as a matrix equation of the form:

```
A * x = b
```
Where:
- `A` is a sparse `n * n` matrix
- `x` is a length `n` -vector of unknown variables (i.e. velocity, pressure) 
- `b` is a length `n` -vector of known values (i.e. related to boundary conditions and source terms) 

Each row in the matrix `A`  typically corresponds to an equation for a specific cell in the grid, while the columns
represent the influence of neighbouring cells. As each cell primarily interacts with its immediate
neighbours, most entries in this matrix are zero. Sparse matrices are beneficial with regard to computational efficiency, 
as specialised storage formats can be employed to reduce memory requirements, allowing for the handling of much larger 
problems than would be feasible
with dense matrix representations [^1]. 

Due to the multiple variables associated with each grid cell, A often exhibits a blockdiagonal structure which
can be represented as:

$$A = 
\begin{bmatrix}
a_{1,1} & a_{1,2} & \cdots & a_{1,n} \\
a_{2,1} & a_{2,2} & \cdots & a_{2,n} \\
\vdots  & \vdots  & \ddots & \vdots  \\
a_{m,1} & a_{m,2} & \cdots & a_{m,n} 
\end{bmatrix}$$

where each $a_{ii}$ is a square block matrix corresponding to the coupling between variables within a single 
cell or a small group of cells.

## Motivation

The complexity of solving the system `Ax = b` directly can be as high as $O(N^{3})$ [^10] [^11] [^12] [^13]. For large systems, 
this can lead to significant computational costs and limitations in terms of memory usage. 

In (FE and FV) simulations, a common approach is to approximate the solution of the matrices by iteratively refining an 
initial guess until a predefined convergence criterion is satisfied [^14] [^15]. For iterative solving 
algorithms such as GMRES, each iteration has a complexity of $O(N^{2})$. 
Hence, depending on the size of the matrix and the number of iterations required, these methods can offer substantial 
computational savings over direct methods, particularly if the matrices are sparse [^14]. 
If the matrices are illconditioned, however, the number of iterations required to find a solution within an acceptable 
error margin can quickly become computationally prohibitive. 

To accelerate the convergence of an iterative solver, a preconditioner matrix `P` can be applied to both sides of the 
equation, where  $P \approx A^{-1}$. Thus, the original system `Ax + b` is transformed into a system 
$PAx = Pb$, whereby `PA` and `Pb` are ideally cheap to compute and have a more favourable eigenvalue 
distribution than the original matrix `A`. 

We then solve the resulting system:
```
A' * x = b'
```

If `P` is cleverly chosen then the number iterations for solving this system will be significantly smaller. The goal is 
thereby to cluster the eigenvalues of `PA` around 1 and away from zero, thereby reducing the condition number and improving the convergence rate of the iterative solver

## Block Jacobi Preconditioner
The Block Jacobi preconditioner is a natural extension of the classical Jacobi method, tailored to handle the block 
structure often encountered in CFD problems [^5]. It is particularly well-suited for systems where variables are tightly 
coupled within local regions but less so across the entire domain. 

Although more sophisticated preconditioners outperform Jacobian methods with regard to their effectiveness [^8], 
the inherent parallelism of the block-Jacobi preconditioner allows for efficient distribution of computational workload 
across multiple processors or nodes in a high-performance computing cluster [^9] [^10] [^5]. This positions the latter at 
an advantage for large-scale problems.

The Block Jacobi preconditioner `P` can be constructed as:

$$P = 
\begin{bmatrix}
A_{11}^{-1} & 0 & \cdots & 0 \\
0 & A_{22}^{-1} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & A_{nn}^{-1}
\end{bmatrix}$$


where each $A_{i,i}^{-1}$ represents the inverse of a diagonal block of the original matrix `A`.

In a parallel implementation, each processor can be assigned one or more blocks, computing the local inverse and applying it to the corresponding part of the vector without needing to communicate with other processors. 
This locality of computation significantly reduces inter-processor communication overhead, which is often a bottleneck in parallel algorithms. 
Moreover, the Block Jacobi preconditioner aligns well with domain decomposition strategies commonly used in CFD, where the computational domain is divided into subdomains. Each subdomain can naturally correspond to a block in the preconditioner, preserving the physical and numerical relationships within the local region [^5].


### Problem Statement

Identifying the related blocks in a matrix can be challenging, particularly when noise blocks are present and the source of the equations is unknown. Building upon the research conducted by [Götz & Anzt (2018)](https://sc18.supercomputing.org/proceedings/workshops/workshop_files/ws_lasalss102s2-file1.pdf), our objective is to boost the convergence of the Generalised Minimal Residual (GMRES) solver, which is commonly used in conjunction with the Block Jacobi preconditioner [Hoekstra2022,BrownBull2020]. The algorithm aims to solve sparse systems of linear equations of the form 'Ax = b', where 'A' is a non-singular 'n x n' matrix, and 'x' and 'b' are vectors of length 'n'. Specifically, GMRES operates by iteratively minimising the residual norm over expanding Krylov subspaces until it became smaller than the predefined convergence criterion. 

Similar to [Gotz2018], we use predictive techniques to determine the location of diagonal blocks inside our sparse matrices. This enables us to rapidly identify and implement the block Jacobi preconditioner. Thus, our initial step involves replicating the findings outlined in their publication using a Convolutional Neural Network (CNN). Subsequently, we extend their research by conducting a comparative analysis of various model designs, encompassing graph representations. The objective is for every matrix to accurately forecast the initiation of each block. Given that a matrix can include several blocks, where each point can either indicate the start of a block or not, we are faced with a multi-label binary classification problem. 

This paper is organised as follows: To begin, we present a comprehensive summary of existing research on the topic. Next, we elucidate our experimental setup as well as some technical details relevant to our work. Subsequently, we proceed to explore each stage of the process methodologically, beginning with data generation and concluding with the modelling process. Finally, we analyse our findings and outline potential avenues for future research. 


### Task Type

Multi-Label Binary Classification

### Results Summary

- **Best Model:** CNN
- **Evaluation Metric:** Accuracy, F1
- **Result:** [95% accuracy, F1-score of 0.73]


## Cover Image

![Project Cover Image](CoverImage/cover_image.png)

## References

[^1]: Botsch, M., Bommes, D., & Kobbelt, L. (2005). Efficient Linear System Solvers for Mesh Processing. In Mathematics of Surfaces XI (pp. 62-83). Springer Berlin Heidelberg. [https://doi.org/10.1007/978-3-540-31835-4_4](https://doi.org/10.1007/978-3-540-31835-4_4)

[^2]: Skotniczny, M., Paszyńska, A., Rojas, S., & Paszyński, M. (2024). Complexity of direct and iterative solvers on space–time formulations and time-marching schemes for h-refined grids towards singularities. Journal of Computational Science, 76, 102216. [https://doi.org/10.1016/j.jocs.2024.102216](https://doi.org/10.1016/j.jocs.2024.102216)

[^3]: Langer, U., & Zank, M. (2021). Efficient Direct Space-Time Finite Element Solvers for Parabolic Initial-Boundary Value Problems in Anisotropic Sobolev Spaces. SIAM Journal on Scientific Computing, 43, A2714-A2736. [https://doi.org/10.1137/20M1358128](https://doi.org/10.1137/20M1358128)

[^4]: Knott, G. (2012). Gaussian Elimination and LU-Decomposition.

[^5]: Hoekstra, R. (2022). Parallel Block Jacobi Preconditioned GMRES for Dense Linear Systems (Master's thesis, Eindhoven University of Technology, Department of Mathematics and Computer Science). [https://pure.tue.nl/ws/portalfiles/portal/303534687/Hoekstra_R.pdf](https://pure.tue.nl/ws/portalfiles/portal/303534687/Hoekstra_R.pdf)

[^6]: Embree, M. (1999). How Descriptive are GMRES Convergence Bounds?

[^7]: Eisenträger, S., Atroshchenko, E., & Makvandi, R. (2020). On the condition number of high order finite element methods: Influence of p-refinement and mesh distortion. Computers & Mathematics with Applications, 80(11), 2289-2339. [https://doi.org/10.1016/j.camwa.2020.05.012](https://doi.org/10.1016/j.camwa.2020.05.012)

[^8]: Zhu, Y., & Sameh, A. H. (2016). How to Generate Effective Block Jacobi Preconditioners for Solving Large Sparse Linear Systems. In Advances in Computational Fluid-Structure Interaction and Flow Simulation: New Methods and Challenging Computations (pp. 231-244). Springer International Publishing. [https://doi.org/10.1007/978-3-319-40827-9_18](https://doi.org/10.1007/978-3-319-40827-9_18)

[^9]: Gotz, M., & Anzt, H. (2018). Machine Learning-Aided Numerical Linear Algebra: Convolutional Neural Networks for the Efficient Preconditioner Generation. In 2018 IEEE/ACM 9th Workshop on Latest Advances in Scalable Algorithms for Large-Scale Systems (scalA) (pp. 49-56). [https://doi.org/10.1109/ScalA.2018.00010](https://doi.org/10.1109/ScalA.2018.00010)

[^10]: Ghai, A., Lu, C., & Jiao, X. (2016). A Comparison of Preconditioned Krylov Subspace Methods for Nonsymmetric Linear Systems. Numerical Linear Algebra with Applications, 26. [https://doi.org/10.1002/nla.2215](https://doi.org/10.1002/nla.2215)

