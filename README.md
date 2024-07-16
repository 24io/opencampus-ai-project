# Detection of block structures in Matrix Sparsity Patterns

## Repository Link

# Detection of block structures in Matrix Sparsity Patterns

## Repository Link

[GitHub repo "opencampus-preconditioner-ai-project"](https://github.com/24io/opencampus-preconditioner-ai-project)


## Background

Many real-world phenomena are continuous in nature and characterised by a multitude of variables that
interact in complex ways. Such phenomena are typically modelled using sets of partial differential equations
(PDEs) which often do not have closed-form solutions [1]. A prime example of this complexity is found in
Computational Fluid Dynamics (CFD), where the behaviour of fluids is simulated accounting for factors such
as viscosity, turbulence, and pressure gradients [2]. In CFD, the continuous flow of fluids is modelled using
the Navier-Stokes equations, a set of PDEs describing the motion of fluid substances [3]. These equations,
while elegant in their continuous form, are extremely difficult to solve analytically for all but the simplest
cases [4] 1 . To address this challenge, numerical methods such as Finite Volume (FV), Finite Element (FE),
or Finite Difference (FD) are employed to discretise the continuous domain into a finite number of points or
cells, allowing for the computation of approximate solutions [6–8].

The discretisation process transforms the continuous fluid flow problem into a large system of algebraic
equations, which can be represented as a matrix equation of the form:

```
A * x = b
```
Where:
- `A` is a sparse `n * n` matrix
- `x` is a length `n` -vector of unknown variables (i.e. velocity, pressure) 
- `b` is a length `n` -vector of known values (i.e. related to boundary conditions and source terms) 

Each row in the matrix A typically corresponds to an equation for a specific cell in the grid, while the columns
represent the influence of neighbouring cells. As each cell primarily interacts with its immediate
neighbours, most entries in this matrix are zero. Sparse matrices are beneficial with regard to computational efficiency, 
as specialised storage formats can be employed to reduce memory requirements, allowing for the handling of much larger 
problems than would be feasible
with dense matrix representations [1]. 

Due to the multiple variables associated with each grid cell, A often exhibits a block-diagonal structure which
can be represented as:

$$A = 
\begin{bmatrix}
a_{1,1} & a_{1,2} & \cdots & a_{1,n} \\
a_{2,1} & a_{2,2} & \cdots & a_{2,n} \\
\vdots  & \vdots  & \ddots & \vdots  \\
a_{m,1} & a_{m,2} & \cdots & a_{m,n} 
\end{bmatrix}$$

where each  <td>a<sub>1,1</sub></td> is a square block matrix corresponding to the coupling between variables within a single cell or 
a small group of cells.

Although much research goes into optimising existing algorithms to solve systems of linear equations efficiently, 
the complexity of direct solving methods, i.e. Gaussian elimination  or LU factorisation, can be as high 
as $O(N^{3})$ \cite{Botsch2005,SKOTNICZNY2024102216,Langer2021,Knott2012}. For large systems, this can lead to 
significant computational costs and limitations in terms of memory usage. 

In FE and FV simulations, a common approach is to approximate the solution of the matrices by iteratively refining an 
initial guess until a predefined convergence criterion is satisfied \cite{Hoekstra2022,Embree1999}. For iterative solving 
algorithms such as GMRES, which is introduced in Section 3, each iteration has a complexity of $O(N^{2})$. Hence, depending on the size of the matrix and the number of iterations required, these methods can offer substantial computational savings over direct methods, particularly if the matrices are sparse \cite{Hoekstra2022}. 
If the matrices are ill-conditioned, however, the number of iterations required to find a solution within an acceptable 
error margin can quickly become computationally prohibitive. This is often the case in CFD, where scale differences and coupled phenomena (e.g., velocity-pressure coupling, fluid-thermal interactions) result in an unfavourable distribution of eigenvalues and high condition numbers \cite{EISENTRAGER2020,Botsch2005, Hoekstra2022}. 

An often used approach in (FE) simulations ist to approximate the result using an iterative method.
However, the number of iterations required for solving these systems wil be quite large for most matrices.


To improve this convergence speed a preconditioner matrix `P` can be applied to both sides of the equations:
```
P * A * x = P * b
```
Here `P * A` and `P * b` can be easily computed. We then solve the resulting system:
```
A' * x = b'
```
If `P` is cleverly chosen then the number iterations for solving this system will be significantly smaller.

A common preconditioner can be a Block-Jacobi-Inverse. Here blocks of variables are identified in the Matrix `A`.
Only these blocks are inverted which commonly can be done quite fast when their dimensions ar significantly smaller than
the dimensions of the matrix itself.

The tricky part is identifying the connected blocks in a matrix, especially when noisy blocks not representing true data
are also present in the matrix.

Since the matrices can be treated as images, this is an ideal job for a convolutional neural network (CNN).

We try to reproduce the model presented in [Götz & Anzt (2018)](
https://sc18.supercomputing.org/proceedings/workshops/workshop_files/ws_lasalss102s2-file1.pdf) to learn about the
pitfalls of implementing the data generation, training and fine-tuning the model, and finally experiment with changing
the parameters of the generated matrices to find the limits of our models.

### Task Type

Multi-Label Binary Classification

### Results Summary

- **Best Model:** CNN
- **Evaluation Metric:** Accuracy, F1
- **Result:** [95% accuracy, F1-score of 0.73]

## Documentation

1. **[Literature Review](0_LiteratureReview/README.md)**
2. **[Dataset Characteristics](1_DatasetCharacteristics/exploratory_data_analysis.ipynb)**
3. **[Baseline Model](2_BaselineModel/baseline_model.ipynb)**
4. **[Model Definition and Evaluation](3_Model/model_definition_evaluation)**
5. **[Presentation](4_Presentation/README.md)**

## Cover Image

![Project Cover Image](CoverImage/cover_image.png)
