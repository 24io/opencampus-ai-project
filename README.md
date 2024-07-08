# Detection of block structures in Matrix Sparsity Patterns

## Repository Link

[GitHub repo "opencampus-preconditioner-ai-project"](https://github.com/24io/opencampus-preconditioner-ai-project)

## Description

Given a matrix `A` of dimensions `m * m` and vectors `x`, `b` of length `m` we want to solve the system:
```
A * x = b
```
An often used approach in (finite element) simulations ist to approximate the result using an iterative method.
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
