# Literature Review

Approaches or solutions that have been tried before on similar projects.

**Summary of Each Work**:

- **Source 1**: Machine learning-aided numerical linear algebra: 
Convolutional neural networks for the efficient preconditioner generation

  - **[Link](https://sc18.supercomputing.org/proceedings/workshops/workshop_files/ws_lasalss102s2-file1.pdf)**
  - **Objective**:  
  The authors trained a CNN to detect coupled variables in a sparse matrix.
  The main goal was to reduce the time to improve convergence (and thus speed) of a generalized minimal residual
  method GMRES solver.  
  - **Methods**:
  A CNN with several convolutions and a single dense layer followed by a dropout (0.1) were used.
  The model was trained on synthetic data since for matrices these are easy to produce. 
  - **Outcomes**:
  The model is able to recognize coupled variables (blocks) in the input matrices.
  - **Relation to the Project**:
  The model can be used as a starting point to get a proper grip of the topic and its implementation. 
  add skeleton for first paper literature review.

<!-- cited as (example):
    M. Götz and H. Anzt, "Machine learning-aided numerical linear algebra: 
    Convolutional neural networks for the efficient preconditioner generation",
    Proc. IEEE/ACM 9th Workshop Latest Adv. Scalable Algorithms Large-Scale Syst.,
    pp. 49-56, Nov. 2018.
-->
  
- **Source 2**: [Title of Source 2]

  - **[Link]()**
  - **Objective**:
  - **Methods**:
  - **Outcomes**:
  - **Relation to the Project**:

- **Source 3**: [Title of Source 3]

  - **[Link]()**
  - **Objective**:
  - **Methods**:
  - **Outcomes**:
  - **Relation to the Project**:
