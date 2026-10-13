This project is a Python implementation of the Finite Element Method (FEM) to solve the 2D Poisson equation ($\nabla^2 u = f$) over user-defined polygonal domains. 
The implementation was inspired by the article by Alberty, Carstensen, and Funken cited in the long-form documentation and guided by 
and w.


The long-form pdf documentation of the project, explaining the solver in more depth is available [here](dokumentace_FEM_2d.pdf)

In short, the solver supports these features: 

* **Custom Polygonal Domains**: Generates triangular meshes over arbitrary 2D polygons using Gmsh.
* **Mixed Boundary Conditions**: Supports both Dirichlet and Neumann boundary conditions.
* **Sparse Matrix Assembly**: Assembles global stiffness matrices and solves reduced sparse linear systems using SciPy.
* **Visualization**: Plots 2D solution heatmaps with mesh overlays, 2D gradient vector fields, and smoothed 3D surface visualizations.

The dependencies of the solver are listed in [requirements.txt](requirements.txt).
