# A Matrix Approach for idopNetwork Reconstruction

We organize the data in a matrix form. Let 

$$
\boldsymbol{Y} = (y_j(i)) \in \mathbb{R}^{n \times m}
$$

be the observation matrix, where $y_j(i)$ denotes the value of variable/feature $j \, (j=1,\ldots,m)$ observed at sample/time $i \, (i=1,\dots,n)$. For each sample $i$, the observation vector is defined as

$$
\mathbf{y}_i =
\begin{bmatrix}
y_i(1) \\ y_i(2) \\ \vdots \\ y_i(m)
\end{bmatrix}
\in \mathbb{R}^m,
$$

which collects all variables measured on sample $i$. Accordingly, the entire dataset can be expressed as

$$
\boldsymbol{Y} =
\left(
\begin{array}{c}
\mathbf{y}_{1}^{\top} \\
\mathbf{y}_{2}^{\top} \\
\vdots \\
\mathbf{y}_{n}^{\top}
\end{array}
\right)
=
\left(
\begin{array}{cccc}
y_{1}(1) & y_{1}(2) & \cdots & y_{1}(m) \\
y_{2}(1) & y_{2}(2) & \cdots & y_{2}(m) \\
\vdots   & \vdots   & \ddots & \vdots   \\
y_{n}(1) & y_{n}(2) & \cdots & y_{n}(m)
\end{array}
\right)
\in \mathbb{R}^{n \times m}
$$

where rows represent samples and columns represent variables. This unified formulation abstracts the specific nature of the observational dimension, providing a common framework for subsequent modeling. Within this framework, static (cross-sectional) data can be transformed into a quasi-dynamic form, whereas dynamic (time-series) data can be directly applied without such transformation.
