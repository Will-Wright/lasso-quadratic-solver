## Description

This software package solves the [lasso](https://en.wikipedia.org/wiki/Lasso_(statistics)) or [basis pursuit denoising](https://en.wikipedia.org/wiki/Basis_pursuit_denoising) problem 

`min ||Ax - b|| + lambda*|x|` 

and the box-constrained [quadratic program](https://en.wikipedia.org/wiki/Quadratic_programming) 

`min 1/2*x'*Q*x + b'*x subject to l <= x <= u`.


Our algorithm `LassoQuadraticSolver` is based on the two papers

* [Forward–backward quasi-Newton methods for nonsmooth optimization problems](https://link.springer.com/article/10.1007/s10589-017-9912-y)

* [Augmented Lagrangians, box constrained QP and extensions](https://academic.oup.com/imajna/article/37/4/1635/3059683)
