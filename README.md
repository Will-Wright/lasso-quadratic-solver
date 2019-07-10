## Description

This software package solves the [lasso](https://en.wikipedia.org/wiki/Lasso_(statistics)) or [basis pursuit denoising](https://en.wikipedia.org/wiki/Basis_pursuit_denoising) problem 

`min ||Ax - b|| + lambda*|x|` 

and the box-constrained [quadratic program](https://en.wikipedia.org/wiki/Quadratic_programming) 

`min 1/2*x'*Q*x + b'*x subject to l <= x <= u`.


The main function `LassoQuadraticSolver` is based on a smoothing method which was presented in two recent papers

* [Forward–backward quasi-Newton methods for nonsmooth optimization problems](https://link.springer.com/article/10.1007/s10589-017-9912-y)

* [Augmented Lagrangians, box constrained QP and extensions](https://academic.oup.com/imajna/article/37/4/1635/3059683)

In the [qualifying exam proposal](https://github.com/Will-Wright/lasso-quadratic-solver/blob/master/will_wright_qualifying_exam_proposal.pdf) we prove that the two methods developed above are equal (note that the lasso problem must be in the Lagrangian form for this equivalence, see page 8-9 of the proposal).

### New Contributions:

* Proof of equivalence of two methods

* Implementation of `LassoQuadraticSolver` method in MATLAB

* Numerical results demonstrating that `LassoQuadraticSolver` is faster than the MATLAB built-in software [lasso](https://www.mathworks.com/help/stats/lasso.html) and [quadprog](https://www.mathworks.com/help/optim/ug/quadprog.html)



## Demo Tutorial





