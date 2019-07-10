function [results] = runtest_lasso(varargin)

% NOTE: solver failing to recover Ax = b, use following to see this
% norm(results.model.A*results.lqs_lasso.x - results.model.b)
%
% just debugged prox operator (project_soft_threshold)
%

import util.*;

ip = inputParser;

ip.addParameter('n', 500);
ip.addParameter('num_coords', 150);
ip.addParameter('num_constraints', 100);
ip.addParameter('seed', 0);
ip.addParameter('disp', 1);

ip.parse(varargin{:});


n = ip.Results.n;
num_coords = ip.Results.num_coords;
m = ip.Results.num_constraints;
seed = ip.Results.seed;
disp = ip.Results.disp;


rng(seed);


noise = 0.0;
A = normc(rand(m, n));
x_idx = randsample(n, num_coords, false);
x_sparse = zeros(n, 1);
x_sparse(x_idx) = rand(num_coords, 1);
x0 = x_sparse + noise*rand(n, 1);
b = A*x0;
lambda = 1;
lqs_lambda = lambda;
results.solution = struct('x_sparse', x_sparse, 'x0', x0, 'x_idx', x_idx);
results.model = struct('type', 'lasso', ...
                      'A', A, 'b', b, 'lambda', lqs_lambda);

tic;
%[x_ML_lasso] = lasso(A, b, 'Lambda', lambda);
[x_ML_lasso] = lasso(A, b);
results.ML_lasso.runtime = toc;
results.ML_lasso.x = x_ML_lasso;

opts = [];
opts.disp = disp;
opts.diff_tol = -1;
opts.apg_maxiters = 100;
opts.apg_tol = 1e-4;
opts.main_maxiters = 0;
opts.main_tol = 1e-10;
opts.Newton_step_min = 1e-4;
opts.cg_maxiters = 20;
opts.cg_tol = 1e-6;
%opts.x0 = zeros(n, 1);

AtA = A'*A; Atb = A'*b;
[x_temp, flag] = pcg(AtA, Atb, 1e-12, 100);
x_sort = sort(abs(x_temp), 'descend'); x_val = x_sort(num_coords); x_nonzeros = (x_temp >= x_val) + (x_temp <= -x_val);
x_idx = (x_nonzeros ~= 0);
x_temp = A(:,x_idx)\b;
x_temp2 = zeros(n,1);
x_temp2(x_idx) = x_temp;
opts.x0 = x_temp2;

tic;
[x_lqs, res, data] = LassoQuadraticSolver(results.model, opts);
time_LQS = toc;

results.LQS = struct('x', x_lqs, 'res', res, 'runtime', time_LQS, 'data', data);

results.obj_opt = get_objective_local(results.model, x0);
results.obj_matlab = get_objective_local(results.model, x_ML_lasso);
results.obj_lqs = get_objective_local(results.model, x_lqs);
if disp >= 1
   fprintf('\n        opt         lqs      Matlab   True Inactive\n')
   fprintf(' %1.4e  %1.4e  %1.4e           %5i\n', results.obj_opt, results.obj_lqs, results.obj_matlab, num_coords);
end

  
end

function [obj] = get_objective_local(model, x)
  if strcmp(model.type, 'box_qp')
    obj = 0.5*(x'*(model.H*x)) + model.f'*x;
  else
    obj = 0.5*norm(model.A*x - model.b)^2;% + model.lambda*norm(x, 1);
  end
end