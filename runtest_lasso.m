function [results] = runtest_lasso()

% NOTE: solver failing to recover Ax = b, use following to see this
% norm(results.model.A*results.sqs_lasso.x - results.model.b)
%
% just debugged prox operator (project_soft_threshold)
%

rng(1);

n = 500;
num_coords = 10;
m = 100;
noise = 0.0;
A = normc(rand(m, n));
x_idx = randsample(n, num_coords, false);
x_sparse = zeros(n, 1);
x_sparse(x_idx) = ones(num_coords, 1);
x0 = x_sparse + noise*rand(n, 1);
b = A*x0;
lambda = 1;
sqs_lambda = lambda;
results.solution = struct('x_sparse', x_sparse, 'x0', x0, 'x_idx', x_idx);
results.model = struct('type', 'lasso', ...
                      'A', A, 'b', b, 'lambda', sqs_lambda);

tic;
%[x_ML_lasso] = lasso(A, b, 'Lambda', lambda);
[x_ML_lasso] = lasso(A, b);
toc
results.ML_lasso = struct('x', x_ML_lasso);

opts = [];
opts.disp = 1;
opts.diff_tol = -1;
opts.apg_maxiters = 1000;
opts.apg_tol = 1e-4;
opts.main_maxiters = 100;
opts.main_tol = 1e-10;
opts.cg_maxiters = 20;
opts.cg_tol = 1e-6;
tic;
[x_sqs] = sparse_quadratic_solver(results.model, opts);
toc
results.sqs_lasso = struct('x', x_sqs);

obj_opt = get_objective(results.model, x0);
obj_matlab = get_objective(results.model, x_ML_lasso);
obj_sqs = get_objective(results.model, x_sqs);
fprintf('\n        opt         sqs      Matlab   True Inactive\n')
fprintf(' %1.4e  %1.4e  %1.4e           %5i\n', obj_opt, obj_sqs, obj_matlab, num_coords);


  
%  [get_objective(results.model, x_ML_lasso), get_objective(results.model, x)]
  
  
  
%{
  n = 300;
  m = 300;
  A = rand(m, n);
  b = rand(m, 1);
  lambda = 0.0000005;
  sqs_lambda = m*lambda;
  results.model = struct('type', 'lasso', ...
                         'A', A, 'b', b, 'lambda', sqs_lambda);

  tic;
  [x_ML_lasso] = lasso(A, b, 'Lambda', lambda);
  toc
  results.ML_lasso = struct('x', x_ML_lasso);

  opts = [];
  opts.display = 1;
  tic;
  [x] = sparse_quadratic_solver(results.model, opts);
  toc
  results.sqs_lasso = struct('x', x);
  
  get_objective(results.model, x) - get_objective(results.model, x_ML_lasso)
  
  [get_objective(results.model, x_ML_lasso), get_objective(results.model, x)]
%}
  
end

function [obj] = get_objective(model, x)
  if strcmp(model.type, 'box_qp')
    obj = 0.5*(x'*(model.H*x)) + model.f'*x;
  else
    obj = 0.5*norm(model.A*x - model.b)^2;% + model.lambda*norm(x, 1);
  end
end