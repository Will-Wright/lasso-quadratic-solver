function [results] = runtest_lasso()

% NOTE: solver failing to recover Ax = b, use following to see this
% norm(results.model.A*results.sqs_lasso.x - results.model.b)
%
% just debugged prox operator (project_soft_threshold)
%

rng(1);


n = 2000;
num_coords = 10;
m = 100;
noise = 0.0;
A = normc(rand(m, n));
%  x_idx = randsample(n, num_coords, false);
%  x_sparse = zeros(n, 1);
%  x_sparse(x_idx) = ones(num_coords, 1) + rand(num_coords, 1);
%  x0 = x_sparse + noise*rand(n, 1);
%  b = A*x0;
b = normc(rand(m, 1));
lambda = 0.5;
sqs_lambda = lambda;
%  results.solution = struct('x_sparse', x_sparse, 'x0', x0, 'x_idx', x_idx);
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
opts.apg_maxiters = 500;
opts.apg_tol = 1e-4;
opts.main_maxiters = 20;
opts.main_tol = 1e-12;
opts.cg_maxiters = 20;
opts.cg_tol = 1e-6;
tic;
[x_sqs] = sparse_quadratic_solver(results.model, opts);
toc
results.sqs_lasso = struct('x', x_sqs);

%obj_opt = get_objective(results.model, x0);
obj_matlab = get_objective(results.model, x_ML_lasso);
obj_sqs = get_objective(results.model, x_sqs);
fprintf('\n        sqs      Matlab  \n')
fprintf(' %1.4e  %1.4e  \n', obj_sqs, obj_matlab);




  
end

function [obj] = get_objective(model, x)
  if strcmp(model.type, 'box_qp')
    obj = 0.5*(x'*(model.H*x)) + model.f'*x;
  else
    obj = 0.5*norm(model.A*x - model.b)^2;% + model.lambda*norm(x, 1);
  end
end