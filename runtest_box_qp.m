function [results] = runtest_box_qp(varargin)

ip = inputParser;

ip.addParameter('n', 500);
ip.addParameter('sparsity_ratio', 0.1);
ip.addParameter('seed', 0);
ip.addParameter('disp', 1);

ip.parse(varargin{:});

n = ip.Results.n;
sparsity = ip.Results.sparsity_ratio;
seed = ip.Results.seed;
disp = ip.Results.disp;

% Generates random problem with exact sparsity ratio in optimal x0
rng(seed);
U = orth(rand(n,n));
d = rand(n, 1);
H = U*diag(d)*U'; H = (0.5)*(H + H');
n_half = round(n/2);
x0 = zeros(n,1);
x0(1:n_half, 1) = rand(n_half, 1);
x0(n_half+1:n, 1) = -rand(n - (n_half+1) + 1, 1);

num_nonzeros = round(n*sparsity);
sparsity_scale_idx = round(num_nonzeros/2);

x_temp = sort(x0(1:n_half, 1));
sp_scale_ub = x_temp(sparsity_scale_idx, 1);

x_temp = sort(-x0(n_half+1:n, 1));
sp_scale_lb = -x_temp(sparsity_scale_idx, 1);

x0 = x0(randperm(n));

f = -H*x0;

lb = sp_scale_lb*ones(n, 1);
ub = sp_scale_ub*ones(n, 1);
results.model = struct('type', 'box_qp', ...
                      'H', H, 'f', f, 'lb', lb, 'ub', ub);
SQS_opts = struct('disp', disp, ...
   'apg_tol', 1e-2, 'apg_maxiters', 500, 'diff_tol', 0, ...
   'cg_tol', 1e-6, 'cg_maxiters', 20, 'main_tol', 1e-12);

tic;
[x_quadprog] = quadprog(H, f, [], [], [], [], lb, ub);
time_qp = toc;
results.quadprog = struct('x', x_quadprog);

tic;
[x] = sparse_quadratic_solver(results.model, SQS_opts);
time_sqs = toc;
results.box_qp = struct('x', x);

fprintf('quadprog runtime %2.5f sec\n', time_qp)
fprintf('SQS runtime      %2.5f sec\n', time_sqs)
obj_sqs = get_objective(results.model, x);
obj_qp = get_objective(results.model, x_quadprog);
if obj_sqs < obj_qp
  fprintf('SQS more accurate: %1.3e\n', obj_qp - obj_sqs)
else
  fprintf('quadprog more accurate: %1.3e\n', obj_sqs - obj_qp)
end

if time_sqs < time_qp
  fprintf('SQS faster: %1.2f%% less runtime\n', abs(time_sqs - time_qp) / time_qp)
else
  fprintf('quadprog faster: %1.2f%% less runtime\n', abs(time_sqs - time_qp) / time_sqs)
end

end


function [obj] = get_objective(model, x)
  obj = 0.5*(x'*(model.H*x)) + model.f'*x;
end