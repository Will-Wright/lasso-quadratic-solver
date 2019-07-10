function [results] = runtest_box_qp(varargin)

import util.*;

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
LQS_opts = struct('disp', disp);
if disp == 2
   options = optimoptions('quadprog','Display','iter');
elseif disp == 1
   options = optimoptions('quadprog','Display','final');
else
   options = optimoptions('quadprog','Display','none');
end


tic;
[x_quadprog] = quadprog(H, f, [], [], [], [], lb, ub, [], options);
time_qp = toc;
Hess_fun = @(x) H*x;
q_vec = f;
eigs_opts.tol = 1e-4;
eigs_opts.issym = true;
sigma = 'LM';
k = 1;
eig_max = eigs(Hess_fun, n, k, sigma, eigs_opts);
eig_max = eig_max * 0.9;
gamma = 1/eig_max;   % forward-backward envelope parameter, in (0, 1/eig_max(H))
[~, res] = prox_grad_step(results.model, Hess_fun, q_vec, x_quadprog, gamma);
results.quadprog = struct('x', x_quadprog, 'res', res, 'runtime', time_qp);


tic;
[x, res, data] = LassoQuadraticSolver(results.model, LQS_opts);
time_LQS = toc;
results.LQS = struct('x', x, 'res', res, 'runtime', time_LQS, 'data', data);


if disp >= 1
   fprintf('quadprog runtime %2.5f sec\n', time_qp)
   fprintf('LQS runtime      %2.5f sec\n', time_LQS)
obj_lqs = get_objective_local(results.model, x);
obj_qp = get_objective_local(results.model, x_quadprog);
end


if disp >= 1
   if obj_lqs < obj_qp
     fprintf('LQS more accurate: %1.3e\n', obj_qp - obj_lqs)
   else
     fprintf('quadprog more accurate: %1.3e\n', obj_lqs - obj_qp)
   end
   if time_LQS < time_qp
     fprintf('LQS faster: %1.2f%% less runtime\n', abs(time_LQS - time_qp) / time_qp)
   else
     fprintf('quadprog faster: %1.2f%% less runtime\n', abs(time_LQS - time_qp) / time_LQS)
   end
end
   

end



function [obj] = get_objective_local(model, x)
  obj = 0.5*(x'*(model.H*x)) + model.f'*x;
end