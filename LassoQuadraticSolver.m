function [x, res, data] = LassoQuadraticSolver(model, varargin)
% This algorithm solves the the lasso problem
%
%   min   (1/2)*||Ax - b||^2_2 + lambda*||x||_1
%
% and the box-constrainted quadratic program
%
%   min   (1/2)*x'*H*x + f'*x
%   st    lb <= x <= ub
%
% INPUT:
%   model = struct('type', type, 
%                  'H', H, 'f', f, 'lb', lb, 'ub', ub,  % for box-qp
%                  'A', A, 'b', b, 'lambda', lambda,  % for LASSO
%                  'verbosity', verbosity);
%   type        'box_qp' or 'lasso'
%   verbosity   0: no display
%               1: only final results
%               2: iterate results
%
% OUTPUT:
%   x           solution
%   
%
% NOTE: THIS SOLVER IS AN ALPHA VERSION / PROTOTYPE based on the qualifying exam paper
%       https://github.com/Will-Wright/lasso-quadratic-solver/blob/master/will_wright_qualifying_exam_proposal.pdf
%
% FUTURE MODIFICATIONS:
%   
%   + Do continuation scheme for all 3 methods
%     + l1 versions are the same
%     + Try mimicking for box_qp by "growing" box
%
%   + Replace FISTA and Lip const with Adaptive Restart (2012 O'Donoghue Candes)
%     + Give solver 2 modes: 'GD' and 'crossover'
%
%   + Add convergence tolerance for box_qp, l1_reg, l1_log
%
%   + Add basic safeguards
%     + E.g., user inputted all necessary objects

import util.*;

ip = inputParser;
ip.addParameter('apg_maxiters', 500); 
ip.addParameter('apg_tol', 1e-3);
% terminates accel_prox_grad if active_set_diff/opts.n < active_set_diff_tol
ip.addParameter('diff_tol', 0); % [0,1], or -1 for termination only on residual
ip.addParameter('main_maxiters', 100);
ip.addParameter('main_num_pg_steps', 1);
ip.addParameter('Newton_step_min', 1e-4);
ip.addParameter('cg_tol', 1e-6);
ip.addParameter('cg_maxiters', 20);
ip.addParameter('main_tol', 1e-9);
ip.addParameter('x0', []);
ip.addParameter('disp', 0)
ip.parse(varargin{:});

opts.apg_maxiters = ip.Results.apg_maxiters;
opts.apg_tol = ip.Results.apg_tol;
opts.diff_tol = ip.Results.diff_tol;
opts.main_loop_maxiters = ip.Results.main_maxiters;
opts.main_num_pg_steps = ip.Results.main_num_pg_steps;
opts.Newton_step_min = ip.Results.Newton_step_min;
opts.cg_tol = ip.Results.cg_tol;
opts.cg_maxiters = ip.Results.cg_maxiters;
opts.main_tol = ip.Results.main_tol;
opts.disp = ip.Results.disp;
x0 = ip.Results.x0;


if strcmp(model.type, 'box_qp')
   opts.n = size(model.f, 1);  
   Hess_fun = @(x) model.H*x;
   q_vec = model.f;
elseif strcmp(model.type, 'lasso')
   [opts.m, opts.n] = size(model.A);
   Hess_fun = @(x) model.A'*(model.A*x);
   q_vec = -model.A'*model.b;
else
   fprintf('Incorrect model type passed.  Program exited.\n')
   return
end

if isempty(x0) || sum(size(x0) == [opts.n, 1]) ~= 2
   x0 = ones(opts.n, 1);
end

% linesearch parameters.  Note: sigma in (0, 1/2)
LS_scale = 1/4;
tau_update = 0.9;


eigs_opts.tol = 1e-4;
eigs_opts.issym = true;
sigma = 'LM';
k = 1;
opts.eig_max = eigs(Hess_fun, opts.n, k, sigma, eigs_opts);
opts.eig_max = opts.eig_max * 0.9;
gamma = 1/opts.eig_max;   % forward-backward envelope parameter, in (0, 1/eig_max(H))

if opts.disp == 2
   print_banner(1);
elseif opts.disp == 1
   if strcmp(model.type, 'box_qp')
      fprintf('\nSolver called for quadratic program\n')
   elseif strcmp(model.type, 'lasso')
      fprintf('\nSolver called for lasso problem\n')
   end
end

tic;

[x, res_norm, bool_active_prev, num_iters_apg] = accel_prox_grad(model, Hess_fun, q_vec, x0, opts, gamma);

if (opts.disp == 2) && opts.main_loop_maxiters > 0
   print_banner(2);
end

iter = 1;
tau = 1; % steplength for Newton step in main loop below
res = [];

% TODO: Fix this Mat-Mat mult.  Allow for passing functions At and A
if strcmp(model.type, 'lasso')
   AtA = model.A'*model.A;
end

while (iter <= opts.main_loop_maxiters) && (res_norm >= opts.main_tol) ...
      && (tau >= opts.Newton_step_min)
   %  Computes prox-grad step to update x and active/inactive indices
   for i = 1:opts.main_num_pg_steps
      [x_pg, res, bool_active, idx_act, idx_inact] = prox_grad_step(model, Hess_fun, q_vec, x, gamma); 
   end
   
   d_act = x_pg(idx_act, 1) - x(idx_act, 1);
   if strcmp(model.type, 'lasso')
      rhs_inact = (1/gamma)*(x_pg(idx_inact, 1) - x(idx_inact, 1)) - AtA(idx_inact, idx_act)*d_act;
      H_mat = AtA(idx_inact, idx_inact);
   elseif strcmp(model.type, 'box_qp')      
      rhs_inact = (1/gamma)*(x_pg(idx_inact, 1) - x(idx_inact, 1)) - model.H(idx_inact, idx_act)*d_act;            
      H_mat = model.H(idx_inact, idx_inact);
   end

   [d_inact, flag_pcg, relres_pcg, iter_pcg, resvec_pcg]  = pcg(H_mat, rhs_inact, opts.cg_tol, opts.cg_maxiters);
   %    [d_inact, flag, relres, iter, resvec]  = cgs(model.H(idx_inact, idx_inact), rhs_inact, opts.cg_tol, opts.cg_maxiters);
   %    [d_inact, flag, relres, iter, resvec]  = bicg(model.H(idx_inact, idx_inact), rhs_inact, opts.cg_tol, opts.cg_maxiters);
   %    [d_inact, flag, relres, iter, resvec]  = bicgstab(model.H(idx_inact, idx_inact), rhs_inact, opts.cg_tol, opts.cg_maxiters);

   d = zeros(opts.n, 1);    
   d(idx_act) = d_act;
   d(idx_inact) = d_inact;
   tau = 1;
   [FBE_f, FBE_grad] = get_FBE(model, Hess_fun, q_vec, x, gamma);
   FBE_f_update = get_FBE(model, Hess_fun, q_vec, x + tau*d, gamma);
   while (FBE_f_update > FBE_f + LS_scale*tau*FBE_grad'*d)
     tau = tau_update*tau;
     FBE_f_update = get_FBE(model, Hess_fun, q_vec, x + tau*d, gamma);
   end
   x = x + tau*d;
   
   obj = get_objective(model, Hess_fun, q_vec, x);
   active_set_diff = sum(abs(bool_active - bool_active_prev));
   num_active = sum(bool_active);
   num_inactive = opts.n - num_active;
   res_norm = norm(res);
   if opts.disp == 2
      print_iter(iter, model, obj, res_norm, active_set_diff, num_inactive, num_active, tau);
   end

   bool_active_prev = bool_active;
   iter = iter + 1;

end


if strcmp(model.type, 'lasso')
   x_idx = (x ~= 0);
   x_temp = model.A(:,x_idx)\model.b;
   x = zeros(opts.n,1);
   x(x_idx) = x_temp;
end

if (res_norm < opts.main_tol)
   if opts.disp >= 1
      fprintf('EXIT - OPTIMAL SOLUTION FOUND\n')
   end
   data.term_cond = 'opt';
elseif (tau < opts.Newton_step_min)
   if opts.disp >= 1
      fprintf('EXIT - MINIMUM STEPLENGTH MET\n')
   end
   data.term_cond = 'min_step';
elseif (iter > opts.main_loop_maxiters)
   if opts.disp >= 1
      fprintf('EXIT - MAXIMUM ITERATIONS MET\n')
   end
   data.term_cond = 'max_iters';
end


data.num_iters = iter;
data.num_iters_apg = num_iters_apg;

end  %  end function sparse_quadratic_solver

