function [x, res_norm, bool_active, iter] = accel_prox_grad(model, Hess_fun, q_vec, x0, opts, gamma)
% Solves initial phase of quadratic program using accelerated proximal
% (projected) gradient descent

import util.*;

maxiters = opts.apg_maxiters;
apg_tol = opts.apg_tol;
active_set_diff_tol = opts.diff_tol;
fista_param = 1;
fista_param_prev = 1;
fista_scale = 0;
x = x0;
x_prev = x0;
y = x0;
bool_active_prev = zeros(opts.n, 1);
res_norm = 1;
active_set_diff = opts.n;

iter = 1;

while (iter <= maxiters) && (res_norm > apg_tol) ...
   && ( (active_set_diff/opts.n > active_set_diff_tol) || iter < 10 )

   [x, res, bool_active] = prox_grad_step(model, Hess_fun, q_vec, y, gamma);
   fista_param = max(roots([1, fista_param_prev^2, -fista_param_prev^2]));
   fista_scale = fista_param_prev*(1 - fista_param_prev)/(fista_param_prev^2 + fista_param);
   y = x + fista_scale*(x - x_prev);
   
   obj = get_objective(model, Hess_fun, q_vec, x);
   active_set_diff = sum(abs(bool_active_prev - bool_active));
   res_norm = norm(res);
   num_active = sum(bool_active);
   num_inactive = opts.n - num_active;
   if opts.disp == 2
      if mod(iter, 100) == 0
         print_banner(3);
      end
      print_iter(iter, model, obj, res_norm, active_set_diff, num_inactive, num_active);
   end

   bool_active_prev = bool_active;
   x_prev = x;
   iter = iter + 1;

end

end
