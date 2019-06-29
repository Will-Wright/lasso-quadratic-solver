function [x_prox_grad, res, bool_active, idx_active, idx_inactive] = prox_grad_step(model, Hess_fun, q_vec, x, gamma)
  
import util.*;

if nargout < 3
   if strcmp(model.type, 'lasso')
      x_prox_grad = project_soft_threshold(model, x - gamma*(Hess_fun(x) + q_vec), gamma);
   elseif strcmp(model.type, 'box_qp')
      x_prox_grad = project_onto_box(model, x - gamma*(Hess_fun(x) + q_vec));
   end
elseif nargout == 3
   if strcmp(model.type, 'lasso')
      [x_prox_grad, bool_active] = project_soft_threshold(model, x - gamma*(Hess_fun(x) + q_vec), gamma);
   elseif strcmp(model.type, 'box_qp')      
      [x_prox_grad, bool_active] = project_onto_box(model, x - gamma*(Hess_fun(x) + q_vec));
   end
else
   if strcmp(model.type, 'lasso')
      [x_prox_grad, bool_active, idx_active, idx_inactive] = project_soft_threshold(model, x - gamma*(Hess_fun(x) + q_vec), gamma);
   elseif strcmp(model.type, 'box_qp')      
      [x_prox_grad, bool_active, idx_active, idx_inactive] = project_onto_box(model, x - gamma*(Hess_fun(x) + q_vec));
   end
end
res = x - x_prox_grad;

end