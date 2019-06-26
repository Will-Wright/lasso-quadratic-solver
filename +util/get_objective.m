function [obj] = get_objective(model, Hess_fun, q_vec, x)
  if strcmp(model.type, 'box_qp')
    obj = 0.5*(x'*(Hess_fun(x))) + q_vec'*x;
  elseif strcmp(model.type, 'lasso')
    obj = 0.5*norm(model.A*x - model.b)^2 + model.lambda*norm(x, 1);
  end
end