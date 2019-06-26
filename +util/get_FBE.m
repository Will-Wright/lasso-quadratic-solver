function [FBE_f, FBE_grad] = get_FBE(model, Hess_fun, q_vec, x, gamma)

Hx = Hess_fun(x);
f_grad = Hx + q_vec;
if strcmp(model.type, 'box_qp')
 f_obj = 0.5*(x'*(Hx)) + q_vec'*x;
elseif strcmp(model.type, 'lasso')
 f_obj = 0.5*norm(model.A*x - model.b)^2 + model.lambda*norm(x, 1);
end

P = get_P(model, f_grad, x, gamma);
G = get_G(P, x, gamma);
FBE_f = f_obj + get_g(model, P) - gamma*f_grad'*G + (gamma/2)*norm(G)^2;
if nargout == 2
  FBE_grad = get_F_grad(Hess_fun, G, gamma);
end
    
end


function [P] = get_P(model, f_grad, x, gamma)
  if strcmp(model.type, 'box_qp')
    P = util.project_onto_box(model, x - gamma*f_grad);
  elseif strcmp(model.type, 'lasso')
    P = util.project_soft_threshold(model, x - gamma*f_grad, gamma);
  end
end

function [G] = get_G(P, x, gamma)
  G = (1/gamma)*(x - P);
end

function [F_grad] = get_F_grad(Hess_fun, G, gamma)
  F_grad = G - gamma*(Hess_fun(G));
end

function [g_obj] = get_g(model, x)
  if strcmp(model.type, 'box_qp')
    if all(model.lb <= x) && all(x <= model.ub)
      g_obj = 0;
    else
      g_obj = Inf;
    end
  elseif strcmp(model.type, 'lasso')
    g_obj = model.lambda*norm(x,1);
  end
end