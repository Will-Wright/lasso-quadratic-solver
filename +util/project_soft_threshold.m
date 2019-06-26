function [x_proj, bool_active, idx_active, idx_inactive] = project_soft_threshold(model, x, gamma)

  x_proj = zeros(size(x));
  bool_active = (abs(x) <= model.lambda*gamma);
  bool_inactive = ~bool_active;
  x_proj(bool_inactive) = sign(x(bool_inactive)).*(abs(x(bool_inactive)) - model.lambda*gamma);
  
  if nargout > 2
    idx_active = find(bool_active == 1);
    idx_inactive = find(bool_active == 0);
  end

end