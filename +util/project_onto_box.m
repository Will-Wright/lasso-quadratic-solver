function [x_proj, bool_active, idx_active, idx_inactive] = project_onto_box(model, x)
  x_proj = min(max(x, model.lb), model.ub);
  if nargout == 2
    bool_active = (x_proj == model.lb) + (x_proj == model.ub);
  elseif nargout > 2
    bool_active = (x_proj == model.lb) + (x_proj == model.ub);
    idx_active = find(bool_active == 1);
    idx_inactive = find(bool_active == 0);
  end
end