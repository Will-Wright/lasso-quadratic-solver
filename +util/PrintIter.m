function [] = PrintIter(iter, model, obj, res_norm, active_set_diff, num_inactive, num_active, tau)
  if nargin == 7
    fprintf(' %4i  |  % 1.8e   %1.5e    %4i     %4i     %4i  \n', iter, obj, res_norm, ...
            active_set_diff, num_inactive, num_active);
  elseif nargin == 8
    fprintf(' %4i  |  % 1.8e   %1.5e    %4i     %4i     %4i       %1.3e\n', iter, obj, res_norm, ...
            active_set_diff, num_inactive, num_active, tau);
  end
end