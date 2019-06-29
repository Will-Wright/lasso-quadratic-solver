function [] = print_banner(phase_num)
  if phase_num == 1
    fprintf('\n');
    fprintf('                        Sparse Quadratic Solver\n');
    fprintf('-------------------------------------------------------------------------------------\n');
    fprintf('          Warm-start Stage: Accelerated Proximal Gradient\n');
    fprintf('                                              Active-Set Info\n');
    fprintf(' Iter  |     Objective       Residual    Changes  Inactive   Active  \n');
    fprintf('-------|-----------------------------------------------------------------------------\n');    
  elseif phase_num == 2
    fprintf('-------------------------------------------------------------------------------------\n');
    fprintf('          Quasi-Newton Stage: Active-Set Conjugate Gradient\n');
    fprintf('                                              Active-Set Info\n');
    fprintf(' Iter  |     Objective       Residual    Changes  Inactive   Active    Steplength \n');
    fprintf('-------|-----------------------------------------------------------------------------\n');    
  elseif phase_num == 3
    fprintf(' Iter  |     Objective       Residual    Changes  Inactive   Active  \n');
    fprintf('-------|-----------------------------------------------------------------------------\n');
  end
end