function [data] = RunDemo()

import util.*;

num_tests_max = 10;
n_range = 100:100:2000;
sparsity_ratio_range_qp = 0.1:0.1:0.9;
sparsity_ratio_range_lasso = 0.05:0.05:0.3;

plot_main_fig = true;
plot_small_fig = false;


folder_name = 'cache/';
file_name = 'cache/data.mat';
if exist(folder_name) ~= 7
   mkdir(folder_name)
   data = struct;
   data.qp.n_range = n_range;
   data.qp.sparsity_ratio_range = sparsity_ratio_range_qp;
   data.qp.num_tests_max = num_tests_max;
   data.qp.lqs_data = cell(length(n_range), length(sparsity_ratio_range_qp), num_tests_max);
   
   data.lasso.n_range = n_range;
   data.lasso.num_tests_max = num_tests_max;
   data.lasso.sparsity_ratio_range = sparsity_ratio_range_lasso;
   data.lasso.lqs_data = cell(length(n_range), length(sparsity_ratio_range_qp), num_tests_max);
   save(file_name, 'data')
end


load(file_name, 'data')



fprintf('\nRunning quadratic program experiments\n');
for n_idx = 1:length(n_range)
   for sparsity_idx = 1:length(sparsity_ratio_range_qp)
      for test_num = 1:num_tests_max
         if isempty(data.qp.lqs_data{n_idx, sparsity_idx, test_num})
            n = n_range(n_idx);
            sparsity = sparsity_ratio_range_qp(sparsity_idx);
            fprintf('Running test n = %i, sparsity = %1.2f, test num = %i\n', n, sparsity, test_num);
            [results] = runtest_box_qp('disp', 0, 'n', n, 'sparsity', sparsity, 'seed', test_num);
            data.qp.qp_res(n_idx, sparsity_idx, test_num) = norm(results.quadprog.res);
            data.qp.lqs_res(n_idx, sparsity_idx, test_num) =  norm(results.LQS.res);
            data.qp.qp_time(n_idx, sparsity_idx, test_num) = results.quadprog.runtime;
            data.qp.lqs_time(n_idx, sparsity_idx, test_num) =  results.LQS.runtime;
            data.qp.lqs_data{n_idx, sparsity_idx, test_num} = results.LQS.data;
            save(file_name, 'data')
         end
      end
   end
end



fprintf('\nRunning lasso experiments\n');
for n_idx = 1:length(n_range)
   for sparsity_idx = 1:length(sparsity_ratio_range_lasso)
      for test_num = 1:num_tests_max
         if isempty(data.lasso.lqs_data{n_idx, sparsity_idx, test_num})
            n = n_range(n_idx);
            sparsity = sparsity_ratio_range_lasso(sparsity_idx);
            num_coords = round(sparsity*n);
            num_constraints = round(n/2);
            fprintf('Running test n = %i, num coords = %i, test num = %i\n', n, num_coords, test_num);
            [results] = runtest_lasso('disp', 0, 'n', n, 'num_coords', num_coords, ...
               'num_constraints', num_constraints, 'seed', test_num);
            
            
            data.lasso.ML_lasso_obj(n_idx, sparsity_idx, test_num) = results.obj_matlab;
            data.lasso.ML_lasso_time(n_idx, sparsity_idx, test_num) = results.ML_lasso.runtime;
            
            data.lasso.lqs_obj(n_idx, sparsity_idx, test_num) =  results.obj_lqs;
            data.lasso.lqs_time(n_idx, sparsity_idx, test_num) =  results.LQS.runtime;
            data.lasso.lqs_data{n_idx, sparsity_idx, test_num} = results.LQS.data;
            
            
            save(file_name, 'data')
         end
      end
   end
end


data.qp.qp_time_mean = mean(data.qp.qp_time,3);
data.qp.lqs_time_mean = mean(data.qp.lqs_time,3);
data.qp.qp_res_mean = mean(data.qp.qp_res,3);
data.qp.lqs_res_mean = mean(data.qp.lqs_res,3);

data.lasso.ML_lasso_time_mean = mean(data.lasso.ML_lasso_time,3);
data.lasso.lqs_time_mean = mean(data.lasso.lqs_time,3);
data.lasso.ML_lasso_obj_mean = mean(data.lasso.ML_lasso_obj,3);
data.lasso.lqs_obj_mean = mean(data.lasso.lqs_obj,3);

save(file_name, 'data')


% Plots performance results

if plot_main_fig

figure;
subplot(2,3,1)
sp=2; 
plot(data.qp.n_range, data.qp.lqs_time_mean(:,sp), 'o'); 
hold on; 
plot(data.qp.n_range, data.qp.qp_time_mean(:,sp), '*');
title('QP: Runtime vs Size')
xlabel('Dimension')
ylabel('Time (seconds)')
legend('LQS', 'quadprog-MATLAB')


subplot(2,3,2)
sp=2; 
plot(data.qp.n_range, data.qp.lqs_res_mean(:,sp), 'o'); 
hold on; 
plot(data.qp.n_range, data.qp.qp_res_mean(:,sp), '*');
title('QP: Accuracy vs Size')
xlabel('Dimension')
ylabel('Residual')
legend('LQS', 'quadprog-MATLAB')



subplot(2,3,3)
n_idx = 20;
plot(data.qp.sparsity_ratio_range, data.qp.lqs_time_mean(n_idx,:), 'o'); 
hold on; 
plot(data.qp.sparsity_ratio_range, data.qp.qp_time_mean(n_idx,:), '*');
title('QP: Runtime vs Sparsity')
xlabel('Sparsity Ratio')
ylabel('Time (seconds)')
legend('LQS', 'quadprog-MATLAB')


subplot(2,3,4)
sp=2; 
plot(data.lasso.n_range, data.lasso.lqs_time_mean(:,sp), 'o'); 
hold on; 
plot(data.lasso.n_range, data.lasso.ML_lasso_time_mean(:,sp), '*');
title('LASSO: Runtime vs Size')
xlabel('Dimension')
ylabel('Time (seconds)')
legend('LQS', 'lasso-MATLAB')


subplot(2,3,5)
sp=2; 
plot(data.lasso.n_range, data.lasso.lqs_obj_mean(:,sp), 'o'); 
hold on; 
plot(data.lasso.n_range, data.lasso.ML_lasso_obj_mean(:,sp), '*');
title('LASSO: Accuracy vs Size')
xlabel('Dimension')
ylabel('Objective Value')
legend('LQS', 'lasso-MATLAB')


subplot(2,3,6)
n_idx = 20;
plot(data.lasso.sparsity_ratio_range, data.lasso.lqs_time_mean(n_idx,:), 'o'); 
hold on; 
plot(data.lasso.sparsity_ratio_range, data.lasso.ML_lasso_time_mean(n_idx,:), '*');
title('LASSO: Runtime vs Sparsity')
xlabel('Sparsity Ratio')
ylabel('Time (seconds)')
legend('LQS', 'lasso-MATLAB')

end





if plot_small_fig

figure;
%{
subplot(1,2,1)
sp=2; 
plot(data.qp.n_range, data.qp.lqs_time_mean(:,sp), 'o'); 
hold on; 
plot(data.qp.n_range, data.qp.qp_time_mean(:,sp), '*');
title('Quadratic Program')
xlabel('Dimension')
ylabel('Seconds')
legend('LQS', 'quadprog-MATLAB')
%}

%subplot(1,2,2)
sp=2; 
plot(data.lasso.n_range, data.lasso.lqs_time_mean(:,sp), 'o'); 
hold on; 
plot(data.lasso.n_range, data.lasso.ML_lasso_time_mean(:,sp), '*');
title('LASSO')
xlabel('Dimension')
ylabel('Seconds')
legend('LQS', 'lasso-MATLAB')

end


end