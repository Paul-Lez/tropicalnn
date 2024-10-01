% load data

%load results_up10_down5_dim3.mat
%load results_up15_down9_dim6.mat
load results_up15_down5_dim7.mat


scaling = 1e4;

% table of Hoffman constants, lower and upper bounds
true_value = max(H_list,[],2)/scaling;
lower_bounds = max(H_lower_list,[],2)/scaling;
upper_bounds = max(H_upper_list,[],2)/scaling;

% bound for minimal effective radius
R = max(R_list,[],2)/scaling;

% table of stats of time

lower_time = mean(Hoff_lower_time_list,2);
true_time = mean(Hoff_time_list,2);
upper_time = mean(Hoff_upper_time_list,2);
LP_time = mean(lin_time_list,2);

% table of loop counts
loops = max(num_loop_list,[],2);