function trop_test(up,down,dim)

% test hoffman constants on randomly sampled tropical rational functions

% set random seed
rng(2024);

% put optimoptions outside any loop as itself costs time!
options = optimoptions('linprog','algorithm','dual-simplex','display','off','ConstraintTolerance',1e-9,'OptimalityTolerance',1e-10);

% number of experiments
num_exps = 10;
% number of monomials in the nominator f
num_mono_up = up;
% number of monomials in the denominator g
num_mono_down = down;
% input dimension
dimension = dim;

m = num_mono_up*num_mono_down;
H_list = zeros(num_exps,m);
H_lower_list = zeros(num_exps,m);
H_upper_list = zeros(num_exps,m);
Hoff_time_list = zeros(num_exps,m);
Hoff_lower_time_list = zeros(num_exps,m);
Hoff_upper_time_list = zeros(num_exps,m);
lin_time_list = zeros(num_exps,m);
num_loop_list = zeros(num_exps,m);
R_list = zeros(num_exps,m);

for i = 1:num_exps
    f = [10-10*rand(num_mono_up,dimension),10-10*rand(num_mono_up,1)];
    g = [10-10*rand(num_mono_down,dimension),10-10*rand(num_mono_down,1)];
    [H,R,Hoff_time,count_loop,lin_time] = Min_Rad(f,g,options);
    [H_lower,H_upper,Hoff_lower_time,Hoff_upper_time] = Trop_Hoffman_bound(f,g,options);
    H_list(i,:) = H;
    R_list(i,:) = R;
    H_lower_list(i,:) = H_lower;
    H_upper_list(i,:) = H_upper;
    Hoff_time_list(i,:) = Hoff_time;
    Hoff_lower_time_list(i,:) = Hoff_lower_time;
    Hoff_upper_time_list(i,:) = Hoff_upper_time;
    lin_time_list(i,:) = lin_time;
    num_loop_list(i,:) = count_loop;
    %fprintf("Average time to compute Hoffman constant" + ...
    %    " per linear region: %f\n", mean(hoff_time));
    %fprintf("Average time to compute linear programming " + ...
    %    "per loop: %f\n", mean(lin_time));
    %fprintf("Average number of loops to compute Hoffman constant" + ...
    %    " per linear region: %d\n", mean(count_loop));
end

filename = sprintf("results_up%d_down%d_dim%d.mat",up,down,dim);
save(filename,"H_list","H_lower_list","H_upper_list", ...
    "Hoff_upper_time_list","Hoff_lower_time_list","Hoff_time_list","lin_time_list", ...
    "num_loop_list","R_list")
