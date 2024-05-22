% test hoffman constants on randomly sampled tropical rational functions

options = optimoptions('linprog','algorithm','dual-simplex','display','off','ConstraintTolerance',1e-9,'OptimalityTolerance',1e-10);

% number of experiments
num_exps = 10;
% number of monomials in the nominator f
num_mono_up = 15;
% number of monomials in the denominator g
num_mono_down = 5;

m = num_mono_up*num_mono_down;
hoff_time_list = zeros(num_exps,m);
lin_time_list = zeros(num_exps,m);
num_loop_list = zeros(num_exps,m);

parfor i = 1:num_exps
    f = [randi(10,num_mono_up,3),randn(num_mono_up,1)];
    g = [randi(10,num_mono_down,3),randn(num_mono_down,1)];
    [H,R,hoff_time,count_loop,lin_time] = Min_Rad(f,g,options);
    hoff_time_list(i,:) = hoff_time;
    lin_time_list(i,:) = lin_time;
    num_loop_list(i,:) = count_loop;
    %fprintf("Average time to compute Hoffman constant" + ...
    %    " per linear region: %f\n", mean(hoff_time));
    %fprintf("Average time to compute linear programming " + ...
    %    "per loop: %f\n", mean(lin_time));
    %fprintf("Average number of loops to compute Hoffman constant" + ...
        " per linear region: %d\n", mean(count_loop));
end


