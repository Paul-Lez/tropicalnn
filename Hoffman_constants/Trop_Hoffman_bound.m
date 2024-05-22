function [H_lower,H_upper,Hoff_lower_time,Hoff_upper_time] = Trop_Hoffman_bound(f,g,options)

% this function returns the lower bound of Hoffman constant for a 
% tropical rational function f-g where f,g are tropical polynomials

% tropical polynomials are represented by matrix [A, b] where each row
% represents the monomial ax+b

n = size(f,2)-1;
mf = size(f,1); mg = size(g,1);

e1 = ones(mf,1); e2 = ones(mg,1);

m = mf*mg;
H_lower = zeros(1,m);
H_upper = zeros(1,m);
Hoff_lower_time = zeros(1,m);
Hoff_upper_time = zeros(1,m);

for k = 1:m
    i = ceil(k/mg);
    j = mod(k-1,mg)+1;

    A1 = e1*f(i,1:n) - f(:,1:n);
    b1 = f(:,n+1) - e1*f(i,n+1);
    A2 = e2*g(j,1:n) - g(:,1:n);
    b2 = g(:,n+1) - e2*g(j,n+1);
    A = [A1;A2];
    s_time = tic;
    Hl = Hoffman_lower(A,options);
    e_time = toc(s_time);
    Hoff_lower_time(k) = e_time;
    s_time = tic;
    Hu = Hoffman_upper(A);
    e_time = toc(s_time);
    Hoff_upper_time(k) = e_time;
    H_lower(k) = Hl;
    H_upper(k) = Hu;
end
