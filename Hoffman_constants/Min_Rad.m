function [H,R,hoff_time,count_loop,lin_time] = Min_Rad(f,g,options)
% this function returns the minimal effective radius for a 
% tropical rational function f-g where f,g are tropical polynomials

% tropical polynomials are represented by matrix [A, b] where each row
% represents the monomial ax+b

n = size(f,2)-1;
mf = size(f,1); mg = size(g,1);

e1 = ones(mf,1); e2 = ones(mg,1);

m = mf*mg;
H = zeros(1, m); R = zeros(1,m);
hoff_time = zeros(1,m);
count_loop = zeros(1,m);
lin_time = zeros(1,m);


for k = 1:m
    i = ceil(k/mg);
    j = mod(k-1,mg)+1;

    A1 = e1*f(i,1:n) - f(:,1:n);
    b1 = f(:,n+1) - e1*f(i,n+1);
    A2 = e2*g(j,1:n) - g(:,1:n);
    b2 = g(:,n+1) - e2*g(j,n+1);
    A = [A1;A2];
    s_time = tic;
    [HA,count,linprog_time,~,~,~] = Hoffman(A,options);
    e_time = toc(s_time);
    count_loop(k) = count;
    hoff_time(k) = e_time;
    lin_time(k) = mean(linprog_time);
    H(k) = HA;
    Rb = max(-[b1;b2]);
    if Rb < 0
       Rb = 0;
    end
    R(k) = HA*Rb;
end

