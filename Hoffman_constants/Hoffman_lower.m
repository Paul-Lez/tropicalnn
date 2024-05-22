function H_lower = Hoffman_lower(A,options)
% compute lower bound of Hoffman constants

m = size(A,1);
H_lower = 0;
%rA = rank(A);

% number of iterations
N = 100;

for i = 1:N
    num_rows = randi(m);
    p = randperm(m);
    rows = p(1:num_rows);
    AA = A(rows,:);
    [y,t] = test(AA,options);
    if (t>0)
        H_lower = max(H_lower,1/t);
    end
end
