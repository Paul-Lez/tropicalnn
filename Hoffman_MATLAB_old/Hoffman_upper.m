function H_upper = Hoffman_upper(A)

m = size(A,1);
H_upper = 0;
rA = rank(A);

% number of iterations
N = 10000;

for i = 1:N
    num_rows = randi(m);
    p = randperm(m);
    AU = A(p(1:num_rows),:);
    rho = min(svd(AU));
    if rho>1e-10
        H_upper = max(H_upper,1/rho);
    end
end
