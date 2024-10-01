using Combinatorics, JuMP, GLPK, LinearAlgebra, Random, Distributions, Plots

function test(A)
    n = size(A, 2)
    m = size(A, 1)

    model = Model(GLPK.Optimizer)

    @variable(model, x[1:m] >= 0)
    @variable(model, t)

    @objective(model, Min, t)
    
    @constraint(model,[t;A'*x] in MOI.NormOneCone(1+n))

    @constraint(model, sum(x) == 1)

    optimize!(model)
    
    x_val = value.(x)
    t_val = value(t)

    x_val = map(v -> abs(v) < 1e-10 ? 0.0 : v, x_val)
    t_val = abs(t_val) < 1e-10 ? 0.0 : t_val
    
    return x_val, t_val
end

function exact_hoff(A)
    m = size(A, 1)
    H = 0.0

    for j in 1:m
        subsets = collect(combinations(1:m, j))
        for subset in subsets
            AA = A[subset, :]
            y, t = test(AA)
            if t > 0
                H = max(H, 1/t)
            end
        end
    end
    return H
end

function upper_hoff(A)
    m,n=size(A)
    HU=0.0

    for j in 1:m
        subsets = collect(combinations(1:m, j))
        for subset in subsets
            AJ=A[subset,:]
            if rank(AJ)==min(j,n)
                p_J=minimum(svdvals(AJ))
                if p_J>0
                    HU=max(HU,1/p_J)
                end
            end
        end
    end
    return HU
end

function lower_hoff(A,B=10)
    m,n=size(A)
    HL=0.0
    B=min(B,2^m)
    for i in 1:B
        K=rand(1:m)
        J=rand(1:m,K)
        AJ=A[J,:]
        x,t=test(AJ)
        if t>0
            HL=max(HL,1/t)
        end
    end
    return HL
end