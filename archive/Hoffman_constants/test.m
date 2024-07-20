function [x,t] = test(A,options)
    n = size(A,2) ;
    m = size(A,1) ;
    f = [zeros(n,1);1] ;
    AA = [A, -ones(m,1) ; -A, -ones(m,1)] ;
    bb = [zeros(2*m,1)] ;
    Aeq = [ones(1,n),0] ;
    beq = 1 ;
    LB = zeros(n+1,1) ;
    %options = optimoptions('linprog','algorithm','dual-simplex','display','off','ConstraintTolerance',1e-9,'OptimalityTolerance',1e-10);
    xx = linprog(f,AA,bb,Aeq,beq,LB,[],options)  ;
    % round small components
    xx(abs(xx)<1e-12) = 0 ;
    x = xx(1:n) ;
    t = xx(n+1) ;