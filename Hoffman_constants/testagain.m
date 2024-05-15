function [x] = testagain(A,c,options)
    n = size(A,2) ;
    m = size(A,1) ;
    Aeq = [A ; ones(1,n)] ;
    beq = [zeros(m,1) ; 1] ;
    LB = zeros(n,1) ;
%    options = optimoptions('linprog','algorithm','dual-simplex','display','off','ConstraintTolerance',1e-9,'OptimalityTolerance',1e-10);
%    options = optimoptions('linprog','algorithm','dual-simplex','display','off');
    xx = linprog(c,[],[],Aeq,beq,LB,[],options)  ;
    % round small components
    xx(abs(xx)<1e-12) = 0 ;
    x = xx(1:n) ;
    