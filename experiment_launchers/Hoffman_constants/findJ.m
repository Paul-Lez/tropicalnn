function [J] = findJ(FF,II)
% Find largest J without certificate
% Detailed explanation goes here
m = max(size(FF,2),size(II,2)) ;
f = -ones(m,1) ;
intcon = 1:m ;
if (size(FF,1) == 0)
    A = II ; b = II*ones(m,1) - 1 ;
elseif (size(II) == 0)
    A = FF - 1 ; b = -ones(size(FF,1),1);
else
    A = [FF - 1;II] ;
    b = [-ones(size(FF,1),1); II*ones(m,1) - 1] ;
end 
options = optimoptions('intlinprog','display','off') ;
J = intlinprog(f,intcon,A,b,[],[],zeros(m,1),ones(m,1),options) ;
J = J' ;

