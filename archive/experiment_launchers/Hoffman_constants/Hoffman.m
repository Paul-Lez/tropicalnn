function [H,count,linprog_time,FF,II,maxJJ] = Hoffman(A,options) 
% [H,FF,II,maxJJ] = Hoffman(A) 
% Compute Hoffman constant H for A.
% Compute also the collections FF, II that jointly provide
% certificates of surjectivity and non-surjectivity.
% Compute also the largest size of the set JJ throughout the algorithm
m = size(A,1) ; 
FF = zeros(0,m) ; II = zeros(0,m) ; JJ = ones(1,m) ; H = 0 ; maxJJ = 1 ;
count = 0;
linprog_time = [];

while (size(JJ,1) > 0) && (maxJJ < 15000000) 
    count = count + 1;
    maxJJ = max(maxJJ,size(JJ,1)) ;
    [~,i] = max(JJ*ones(m,1)) ;
    % choose set J in JJ of maximal size
    J = JJ(i,:) ;    
    % test J for A-surjectivity
    AA = A(J>0,:)' ;
    s_time = tic;
    [y,t] = test(AA,options) ;
    e_time = toc(s_time);
    linprog_time = [linprog_time,e_time];
    if (t > 0) 
        % J is A-surjective.  Add it to FF and use it to trim JJ
        FF = [FF;J] ;
        indx = find(J*JJ' == sum(JJ')) ;
        JJ(indx,:) = [] ;
        H = max(H,1/t) ;
    else
        % get a certificate of non-surjectivity for J. Add it to II and use
        % it to update JJ
        yy = zeros(m,1) ; yy(J>0) = y ;
        I = zeros(1,m) ; I(yy>0) = ones(1,nnz(yy)) ; II = [II;I] ; 
        JJ = adjustJJ(JJ,I,FF) ;
        % if possible, find other certificates of non-surjectivity for J
        % that are disjoint from I and repeat the above
        eraseindx = find(I>0) ; smallJ = J ;
        smallJ(eraseindx) = zeros(length(eraseindx),1) ;
        AA = A(smallJ>0,:)' ;
        if isempty(AA)
            t = 10 ;
        else
            [y,t] = test(AA,options) ;
        end 
        if (t > 0)
            % Could not find a disjoint certificate of non-surjectivity. 
            % Instead try to find one sufficiently different from I
            AA = A(J>0,:)' ;
            c = zeros(m,1) ;
            c(I>0) = 100*ones(sum(I>0),1) ;                
            [y] = testagain(AA,c(J>0),options) ; yy = zeros(m,1) ; yy(J>0) = y ;
            newI = zeros(1,m) ; newI(yy>0) = ones(1,nnz(yy)) ;
            if (norm(newI-I) > 0)
                II = [II;newI] ; 
                JJ = adjustJJ(JJ,newI,FF) ;
            end 
        end 
        while (t==0)
            yy = zeros(m,1) ; yy(smallJ>0) = y ;
            newI = zeros(1,m) ; newI(yy>0) = ones(1,nnz(yy)) ;
            II = [II;newI] ; JJ = adjustJJ(JJ,newI,FF) ;
            eraseindx = find(newI>0) ;
            smallJ(eraseindx) = zeros(length(eraseindx),1) ;
            AA = A(smallJ>0,:)' ;
            if isempty(AA)
                t = 10 ;
            else
                [y,t] = test(AA,options) ;
            end 
        end
    end
end 