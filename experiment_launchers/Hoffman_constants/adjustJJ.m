function [JJ] = adjustJJ(JJ,I,FF) 
    % [JJ] = adjustJJ(JJ,I) 
    % update the collection JJ by using new certificate of non-surjectivity I 
    % and all certificates of surjectivity FF
    m=numel(I) ;
    sizeI = numel(find(I>0)) ;
    indx = find(JJ*I' == sizeI) ;
    newJ = JJ(indx,:) ;  % sets in JJ containing I
    JJ(indx,:) = [] ; 
    % refine newJ
    j = find(I>0) ; i = 1:numel(j) ;
    matI = full(sparse(i,j,ones(numel(j),1))) ;
    matI = [matI,zeros(sizeI,m-size(matI,2))] ;
    newallJ = zeros(sizeI*numel(indx),m) ;
    for i=1:numel(indx)
       newallJ((i-1)*sizeI+1:i*sizeI,:) = ones(sizeI,1)*newJ(i,:)-matI ;
    end ;
    % trim newallJ using FF.  To that end, delete the sets in newallJ that 
    % are contained in some set in FF
    if (size(FF,1) > 0)
        indx = find(max([FF*newallJ';zeros(1,size(newallJ,1))]) == sum(newallJ')) ;
        newallJ(indx,:) = [] ;
    end ;    
    % Add new sets to JJ avoiding repetitions
    [~,indx] = unique(newallJ*2.^(0:m-1)') ;
    JJ = [JJ; newallJ(indx,:)] ;     
end

