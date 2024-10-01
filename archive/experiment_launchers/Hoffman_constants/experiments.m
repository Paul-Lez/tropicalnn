% Run code for Hoffman constant computation on a collection
% of N randomly chosen instances m by n.
% Save the output  (H,|FF|,|II|,maxJJ) 
% in the N by 4 matrix "results"
m = 20; n = 10; N = 50 ;
results = zeros(4,N) ;
for i=1:N
    i
    A = randn(m,n) ; 
    tic
    [H,FF,II,maxJJ] = Hoffman(A) ;
    toc
    results(:,i) = [H,size(FF,1),size(II,1),maxJJ]; 
end ;

% Draw boxplots for non-surjective instances
figure
indx = find(results(2,:)>1) ;
boxplot(results(2:3,indx)','symbol', 'k.') ;
title('boxplots of |FF| and |II|') ;