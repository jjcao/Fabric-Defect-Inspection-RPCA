function [A_hat E_hat F_hat] = inexact_alm_NRPCA(D, lambda,bita,tol, maxIter)
%N-RPCA model
% 
% D - m x n matrix of observations/data (required input)
% 
% lambda - weight on sparse error term in the cost function
% 
% tol - tolerance for stopping criterion.
%     - DEFAULT 1e-7 if omitted or -1.
% 
% maxIter - maximum number of iterations
%         - DEFAULT 1000, if omitted or -1.
% 
% Initialize A,E,Y,u
% while ~converged 
%   minimize (inexactly, update A and E only once)
%     L(A,E,Y,u) = |A|_* + lambda * |E|_1 + <Y,D-A-E> + bita/2*|F|_F^2+mu/2 * |D-A-E|_F^2
%   Y = Y + \mu * (D - A - E-F);
%   \mu = \rho * \mu;
% end
% 
% Minming Chen, October 2009. Questions? v-minmch@microsoft.com ; 
% Arvind Ganesh (abalasu2@illinois.edu)
% 
% Copyright: Perception and Decision Laboratory, University of Illinois, Urbana-Champaign
%            Microsoft Research Asia, Beijing

addpath PROPACK;

[m n] = size(D);

if nargin < 2
    lambda = 1 / sqrt(m);
end
if nargin < 3
    bita = 1e-7 / sqrt(m);
end
% if nargin<4
%     perior=ones(m,n);
% end
if nargin <4
    tol = 1e-7;
elseif tol == -1
    tol = 1e-7;
end

if nargin < 5
    maxIter = 1000;
elseif maxIter == -1
    maxIter = 1000;
end
% initialize
Y = D;
norm_two = lansvd(Y, 1, 'L');
norm_inf = norm( Y(:), inf) / lambda;
dual_norm = max(norm_two, norm_inf);
Y = Y / dual_norm;
J=zeros(m,n);
A_hat = zeros( m, n);
E_hat = zeros( m, n);
%% 改进 加入小噪声部分 F
F_hat = zeros( m, n);
%% 
mu = 1.25/norm_two ;% this one can be tuned

mu_bar = mu * 1e7;

rho = 1.5   ;       % this one can be tuned
d_norm = norm(D, 'fro');

iter = 0;
total_svd = 0;
converged = false;
stopCriterion = 1;
sv = 10;
while ~converged       
    iter = iter + 1;
    temp_T = D - A_hat -F_hat+ (1/mu)*Y;
    E_hat = max(temp_T - lambda/mu, 0);
    E_hat = E_hat+min(temp_T + lambda/mu, 0);
    F_hat=(mu*(D-A_hat-E_hat)+Y)/(bita+mu);
    dey=D - E_hat-F_hat + (1/mu)*Y;
    temp = dey*J';
    [U , ~, V] = svd(temp,'econ');
    Q = U*V';
    temp = Q'*dey;
    [U S V] = svd(temp,'econ');
    diagS = diag( S );
    diagS = sign( diagS ) .* max( abs( diagS ) - 1/mu,0);
    J = U * diag( diagS ) * V';   
    A_hat=Q*J;  
    total_svd = total_svd + 1;  
    Z = D - A_hat - E_hat-F_hat;
    Y = Y + mu*Z;
    mu = min(mu*rho, mu_bar);
% stop Criterion    
    stopCriterion = norm(Z, 'fro') / d_norm;
    if stopCriterion < tol
        converged = true;
    end    
    
    if mod( total_svd, 10) == 0
        disp(['#svd ' num2str(total_svd) ' r(A) ' num2str(rank(A_hat))...
            ' |E|_0 ' num2str(length(find(abs(E_hat)>0)))...
            ' stopCriterion ' num2str(stopCriterion)]);
    end    
    
    if ~converged && iter >= maxIter
        %disp('Maximum iterations reached') ;
        converged = 1 ;       
    end
end

