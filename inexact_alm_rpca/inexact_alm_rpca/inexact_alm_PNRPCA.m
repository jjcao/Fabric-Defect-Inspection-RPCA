function [A_hat E_hat F_hat] = inexact_alm_PNRPCA(D,lambda,bita,perior,tol, maxIter)
%%    minimize (inexactly, update A and E and F only once)
%     L(A,E,F,Y,u) = |A|_* + lambda * |w*E|_1 + <Y,D-A-E-F> + bita/2*|F|_F^2+mu/2 * |D-A-E|_F^2
%   Y = Y + \mu * (D - A - E-F);
%   \mu = \rho * \mu;
addpath PROPACK;

[m n] = size(D);

if nargin < 2
    lambda = 1 / sqrt(m);
end
if nargin < 3
    bita = 1e-7 / sqrt(m);
end
if nargin<4
    perior=ones(m,n);
end
if nargin < 5
    tol = 1e-7;
elseif tol == -1
    tol = 1e-7;
end

if nargin < 6
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
    E_hat = max(temp_T - lambda/mu*exp(-perior), 0);
    E_hat = E_hat+min(temp_T + lambda/mu*exp(-perior), 0);
    F_hat=(mu*(D-A_hat-E_hat)+Y)/(bita+mu);
%     if choosvd(n, sv) == 1
%         [U S V] = lansvd(D - E_hat-F_hat + (1/mu)*Y, sv, 'L');
%     else
%         [U S V] = svd(D - E_hat-F_hat + (1/mu)*Y, 'econ');
%     end   
%     diagS = diag(S);
%     svp = length(find(diagS > 1/mu));
%     if svp < sv
%         sv = min(svp + 1, n);
%     else
%         sv = min(svp + round(0.05*n), n);
%     end
%     
%     
%     A_hat = U(:, 1:svp) * diag(diagS(1:svp) - 1/mu) * V(:, 1:svp)';   
    
    %%
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
    %%
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

%% 余弦基
% function [L,E_hat,F_hat] = inexact_alm_withW(D,lambda,alfa,bita,B1,B2,perior,tol, maxIter)
% addpath PROPACK;
% 
% [m n] = size(D);
% 
% if nargin < 2
%     lambda = 1 / sqrt(m);
% end
% if nargin < 3
%     alfa = 1 / sqrt(m);
% end
% if nargin < 4
%     bita = 1e-7 / sqrt(m);
% end
% if nargin < 5
%     B1=dctmtx(m);
% end
%  if nargin < 6
%     B2=dctmtx(n);
%  end
%  if nargin<7
%      pathch_size=16;
%  end
%  if nargin<8
%      over_size=8;
%  end
%  if nargin < 7
%      perior=ones(m,n);
%  end
% if nargin < 8
%     tol = 1e-7;
% elseif tol == -1
%     tol = 1e-7;
% end
% 
% if nargin < 9
%     maxIter = 1000;
% elseif maxIter == -1
%     maxIter = 1000;
% end
% 
% %initialize
% %Y1=zeros(m,n);
% 
% J=zeros(m,n);
% A_hat = zeros( m, n);
% E_hat = zeros( m, n);
% W_hat=  zeros( m, n);
% % 改进 加入小噪声部分 F
% F_hat = zeros( m, n);
% Y2= D;
%  norm_two = lansvd(D, 1, 'L');
% norm_inf = norm( Y2(:), inf) / lambda;
% dual_norm = max(norm_two, norm_inf);
% Y2= Y2 / dual_norm;
% % Y1=zeros( m, n);
% Y1=Y2;
% % 
% mu = 1.25/norm_two ;% this one can be tuned
% 
% yita=3;
% mu_bar = mu * 1e7;
% % mu_bar=1e6;
% % mu=1e-6;
% rho = 1.5   ;       % this one can be tuned
% d_norm = norm(D, 'fro');
% 
% iter = 0;
% total_svd = 0;
% converged = false;
% stopCriterion = 1;
% sv = 10;
% while ~converged       
%     iter = iter + 1;
%     dey=W_hat-(1/mu)*Y1;
%     temp = dey*J';
%     [U , ~, V] = svd(temp,'econ');
%     Q = U*V';
%     temp = Q'*dey;
%     [U S V] = svd(temp,'econ');
%     diagS = diag( S );
%     diagS = sign( diagS ) .* max( abs( diagS ) - 1/mu,0);
%     svp=length(diagS);
%     J = U * diag( diagS ) * V';
%     
%     A_hat=Q*J;
%     
% %      if choosvd(n, sv) == 1
% %         [U S V] = lansvd(temp1, sv, 'L');
% %     else
% %         [U S V] = svd(temp1, 'econ');
% %      end
% %     diagS = diag(S);
% %     svp = length(find(diagS > 1/mu));
% %     if svp < sv
% %         sv = min(svp + 1, n);
% %     else
% %         sv = min(svp + round(0.05*n), n);
% %     end   
% %    A_hat = U(:, 1:svp) * diag(diagS(1:svp) - 1/mu) * V(:, 1:svp)'; 
%   
%    temp2= W_hat-(1/yita)*(W_hat-A_hat-Y1/mu+B1'*(B1*W_hat*B2'+E_hat+F_hat-D+Y2/mu)*B2);
%    %W_hat=sign(W_hat).*max(abs(W_hat)-lambda/(mu*yita),0);
%    
%    W_hat = max(temp2 - lambda/(mu*yita), 0);
%  W_hat = W_hat+min(temp2 +lambda/(mu*yita), 0);
%  %temp3=D-B1*W_hat*B2'-F_hat-1/mu*Y2;  
% temp3=E_hat-1/yita*(B1*W_hat*B2'+E_hat+F_hat-D+Y2/mu);
%  % E_hat = sign(temp3).*max(abs(temp3)-alfa/mu*exp(-perior), 0);
%   E_hat = max(temp3 - alfa/(mu*yita)*exp(-perior), 0);  
%  E_hat = E_hat+min(temp3 +alfa/(mu*yita)*exp(-perior), 0);
% % E_hat=sign(E_hat).*max(abs(temp3)-alfa/mu,0);
% 
%     temp4=D-B1*W_hat*B2'-E_hat-Y2/mu;
%     F_hat=mu*temp4/(bita+mu);
% 
%       
%     total_svd = total_svd + 1;
%     
%     Z = D - E_hat-F_hat-B1*W_hat*B2';
%     Y1 = Y1 + mu*(A_hat-W_hat);
%      Y2 = Y2 - mu*Z;
%     mu = min(mu*rho, mu_bar);
%    
%     % stop Criterion    
%     stopCriterion = norm(Z, 'fro') / d_norm;
%     if stopCriterion < tol
%         converged = true;
%     end    
%     
%     if mod( total_svd, 10) == 0
%         disp(['#svd ' num2str(total_svd) ' r(A) ' num2str(rank(A_hat))...
%             ' |E|_0 ' num2str(length(find(abs(E_hat)>0)))...
%             ' stopCriterion ' num2str(stopCriterion)]);
%     end    
%     
%     if ~converged && iter >= maxIter
%         disp('Maximum iterations reached') ;
%         converged = 1 ;       
%     end
% end
% L=B1*W_hat*B2';