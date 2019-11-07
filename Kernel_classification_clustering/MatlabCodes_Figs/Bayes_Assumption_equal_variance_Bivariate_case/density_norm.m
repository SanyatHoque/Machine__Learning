
%% Implements Normal probability density with parameters mu and covariance 
%X - points at which probability density is evaluated 
function probability_density = density_norm(X,mu,cov_mat);
[n,d] = size(X);
% X = (X-ones(n,1)*mu);
probability_density = 1/((2*pi)^(d/2)*sqrt(det(cov_mat)))*exp((-0.5)*diag((X-ones(n,1)*mu)*inv(cov_mat)*(X-ones(n,1)*mu)'));

