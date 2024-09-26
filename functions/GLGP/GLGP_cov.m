function K = GLGP_cov(X,k,eps,t,sig,useCorr, sig2)

if nargin < 7
    sig2 = 0;
end


% to debug
% X,k,eps,t,sig,useCorr
% X = x;
% k = 50;
% eps = 0.1;
% t = 1;
% sig = 1e-5;
% useCorr = true;


[n, ~] = size(X) ;

[index,distance]= knnsearch(X, X,'k', n);
distance(:,1)=0; % OK, that's a bit wired.
ker = exp(-distance.^2/(4*eps));
ii = (1:n)'*ones(1,n); % 1:n for n columns
W = sparse(ii, index, ker, n, n); % create sparse matrix
D = sum(W, 2);
W = bsxfun(@rdivide, bsxfun(@rdivide, W, D), transpose(D));
D = sqrt(sum(W, 2)); 
W = bsxfun(@rdivide, bsxfun(@rdivide, W, D), transpose(D));
[U,S] = eigs(W, k);
U = bsxfun(@rdivide, U(:, 1:end), D);

% Construct heat kernel
K = zeros(n,n);
for kk=1:k
    s_tmp = t*(S(kk,kk)-1)/eps;
    K = K + exp(s_tmp)*U(1:n,kk)*U(1:n,kk)'/norm(U(1:n,kk))^2*n;
end

K = real(K);
K = K + sig*eye(n);
if useCorr
   K = corrcov(K); 
end
K = K + sig2*eye(n);

end