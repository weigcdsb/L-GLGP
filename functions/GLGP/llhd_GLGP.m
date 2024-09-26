function llhd = llhd_GLGP(y,X,k,eps,t, sig)

% to debug
% y
% X = X1;
% k = 1;
% eps = 1e-6;
% t = 1;
% sig = 1e-3;

if nargin<6, sig = 1e-3; end

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
C = zeros(n,n);
% logDet = 0;
for kk=1:k
    s_tmp = t*(S(kk,kk)-1)/eps;
%     logDet = logDet + s_tmp;
    C = C + exp(s_tmp)*U(1:n,kk)*U(1:n,kk)'/norm(U(1:n,kk))^2*n;
end
% the imaginary part is usually very small: just drop it

% if sum(abs(imag(C)), 'all') ~= 0
%     llhd = -Inf;
%     return;
% end
C = real(C);

% log(det(C + eps_noise*eye(n)))
% log(det(C))
% llhd = -0.5*logDet - 0.5*y'*pinv(C)*y; % - 0.5*n*log(2*pi);

llhd = -0.5*log(det(C + sig*eye(n))) - ...
    0.5*y'*inv(C + sig*eye(n))*y; % - 0.5*n*log(2*pi);

end