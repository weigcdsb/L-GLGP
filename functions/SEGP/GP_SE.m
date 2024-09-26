function [K, invK, logdetK] = GP_SE(c, x, r, useCorr)

% to debug
% c = c_zeta;
% useCorr = true;

N = size(x, 1);
K = zeros(N);
for ii=1:N
    for jj=1:N
        dist2_ii_jj = sum((x(ii,:)-x(jj,:)).^2);
        K(ii,jj) = exp(-c*(dist2_ii_jj));
    end
end
K = K + diag(r*ones(1,N));
if useCorr
   K = corrcov(K); 
end
invK = inv(K);
logdetK = 2*sum(log(diag(chol(K))));

end