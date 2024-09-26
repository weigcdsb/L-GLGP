function [K, invK, logdetK] = GP_GL(glgp_psi,sig, x, r, useCorr)

% to debug
% useCorr = true;

N = size(x, 1);
K = GLGP_cov(x,glgp_psi.k,glgp_psi.eps,glgp_psi.t,sig,useCorr);
K = K + diag(r*ones(1,N));
if useCorr
   K = corrcov(K); 
end

invK = inv(K);
logdetK = 2*sum(log(diag(chol(K))));

end
