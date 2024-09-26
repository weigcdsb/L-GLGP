function c_samp = c_samp_gridSearch(c_now, llhd_func, nGrid)

% to debug
% c_now = 10;
% llhd_func = @(c_psi) llhd_K_psi_margin(y,c_psi,theta,zeta,invSig_vec, x, r)
% nGrid = 10;

c_grid = linspace(max(0.1, c_now - 1), c_now + 1, nGrid); % just use uniform prior
Pk = zeros(1, nGrid)*-Inf;
for kk = 1:nGrid
    Pk(kk) = llhd_func(c_grid(kk));
end

% why you sample like this?
Pk = cumsum(exp(Pk-max(Pk)));
c_ind = 1 + sum(Pk(end)*rand(1) > Pk);
c_samp = c_grid(c_ind);

end