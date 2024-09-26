function llhd = llhd_K_psi_GLGP_cond(psi,eps_t,k_psi, x, r,sig)


glgp_psi_tmp.eps = eps_t(1);
glgp_psi_tmp.k = k_psi;
glgp_psi_tmp.t = eps_t(2);

[k, N] = size(psi);

try
    [K_psi, ~, ~] = GP_GL(glgp_psi_tmp,sig, x, r, true);
    llhd = 0;
    for kk = 1:k
        llhd = llhd + normpdfln(psi(kk,:),zeros(N,1),K_psi);
    end
catch
        llhd = -Inf;
end


end