function llhd = llhd_K_zeta_GLGP_cond(zeta,eps_t,k_zeta, x, r, sig)

[L, k, N] = size(zeta);
glgp_zeta_tmp.eps = eps_t(1);
glgp_zeta_tmp.k = k_zeta;
glgp_zeta_tmp.t = eps_t(2);

[~,invK_zeta,logdetK_zeta] = GP_GL(glgp_zeta_tmp,sig, x, r, true);
zeta_tmp = reshape(zeta, L*k, N)';
llhd = -0.5*sum(diag(zeta_tmp'*(invK_zeta*zeta_tmp)))- 0.5*(L*k)*logdetK_zeta;


end