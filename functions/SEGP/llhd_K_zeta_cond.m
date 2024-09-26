function llhd = llhd_K_zeta_cond(zeta,c_zeta, x, r)

[L, k, N] = size(zeta);
[~,invK_zeta,logdetK_zeta] = GP_SE(c_zeta, x, r, true);

zeta_tmp = reshape(zeta,[L*k N])';
llhd = -0.5*sum(diag(zeta_tmp'*(invK_zeta*zeta_tmp)))- 0.5*(L*k)*logdetK_zeta;

end