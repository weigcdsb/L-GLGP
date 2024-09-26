function llhd = llhd_K_psi_GLGP_margin(y,eps_t,k_psi,theta,zeta,invSig_vec, x, r, sig)

% to debug...
% eps_t = [glgp_psi.eps, glgp_psi.t];
% k_psi = glgp_psi.k;

zeta_perm = permute(zeta, [1,3,2]);
k_zeta = size(zeta, 2);
[p, N] = size(y);

glgp_psi_tmp.eps = eps_t(1);
glgp_psi_tmp.k = k_psi;
glgp_psi_tmp.t = eps_t(2);

[K_psi, ~, ~] = GP_GL(glgp_psi_tmp,sig, x, r, true);

tmp_mat = sparse(diag(repmat(1./invSig_vec,1,N)));
for kk = 1:k_zeta
    
    Lam_tmp = theta*zeta_perm(:,:,kk);
    C = mat2cell(Lam_tmp, size(Lam_tmp,1), ones(1,size(Lam_tmp,2)));
    diag_tmp = sparse(blkdiag(C{:}));
    
    tmp_mat = tmp_mat + diag_tmp*(K_psi + eye(N))*diag_tmp';
end

llhd = normpdfln(y(:),zeros(p*N,1),tmp_mat);

end