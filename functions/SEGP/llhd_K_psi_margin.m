function llhd = llhd_K_psi_margin(y,c_psi,theta,zeta,invSig_vec, x, r)

zeta_perm = permute(zeta, [1,3,2]);
K = size(zeta, 2);
[p, N] = size(y);

[K_psi, ~, ~] = GP_SE(c_psi, x, r, true);

tmp_mat = sparse(diag(repmat(1./invSig_vec,1,N)));
for kk = 1:K
    
    Lam_tmp = theta*zeta_perm(:,:,kk);
    C = mat2cell(Lam_tmp, size(Lam_tmp,1), ones(1,size(Lam_tmp,2)));
    diag_tmp = sparse(blkdiag(C{:}));
    
    tmp_mat = tmp_mat + diag_tmp*(K_psi + eye(N))*diag_tmp';
end

llhd = normpdfln(y(:),zeros(p*N,1),tmp_mat);

end