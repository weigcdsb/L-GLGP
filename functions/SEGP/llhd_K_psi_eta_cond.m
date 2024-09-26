function llhd = llhd_K_psi_eta_cond(eta,c_psi, x, r)

% to debug...

[k, N] = size(eta);
[K_psi,~,~] = GP_SE(c_psi, x, r, true);

llhd = 0;
for kk = 1:k
    llhd = llhd + normpdfln(eta(kk,:),zeros(N,1),K_psi + eye(N));
end

end
