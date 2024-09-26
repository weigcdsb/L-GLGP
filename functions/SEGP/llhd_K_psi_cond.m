function llhd = llhd_K_psi_cond(psi,c_psi, x, r)

% to debug...

[k, N] = size(psi);
[K_psi,~,~] = GP_SE(c_psi, x, r, true);

llhd = 0;
for kk = 1:k
    llhd = llhd + normpdfln(psi(kk,:),zeros(N,1),K_psi);
end

end
