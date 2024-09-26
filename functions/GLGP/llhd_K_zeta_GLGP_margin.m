function llhd = llhd_K_zeta_GLGP_margin(y,eps_t,k_zeta,...
    theta,eta,invSig_vec, x, r, sig)

% to debug
% y
% eps_t
% theta
% eta
% invSig_vec
% x
% r
% sig

[k, N] = size(eta);
[p, L] = size(theta);

glgp_zeta_tmp.eps = eps_t(1);
glgp_zeta_tmp.k = k_zeta;
glgp_zeta_tmp.t = eps_t(2);

[K_zeta, ~, ~] = GP_GL(glgp_zeta_tmp,sig, x, r, true);


tmp_mat = sparse(diag(repmat(1./invSig_vec,1,N)));
if p < sqrt(N)
    
    for ll=1:L
        theta_ll = theta(:,ll);
        for kk=1:k
            eta_kk = eta(kk,:);
            tmp_mat = tmp_mat + kron((eta_kk*eta_kk').*K_zeta,theta_ll*theta_ll');
        end
    end
    
else
    
    for ll=1:L
        theta_ll = theta(:,ll);
        for kk=1:k
            eta_kk = eta(kk,:);
            eta_mat_kk = diag(eta_kk);
            eta_theta_ll_kk = kron(eta_mat_kk,theta_ll);
            
            tmp_mat = tmp_mat + eta_theta_ll_kk*K_zeta*eta_theta_ll_kk';
        end
    end
    
end


llhd = normpdfln(y(:),zeros(p*N,1),tmp_mat);


end
