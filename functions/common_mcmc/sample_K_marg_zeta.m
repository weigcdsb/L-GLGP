function K_ind = sample_K_marg_zeta(y,theta,eta,invSig_vec,prior_params,K_ind)

[k N] = size(eta);
[p L] = size(theta);
c_prior = prior_params.c_prior;
K = prior_params.K;

grid_size = length(c_prior);

Pk = -Inf*ones(1,grid_size);
Pk_tmp = -Inf*ones(1,grid_size);

% For computational reasons, one can restrict to examining just a local
% neighborhood of the current length scale parameter, but not exact:
nbhd = Inf; %ceil(grid_size/100);
nbhd_vec = [max(K_ind-nbhd,1):min(K_ind+nbhd,grid_size)];

tmp_mat_init = diag(repmat(1./invSig_vec,1,N));

if p < sqrt(N)
    
    for cc=nbhd_vec
        Kc = K(:,:,cc);
        tmp_mat = tmp_mat_init;
        for ll=1:L
            theta_ll = theta(:,ll);
            for kk=1:k
                eta_kk = eta(kk,:);
                
                tmp_mat = tmp_mat + kron((eta_kk*eta_kk').*Kc,theta_ll*theta_ll');
            end
        end
        
        Pk(cc) = normpdfln(y(:),zeros(p*N,1),tmp_mat);
    end
    
else
    
    for cc=nbhd_vec
        Kc = K(:,:,cc);
        Kc = sparse(Kc);
        tmp_mat = tmp_mat_init;
        for ll=1:L
            theta_ll = theta(:,ll);
            for kk=1:k
                eta_kk = eta(kk,:);
                eta_mat_kk = diag(eta_kk);
                eta_theta_ll_kk = kron(eta_mat_kk,theta_ll);
                
                tmp_mat = tmp_mat + eta_theta_ll_kk*Kc*eta_theta_ll_kk';
            end
        end

        Pk(cc) = normpdfln(y(:),zeros(p*N,1),tmp_mat);
    end
    
end

nbhd_mask = zeros(1,length(prior_params.c_prior));
nbhd_mask(nbhd_vec) = 1;
Pk = Pk + log(prior_params.c_prior).*nbhd_mask;

Pk = cumsum(exp(Pk-max(Pk)));
K_ind = 1 + sum(Pk(end)*rand(1) > Pk);

return;

