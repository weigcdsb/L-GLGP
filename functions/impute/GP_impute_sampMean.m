function [mean_q, Sig_q, var_q] = GP_impute_sampMean(fitRes,...
    n_q, xy_samp, xy_q, niter, GLGP, r, sig, idx_use)

% should be much better, bro
% after doing this, I cannot do anything else...
% to debug...
% fitRes = out_seGP;
% n_q = n_imp;
% xy_samp = xy_train;
% xy_q = xy_imp;
% niter = niter_pre;
% GLGP = false;
% r = 1e-5;
% sig = 1e-2;
% idx_use = round(niter/2):niter; 


N = size(fitRes.theta_samp, 1);
L = size(fitRes.zeta_samp(:,:,:,1), 1);
k = size(fitRes.zeta_samp(:,:,:,1), 2);
n_samp = size(fitRes.zeta_samp(:,:,:,1), 3);
xy_comb = [xy_q; xy_samp];

mean_q_samp_sum = zeros(N, n_q);
Sig_q_samp_sum = zeros(N, N, n_q);
var_q_samp_sum = zeros(N, n_q);


if GLGP % GL-GP
    psi_eps_samp = zeros(niter, 1);
    psi_t_samp = zeros(niter, 1);
    psi_k_samp = zeros(niter, 1);

    zeta_eps_samp = zeros(niter, 1);
    zeta_t_samp = zeros(niter, 1);
    zeta_k_samp = zeros(niter, 1);
    
    for it = 1:niter
        psi_eps_samp(it) = fitRes.glgp_psi_samp{it}.eps;
        psi_t_samp(it) = fitRes.glgp_psi_samp{it}.t;
        psi_k_samp(it) = fitRes.glgp_psi_samp{it}.k;

        zeta_eps_samp(it) = fitRes.glgp_zeta_samp{it}.eps;
        zeta_t_samp(it) = fitRes.glgp_zeta_samp{it}.t;
        zeta_k_samp(it) = fitRes.glgp_zeta_samp{it}.k;

    end
    
    glgp_psi_samp.eps = mean(psi_eps_samp(idx_use));
    glgp_psi_samp.k = round(median(psi_k_samp(idx_use)));
    glgp_psi_samp.t = mean(psi_t_samp(idx_use));
    
    glgp_zeta_samp.eps = mean(zeta_eps_samp(idx_use));
    glgp_zeta_samp.k = round(median(zeta_k_samp(idx_use)));
    glgp_zeta_samp.t = mean(zeta_t_samp(idx_use));
    
    Sig_psi_q = sparse(GP_GL(glgp_psi_samp,sig, xy_comb, r, true));
    Sig_zeta_q = sparse(GP_GL(glgp_zeta_samp,sig, xy_comb, r, true));
    
else % SE-GP
    c_psi = mean(fitRes.c_psi_samp(idx_use));
    c_zeta = mean(fitRes.c_zeta_samp(idx_use));
    
    Sig_psi_q = sparse(GP_SE(c_psi, xy_comb, r, true));
    Sig_zeta_q = sparse(GP_SE(c_zeta, xy_comb, r, true));
end


for it = idx_use
    disp(it)


%     Sig_psi_q = sparse(GP_GL(fitRes.glgp_psi_samp{it},sig, xy_comb, r, true));
%     Sig_zeta_q = sparse(GP_GL(fitRes.glgp_zeta_samp{it},sig, xy_comb, r, true));

    [psi_samp_q, psi_mu_q, psi_Sig_q] =...
        condNormSamp(xy_q, fitRes.psi_samp(:,:,it)', Sig_psi_q);
    
    zeta_samp_obs_reshape = reshape(fitRes.zeta_samp(:,:,:,it),[L*k n_samp])';
    [zeta_samp_reshape_q, zeta_mu_reshape_q, zeta_Sig_reshape_q] =...
        condNormSamp(xy_q, zeta_samp_obs_reshape, Sig_zeta_q);
    zeta_samp_q = reshape(zeta_samp_reshape_q', [L,k,n_q]);
    zeta_mu_q = reshape(zeta_mu_reshape_q', [L,k,n_q]);
    
    for nn = 1:n_q
        Lam_tmp = fitRes.theta_samp(:,:,it)*zeta_samp_q(:,:,nn);
        mean_q_samp_sum(:,nn) =...
            mean_q_samp_sum(:,nn) + Lam_tmp * psi_samp_q(nn,:)';
        Sig_q_samp_sum(:,:,nn) =...
            Sig_q_samp_sum(:,:,nn) + Lam_tmp * Lam_tmp' +...
            diag(1./fitRes.invSig_vec_samp(:,it));
        if it == niter
            var_q_samp_sum(:,nn) = diag(Sig_q_samp_sum(:,:,nn));
        end
    end
end

mean_q = mean_q_samp_sum./length(idx_use);
Sig_q = Sig_q_samp_sum./length(idx_use);
var_q = var_q_samp_sum./length(idx_use);

end