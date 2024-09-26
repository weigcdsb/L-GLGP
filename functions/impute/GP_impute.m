function [mean_q, Sig_q, var_q] = GP_impute(fitRes,...
    n_q, xy_samp, xy_q, niter, GLGP, r, sig)

% to debug
% fitRes = out;
% fitRes = out_seGP;
% n_q = 1000;
% xy_samp = x;
% idx_q = randsample(setdiff(1:n_all, id_samp), n_q);
% xy_all;
% niter;
% GLGP = true;
% r = 1e-2;

N = size(fitRes.theta_samp, 1);
L = size(fitRes.zeta_samp(:,:,:,1), 1);
k = size(fitRes.zeta_samp(:,:,:,1), 2);
n_samp = size(fitRes.zeta_samp(:,:,:,1), 3);

% xy_q = xy_all(idx_q,:);
xy_comb = [xy_q; xy_samp];
idx_use = round(niter/2):niter; 

psi_samp_obs = mean(fitRes.psi_samp(:,:,idx_use), 3);

if GLGP % GL-GP
    psi_eps_samp = zeros(niter, 1);
    psi_t_samp = zeros(niter, 1);
    zeta_eps_samp = zeros(niter, 1);
    zeta_t_samp = zeros(niter, 1);
    
    for it = 1:niter
        psi_eps_samp(it) = fitRes.glgp_psi_samp{it}.eps;
        psi_t_samp(it) = fitRes.glgp_psi_samp{it}.t;
        zeta_eps_samp(it) = fitRes.glgp_zeta_samp{it}.eps;
        zeta_t_samp(it) = fitRes.glgp_zeta_samp{it}.t;
    end
    
    glgp_psi_samp.eps = mean(psi_eps_samp(idx_use));
    glgp_psi_samp.k = fitRes.glgp_psi_samp{1}.k;
    glgp_psi_samp.t = mean(psi_t_samp(idx_use));
    
    glgp_zeta_samp.eps = mean(zeta_eps_samp(idx_use));
    glgp_zeta_samp.k = fitRes.glgp_zeta_samp{1}.k;
    glgp_zeta_samp.t = mean(zeta_t_samp(idx_use));
    
    Sig_psi_all = sparse(GP_GL(glgp_psi_samp,sig, xy_comb, r, true));
    Sig_zeta_all = sparse(GP_GL(glgp_zeta_samp,sig, xy_comb, r, true));
    
else % SE-GP
    c_psi = mean(fitRes.c_psi_samp(idx_use));
    c_zeta = mean(fitRes.c_zeta_samp(idx_use));
    
    Sig_psi_all = sparse(GP_SE(c_psi, xy_comb, r, true));
    Sig_zeta_all = sparse(GP_SE(c_zeta, xy_comb, r, true));
    
end

zeta_samp_obs = mean(fitRes.zeta_samp(:,:,:,idx_use), 4);
theta_samp = mean(fitRes.theta_samp(:,:,idx_use), 3);
Sig_samp = diag(1./mean(fitRes.invSig_vec_samp(:,idx_use), 2));


[psi_samp_q, psi_mu_q, psi_Sig_q] =...
    condNormSamp(xy_q, psi_samp_obs', Sig_psi_all);

zeta_samp_obs_reshape = reshape(zeta_samp_obs,[L*k n_samp])';
[zeta_samp_reshape_q, zeta_mu_reshape_q, zeta_Sig_reshape_q] =...
    condNormSamp(xy_q, zeta_samp_obs_reshape, Sig_zeta_all);
zeta_samp_q = reshape(zeta_samp_reshape_q', [L,k,n_q]);
zeta_mu_q = reshape(zeta_mu_reshape_q', [L,k,n_q]);

mean_q = zeros(N, n_q);
Sig_q = zeros(N,N,n_q);
var_q = zeros(N, n_q);
for nn = 1:n_q
    Lam_tmp = theta_samp * zeta_mu_q(:,:,nn);
    mean_q(:,nn) = Lam_tmp * psi_mu_q(nn,:)';
    
    Sig_q(:,:,nn) = Lam_tmp * Lam_tmp' + Sig_samp;
    var_q(:,nn) = diag(Sig_q(:,:,nn));
end




end