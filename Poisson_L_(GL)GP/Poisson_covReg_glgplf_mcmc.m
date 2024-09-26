function [out,lastOut] = Poisson_covReg_glgplf_mcmc(y,x,k,L,R,...
    niter,priors,...
    glgp_psi0, glgp_zeta0, samp_glgp_psi,...
    samp_glgp_zeta, step_psi, step_zeta,...
    initial_pre, init_params, lastStep, varargin)


% to debug
% y = y_train;
% x = x_train;
% samp_glgp_psi = true;
% samp_glgp_zeta = true;
% step_psi = 0.1;
% step_zeta = 0.1;
% initial_pre = true;
% init_params = last_seGP;

nrep_zeta = 1;
if (~isempty(varargin))
    c = 1 ;
    while c <= length(varargin)
        switch varargin{c}
            case {'nrep_zeta'}
                nrep_zeta = varargin{c+1};
        end % switch
        c = c + 2;
    end % for
end % if

[N, n_samp] = size(y);
inds_y = ones(size(y));
inds_y = inds_y > 0;
logy_proxy = log(y + 0.5*(y == 0));

% GP hyper parameters
r = 1e-2;
sig = 1e-1;
glgp_zeta = glgp_zeta0;
glgp_psi = glgp_psi0;

if initial_pre
    
    delta = init_params.delta;
    tau = init_params.tau;
    phi = init_params.phi;
    theta = init_params.theta;
    xi = init_params.xi;
    psi = init_params.psi;
    eta = init_params.eta;
    invSig_vec = init_params.invSig_vec;
    invK_zeta = init_params.invK_zeta;
    invK_psi = init_params.invK_psi;
    zeta = init_params.zeta;
else
    
    % 1. hyp for theta
    delta = zeros(1,L);
    delta(1) = gamrnd(priors.hypers.a1,1);
    delta(2:L) = gamrnd(priors.hypers.a2*ones(1,L-1),1);
    tau = exp(cumsum(log(delta)));
    phi = gamrnd(priors.hypers.a_phi*ones(N,L),1) / priors.hypers.b_phi;
    
    % 2. theta
    theta = zeros(N,L);
    for pp=1:N
        theta(pp,:) = chol(diag(1./(phi(pp,:).*tau)))'*randn(L,1);
    end
    
    % 3. xi, psi & eta
    xi = randn(k,n_samp);
    psi = zeros(k,n_samp);
    eta = psi + xi;
    
    % 4. inv_sig
    invSig_vec = gamrnd(priors.sig.a_sig*ones(1,N),1) / priors.sig.b_sig;
    
    % 5. GP kernel
    [K_zeta, invK_zeta, logdetK_zeta] = GP_GL(glgp_zeta,sig, x, r, true); % 5.1 zeta
    [K_psi, invK_psi, logdetK_psi] = GP_GL(glgp_psi,sig, x, r, true); % 5.2 psi
    
    % 6. sample zeta
    zeta = zeros(L,k,n_samp);
    zeta = sample_zeta(logy_proxy,theta,eta,invSig_vec,zeta,invK_zeta,inds_y);
    
end

% pre-allocation: what I want to store?
theta_samp = zeros([size(theta) niter]);
zeta_samp = zeros([size(zeta) niter]);
psi_samp = zeros([size(psi) niter]);
eta_samp = zeros([size(eta) niter]);
invSig_vec_samp = zeros([length(invSig_vec) niter]);

theta_samp(:,:,1) = theta;
zeta_samp(:,:,:,1) = zeta;
psi_samp(:,:,1) = psi;
eta_samp(:,:,1) = eta;
invSig_vec_samp(:,1) = invSig_vec;
glgp_zeta_samp{1} = glgp_zeta;
glgp_psi_samp{1} = glgp_psi;

%% begin to do sampling...

eps_t_prior.lmean = [0,0]';
eps_t_prior.lvar = eye(2)*100;
for it = 2:niter
    
    if mod(it,round(niter/10)) == 0
        disp(strcat('iter =', num2str(it)));
    end
    
    
    % (1) PG augmentation
    ETA_tmp = squeeze(pagemtimes(pagemtimes(theta,zeta),...
        reshape(eta, [k,1,n_samp])));
    
    b_tmp = R + y;
    c_tmp = ETA_tmp - log(R);
    b_tmp2 = b_tmp(:);
    c_tmp2 = c_tmp(:);
    
    % PG draw
    w_tmp_raw = pgdraw_expand(b_tmp2, c_tmp2);
    w_tmp = reshape(w_tmp_raw, N, []);
    k_tmp = (y - R)/2 + w_tmp.*log(R);
    
    % yhat_tmp should be sampled, this is not the same as before
    % OK, the derivation is super super simple...
    yhat_V = 1./(w_tmp + repmat(invSig_vec',1, n_samp));
    yhat_m = yhat_V.*(k_tmp + repmat(invSig_vec',1, n_samp).*ETA_tmp);
    yhat_tmp = normrnd(yhat_m,sqrt(yhat_V));
    
    % (2) sample invSig_vec
    invSig_vec = sample_sig(yhat_tmp,theta,eta,zeta,priors.sig,inds_y);
    invSig_vec_samp(:,it) = invSig_vec;
    
    % (3) sample hyp for theta
    [phi, tau] = sample_hypers(theta,phi,tau,priors.hypers);
    
    % (4) sample theta
    theta = sample_theta(yhat_tmp,eta,invSig_vec,zeta,phi,tau,inds_y);
    theta_samp(:,:,it) = theta;
    
    % (5) sample psi
    psi = sample_psi_margxi(yhat_tmp,theta,invSig_vec,zeta,psi,invK_psi,inds_y);
    psi_samp(:,:,it) = psi;
    
    % (6) sample xi
    xi = sample_xi(yhat_tmp,theta,invSig_vec,zeta,psi,inds_y);
    eta = psi + xi;
    eta_samp(:,:,it) = eta;
    
    
    % (7) sample glgp_psi
    % let's do HMC, grid search is far more too expensive...
    % sample eps & t? or just t?
    if samp_glgp_psi
        %         llhd_func = @(eps_t) llhd_K_psi_GLGP_margin(y,eps_t,glgp_psi.k,...
        %             theta,zeta,invSig_vec, x, r, sig);


        try
%             if mod(it, 10) == 0
%                 k_grid = max(1, glgp_psi.k - 20):5:min(glgp_psi.k + 20, n_samp);
%                 n_k = size(k_grid, 2);
%                 llhd_vec = zeros(n_k, 1);
%                 warning('off')
%                 for kk = 1:n_k
%                     llhd_vec(kk) = llhd_K_glgp(psi, x,k_grid(kk),...
%                         glgp_psi.eps,glgp_psi.t,sig,true, 0);
%                 end
%                 warning('on')
%                 [~,kid] = max(llhd_vec);
%                 glgp_psi.k = k_grid(kid(1));
%             end


            llhd_func = @(eps_t) llhd_K_psi_GLGP_cond(psi,eps_t,...
                glgp_psi.k, x, r,sig);
            eps_t = eps_t_samp_hmc([glgp_psi.eps, glgp_psi.t],...
            llhd_func, eps_t_prior, step_psi);
        
            glgp_psi.eps = eps_t(1);
            glgp_psi.t = eps_t(2);
            [K_psi, invK_psi, logdetK_psi] = GP_GL(glgp_psi,sig, x, r, true);
        catch
            disp(strcat('iter =', num2str(it), ', GLGP_psi_fail'))
        end
        
    end
    glgp_psi_samp{it} = glgp_psi;
    
    
    % (8) sample zeta
    for ii=1:nrep_zeta % one can cycle through this sampling stage multiple times by adjusting num_iters
        zeta = sample_zeta(yhat_tmp,theta,eta,invSig_vec,zeta,invK_zeta,inds_y);
    end
    zeta_samp(:,:,:,it) = zeta;
    
    
    % (9) sample glgp_zeta
    if samp_glgp_zeta
        %         llhd_func = @(eps_t) llhd_K_zeta_GLGP_margin(y,eps_t,glgp_zeta.k,...
        %             theta,eta,invSig_vec, x, r, sig);

        try
            % this is more robust: only use zeta with enough contribution
            meanAbs_theta = mean(abs(theta), 1);
            zeta_use = zeta(meanAbs_theta > quantile(meanAbs_theta, 0.5),:,:);

%             if mod(it, 10) == 0
%                 k_grid = max(1, glgp_zeta.k - 20):5:min(glgp_zeta.k + 20, n_samp);
%                 n_k = size(k_grid, 2);
%                 llhd_vec = zeros(n_k, 1);
%                 warning('off')
%                 for kk = 1:n_k
%                     llhd_vec(kk) = llhd_K_glgp(reshape(zeta_use,[],n_samp), x,k_grid(kk),...
%                         glgp_zeta.eps,glgp_zeta.t,sig,true, 0);
%                 end
%                 warning('on')
%                 [~,kid] = max(llhd_vec);
%                 glgp_zeta.k = k_grid(kid(1));
% %                 disp(glgp_zeta.k)
%             end


            llhd_func = @(eps_t) llhd_K_zeta_GLGP_cond(zeta_use,eps_t,glgp_zeta.k,...
                x, r, sig);
            eps_t = eps_t_samp_hmc([glgp_zeta.eps, glgp_zeta.t],...
            llhd_func, eps_t_prior, step_zeta);
            glgp_zeta.eps = eps_t(1);
            glgp_zeta.t = eps_t(2);
            [K_zeta, invK_zeta, logdetK_zeta] = GP_GL(glgp_zeta,sig, x, r, true);
        catch
            disp(strcat('iter =', num2str(it), ', GLGP_zeta_fail'))
        end
        
    end
    glgp_zeta_samp{it} = glgp_zeta;
    
    
end

out.theta_samp = theta_samp;
out.zeta_samp = zeta_samp;
out.psi_samp = psi_samp;
out.eta_samp = eta_samp;
out.invSig_vec_samp = invSig_vec_samp;
out.glgp_zeta_samp = glgp_zeta_samp;
out.glgp_psi_samp = glgp_psi_samp;

lastOut = [];

if lastStep
    
    lastOut.delta = delta;
    lastOut.tau = tau;
    lastOut.phi = phi;
    lastOut.theta = theta;
    lastOut.xi = xi;
    lastOut.psi = psi;
    lastOut.eta = eta;
    lastOut.invSig_vec = invSig_vec;
    lastOut.invK_zeta = invK_zeta;
    lastOut.invK_psi = invK_psi;
    lastOut.zeta = zeta;
    
end




end