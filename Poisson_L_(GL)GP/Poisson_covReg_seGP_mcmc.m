function [out,lastOut] = Poisson_covReg_seGP_mcmc(y,x,k,L,R,niter,priors,...
    c_psi0, c_zeta0, samp_c_psi, samp_c_zeta,...
    samp_c_psi_marg, samp_c_zeta_marg,...
    step_psi, step_zeta, lastStep, varargin)

% to debug
% R = 10*ones(size(y));
% niter = niter_pre;
% c_psi0 = 10;
% c_zeta0 = 10;
% samp_c_psi = true;
% samp_c_zeta = true;
% step_psi = 0.01;
% step_zeta = 0.01;
% lastStep = true;
start = false;
startInput = {};
[N, n_samp] = size(y);
inds_y = ones(size(y));
inds_y = inds_y > 0;
logy_proxy = log(y + 1e-2*(y == 0));

nrep_zeta = 1;
if (~isempty(varargin))
    c = 1 ;
    while c <= length(varargin)
        switch varargin{c}
            case {'start'}
                start = varargin{c+1};
            case {'startInput'}
                startInput = varargin{c+1};
            case {'nrep_zeta'}
                nrep_zeta = varargin{c+1};
        end % switch
        c = c + 2;
    end % for
end % if


% 1. hyp for theta
delta = zeros(1,L);
delta(1) = gamrnd(priors.hypers.a1,1);
delta(2:L) = gamrnd(priors.hypers.a2*ones(1,L-1),1);
tau = exp(cumsum(log(delta)));
phi = gamrnd(priors.hypers.a_phi*ones(N,L),1) / priors.hypers.b_phi;

% GP kernel
r = 1e-5;
c_zeta = c_zeta0;
[K_zeta, invK_zeta, logdetK_zeta] = GP_SE(c_zeta, x, r, true); % 5.1 zeta
c_psi = c_psi0;
[K_psi, invK_psi, logdetK_psi] = GP_SE(c_psi, x, r, true); % 5.2 psi


if start % this for debug...
    theta = startInput.theta;
    eta = startInput.eta;
    psi = startInput.psi;
    xi = eta - psi;
    invSig_vec = startInput.invSig_vec;
    zeta = startInput.zeta;
else
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

    % 5. sample zeta
    zeta = zeros(L,k,n_samp);
    for ii = 1:50
        zeta = sample_zeta(logy_proxy,theta,eta,invSig_vec,zeta,invK_zeta,inds_y);
    end
end

% 7. pre-allocation: what I want to store?
theta_samp = zeros([size(theta) niter]);
zeta_samp = zeros([size(zeta) niter]);
psi_samp = zeros([size(psi) niter]);
eta_samp = zeros([size(eta) niter]);
invSig_vec_samp = zeros([length(invSig_vec) niter]);
c_zeta_samp = zeros([length(c_zeta) niter]);
c_psi_samp = zeros([length(c_psi) niter]);

theta_samp(:,:,1) = theta;
zeta_samp(:,:,:,1) = zeta;
psi_samp(:,:,1) = psi;
eta_samp(:,:,1) = eta;
invSig_vec_samp(:,1) = invSig_vec;
c_psi_samp(1) = c_psi;
c_zeta_samp(1) = c_zeta;

%%

% tt1 = squeeze(pagemtimes(pagemtimes(theta,zeta), reshape(eta, [k,1,n_samp])));
% tt2 = zeros(N,n_samp);
% for nn = 1:n_samp
%     tt2(:,nn) = theta*zeta(:,:,nn)*eta(:,nn);
% end
% ismembertol(tt1, tt2, 1e-7)


for it = 2:niter
    disp(strcat('iter = ', num2str(it)));

    if mod(it,round(niter/10)) == 0
        disp(strcat('iter = ', num2str(it)));
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
    [phi tau] = sample_hypers(theta,phi,tau,priors.hypers);
    
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
    
    % (7) sample c_psi
    if samp_c_psi
        if samp_c_psi_marg
            llhd_func = @(c) llhd_K_psi_margin(yhat_tmp,c,theta,zeta,invSig_vec,x, r);
        else
            llhd_func = @(c) llhd_K_psi_cond(psi,c, x, r);
        end
        c_psi = c_samp_hmc(c_psi, llhd_func, priors.cPrior, step_psi);
        [K_psi, invK_psi, logdetK_psi] = GP_SE(c_psi, x, r, true);
    end
    c_psi_samp(it) = c_psi;
    
    % (8) sample zeta
    for ii=1:nrep_zeta % one can cycle through this sampling stage multiple times by adjusting num_iters
        zeta = sample_zeta(yhat_tmp,theta,eta,invSig_vec,zeta,invK_zeta,inds_y);
    end
    zeta_samp(:,:,:,it) = zeta;
    
    % (9) sample c_zeta
    if samp_c_zeta
        if samp_c_zeta_marg
            llhd_func = @(c) llhd_K_zeta_margin(yhat_tmp,c,theta,eta,invSig_vec, x, r);
        else
            % this is more robust: only use zeta with enough contribution
            meanAbs_theta = mean(abs(theta), 1);
            zeta_use = zeta(meanAbs_theta > quantile(meanAbs_theta, 0.5),:,:);
            llhd_func = @(c) llhd_K_zeta_cond(zeta_use,c, x, r);
        end
        
        c_zeta = c_samp_hmc(c_zeta, llhd_func, priors.cPrior, step_zeta);
        [K_zeta, invK_zeta, logdetK_zeta] = GP_SE(c_zeta, x, r, true);
    end
    c_zeta_samp(it) = c_zeta;
    
    if mod(it,100) == 0
        figure(1)
        subplot(2,1,1)
        plot(c_psi_samp(max(1, it - 200):it))
        title('psi')
        subplot(2,1,2)
        plot(c_zeta_samp(max(1, it - 200):it))
        title('zeta')
    end
    
end

out.theta_samp = theta_samp;
out.zeta_samp = zeta_samp;
out.psi_samp = psi_samp;
out.eta_samp = eta_samp;
out.invSig_vec_samp = invSig_vec_samp;
out.c_zeta_samp = c_zeta_samp;
out.c_psi_samp = c_psi_samp;

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
