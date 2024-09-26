function llhd = llhd_K_glgp(samps, X,k,eps,t,sig,useCorr, sig2)

% X = x;
% k = k_grid(kk);
% eps = eps_grid(ee);
% t = t_tmp;

% to debug
% psi_samps = [];
% for it = 150:200
%     psi_samps = [psi_samps;out_seGP.psi_samp(:,:,it)];
% end
% X = x';
% k = 50;
% eps = 0.1;
% t = 1;
% sig = 1e-2;
% useCorr = true;

% 1. calculate the kernel function
N = size(samps, 2);
K = GLGP_cov(X,k,eps,t,sig,useCorr, sig2);

% 2. calculate the likelihood
llhd = sum(logmvnpdf(samps,zeros(1,N),K));

end