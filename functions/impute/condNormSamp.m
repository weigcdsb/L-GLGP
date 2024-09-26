function [samp_q, mu_q, Sig_q] = condNormSamp(x_q, samp_obs, Sig_all)

% to debug...
% x_q = xy_q;
% % x_obs = xy_samp;
% samp_obs = fitRes.psi_samp(:,:,it)';
% Sig_all = Sig_psi_q;

k = size(samp_obs, 2);
n_q_samp = size(x_q, 1);
% n_obs_samp = size(x_obs, 1);

Sig_11 = Sig_all(1:n_q_samp, 1:n_q_samp);
Sig_12 = Sig_all(1:n_q_samp, (n_q_samp+1):end);
Sig_21 = Sig_all((n_q_samp+1):end, 1:n_q_samp);
Sig_22 = Sig_all((n_q_samp+1):end, (n_q_samp+1):end);


Sig_q = Sig_11 - (Sig_12/Sig_22)*Sig_21;
mu_q = Sig_12/Sig_22*samp_obs;



% draw samples...
R = chol(Sig_q,'lower');
z = randn(n_q_samp,k);
samp_q = mu_q + R*z;


end