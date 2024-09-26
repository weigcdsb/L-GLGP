function K_ind = sample_K_cond_zeta(zeta,prior_params)

[L k N] = size(zeta);
c_prior = prior_params.c_prior;
invK = prior_params.invK;

zeta_tmp = reshape(zeta,[L*k N])';

Pk = zeros(1,length(c_prior));
for ii=1:length(c_prior)
    Pk(ii) = - sum(diag(zeta_tmp'*(invK(:,:,ii)*zeta_tmp)));
end
Pk = Pk - 0.5*(L*k)*prior_params.logdetK + log(prior_params.c_prior);
Pk = cumsum(exp(Pk-max(Pk)));
K_ind = 1 + sum(Pk(end)*rand(1) > Pk);

return;