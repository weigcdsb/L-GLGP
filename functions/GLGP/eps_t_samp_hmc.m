function eps_t_samp = eps_t_samp_hmc(eps_t_now, llhd_func, eps_t_prior, stepsize)

% to debug
% eps_t_now = [glgp_psi.eps, glgp_psi.t];
% llhd_func 
% eps_t_prior.lmean = [0,0]';
% eps_t_prior.lvar = eye(2);
% stepsize = 0.1;

logpdf = @(log_epst) llhd_func(exp(log_epst)) +...
    normpdfln(log_epst, eps_t_prior.lmean, eps_t_prior.lvar);
smp = hmcSampler(logpdf, log(eps_t_now), 'NumSteps',1,...
    'UseNumericalGradient',true,'CheckGradient',0,'StepSize',stepsize);
log_epst_samp = drawSamples(smp, 'Burnin', 0,...
    'NumSamples',1, 'StartPoint',log(eps_t_now));
eps_t_samp = exp(log_epst_samp);


end