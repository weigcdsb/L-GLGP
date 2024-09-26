function c_samp = c_samp_hmc(c_now, llhd_func, cPrior, stepsize)

% c_now = c_psi;
% cPrior = priors.cPrior;
% stepsize = step_psi;
% c_prior
% log(c) ~ N(0, 1)
% cPrior.lmean = 0;
% cPrior.lvar = 1;
% llhd_func = @(c) llhd_K_psi_margin(y,c,theta,zeta,invSig_vec,x, r);

% logc = log(c_now)

logpdf = @(logc) llhd_func(exp(logc)) +...
    normpdfln(logc, cPrior.lmean, cPrior.lvar);

smp = hmcSampler(logpdf, log(c_now), 'NumSteps',1,...
    'UseNumericalGradient',true,'CheckGradient',0,'StepSize',stepsize);
logc_samp = drawSamples(smp, 'Burnin', 0,...
    'NumSamples',1, 'StartPoint',log(c_now));
c_samp = exp(logc_samp);


end