function res = mixMvnpdf(xy, x_samp, y_samp, sign_samp, sig2_samp)
% make this to be deterministic, so that all the random effect can be
% stored...

n_samp_tmp = length(x_samp);
% xy = [x_mesh, y_mesh];

%  to debug
% sign_samp = randsample([1, -1],n_samp_tmp, true);
% sig2_samp = unifrnd(0.05, 0.3, n_samp_tmp, 1);
res = sign_samp(1)*mvnpdf(xy,[x_samp(1),...
    y_samp(1)], sig2_samp(1)*eye(2));
for kk = 2:n_samp_tmp
    res = res + sign_samp(kk)*mvnpdf(xy,[x_samp(kk), y_samp(kk)],...
        sig2_samp(kk)*eye(2));
    
end

