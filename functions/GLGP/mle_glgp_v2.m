function [eps, k, t] = mle_glgp_v2(iterMax, samps, x,sig,useCorr,...
    eps0, k0, t0, tol, sig2, varargin)

% iterMax, samps, x,sig,true, eps0, k0, t0, tol, tlb
% useCorr = true;

% to debug
% iterMax = 1000;
% samps = [];
% for it = 150:200
%     samps = [samps;out_seGP.psi_samp(:,:,it)];
% end
% sig = 1e-2;
% useCorr = true;
% eps0 = 0.1;
% k0 = 20;
% t0 = 1;
% tol = 1e-4;
% tlb = 0.01;
% sig2 = 0;

k_stepSize = 2;
tlb = 1e-3;
epslb = 1e-3;
if (~isempty(varargin))
    c = 1 ;
    while c <= length(varargin)
        switch varargin{c}
            case {'k_stepSize'}
                k_stepSize = varargin{c+1};
            case {'tlb'}
                tlb = varargin{c+1};
            case {'epslb'}
                epslb = varargin{c+1};    
        end % switch
        c = c + 2;
    end % for
end % if

n_samp = size(samps, 2);

eps_tmp = eps0;
k_tmp = k0;
t_tmp = t0;

for iter = 1:iterMax
    
    % A. grid search for k
    k_grid = max(1, k_tmp - 40):k_stepSize:min(k_tmp + 40, n_samp);
    n_k = size(k_grid, 2);
    llhd_vec = zeros(n_k, 1);
    warning('off')
    for kk = 1:n_k
        llhd_vec(kk) = llhd_K_glgp(samps, x,k_grid(kk),...
            eps_tmp,t_tmp,sig,useCorr, sig2);
    end
    warning('on')
    [~,kid] = max(llhd_vec);
    k_new = k_grid(kid(1));
    
    % B. MLE for eps & t
    target = @(log_epslb_tlb) -llhd_K_glgp(samps, x,...
        k_new, exp(log_epslb_tlb(1)) + epslb,...
        exp(log_epslb_tlb(2)) + tlb,sig,useCorr, sig2);
    
    log_epslb_tlb_0 = [log(eps_tmp - epslb) log(t_tmp - tlb)];
    try
        log_epslb_tlb_tmp = fminunc(target, log_epslb_tlb_0,...
            optimoptions(@fminunc,'Display', 'off'));
    catch
        log_epslb_tlb_tmp = fminsearch(target, log_epslb_tlb_0);
    end
    
    eps_new = exp(log_epslb_tlb_tmp(1)) + epslb;
    t_new = exp(log_epslb_tlb_tmp(2)) + tlb;
    
    if sum([eps_new, t_new] == [eps_tmp, t_tmp]) == 2
        log_epslb_tlb_tmp = fminsearch(target, log_epslb_tlb_0);
        eps_new = exp(log_epslb_tlb_tmp(1)) + epslb;
        t_new = exp(log_epslb_tlb_tmp(2)) + tlb;
    end
    
    disp([k_new, eps_new, t_new])
    if(norm([eps_new, t_new] - [eps_tmp, t_tmp]) < tol) && (iter > 5)
        break;
    end
    
    k_tmp = k_new;
    eps_tmp = eps_new;
    t_tmp = t_new;
end

k = k_new;
eps = eps_new;
t = t_new;

end