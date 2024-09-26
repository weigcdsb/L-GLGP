function [eps, k, t] = mle_glgp(iterMax, samps, x,sig,useCorr,...
    eps0, k0, t0, tol, tlb, sig2, varargin)

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

k_stepSize = 2;
if (~isempty(varargin))
    c = 1 ;
    while c <= length(varargin)
        switch varargin{c}
            case {'k_stepSize'}
                k_stepSize = varargin{c+1};
        end % switch
        c = c + 2;
    end % for
end % if

N = size(samps, 2);
n_eps = 20;

eps_tmp = eps0;
k_tmp = k0;
t_tmp = t0;

for iter = 1:iterMax
    
    % A. grid search for eps & k
    eps_grid = linspace(max(1e-4, eps_tmp - 0.5),...
        min(eps_tmp + 0.5, 1), n_eps);
    k_grid = max(1, k_tmp - 40):k_stepSize:min(k_tmp + 40, N);
    n_k = size(k_grid, 2);
    
    llhd_mat = zeros(n_eps, n_k);
    warning('off')
    for ee = 1:n_eps
        for kk = 1:n_k
            try
                llhd_mat(ee,kk) =...
                    llhd_K_glgp(samps, x,k_grid(kk),...
                    eps_grid(ee),t_tmp,sig,useCorr, sig2);
            catch
                llhd_mat(ee,kk) = -Inf;
            end
        end
    end
    warning('on')
    
    ll_max = max(llhd_mat,[],"all");
    [rid,cid]=find(llhd_mat==ll_max);
    eps_new = eps_grid(rid(1));
    k_new = k_grid(cid(1));
    
    % B. MLE for t
    target = @(logtlb) -llhd_K_glgp(samps, x,...
        k_new,eps_new,exp(logtlb) + tlb,sig,useCorr, sig2);
    logtlb_tmp = fminunc(target,log(t_tmp - tlb),...
        optimoptions(@fminunc,'Display', 'off'));
    t_new = exp(logtlb_tmp) + tlb;
    if t_new == t_tmp
        logtlb_tmp = fminsearch(target,log(t_tmp -tlb));
        t_new = exp(logtlb_tmp) + tlb;
    end
    
    disp([k_new, eps_new, t_new])
    if(abs(t_new - t_tmp) < tol)
        break;
    end
    
    eps_tmp = eps_new;
    k_tmp = k_new;
    t_tmp = t_new;
end

eps = eps_new;
k = k_new;
t = t_new;

end