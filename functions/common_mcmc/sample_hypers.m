function [phi tau] = sample_hypers(theta,phi,tau,prior_hypers)

% to debug
% theta
% phi
% tau
% prior_hypers = prior_params.hypers;



[p L] = size(theta);

a1 = prior_hypers.a1;
a2 = prior_hypers.a2;
a_phi = prior_hypers.a_phi;
b_phi = prior_hypers.b_phi;

a = [a1 a2*ones(1,L-1)];
delta = exp([log(tau(1)) diff(log(tau))]);

for numIter = 1:50 % keep sample it 50 times
    
    phi = gamrnd((a_phi + 0.5)*ones(p,L),1) ./ (b_phi + 0.5*tau(ones(1,p),:).*(theta.^2));

    sum_phi_theta = sum(phi.*(theta.^2),1);
    for hh=1:L
        tau_hh = exp(cumsum(log(delta))).*[zeros(1,hh-1) ones(1,L-hh+1)./delta(hh)];
        delta(hh) = gamrnd(a(hh) + 0.5*p*(L-hh+1),1) ./ (1 + 0.5*sum(tau_hh.*sum_phi_theta));
    end
    
    tau = exp(cumsum(log(delta)));
    
end

return;