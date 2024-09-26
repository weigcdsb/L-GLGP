function invSig_vec = sample_sig(y,theta,eta,zeta,prior_sig,inds_y)

% to debug
% y
% theta
% eta
% zeta
% prior_sig = prior_params.sig;
% inds_y


[p N] = size(y);

a_sig = prior_sig.a_sig;
b_sig = prior_sig.b_sig;

inds_vec = [1:N];

invSig_vec = zeros(1,p);
for pp = 1:p
    sq_err = 0;
    for nn=inds_vec(inds_y(pp,:))
        sq_err = sq_err + (y(pp,nn) - theta(pp,:)*zeta(:,:,nn)*eta(:,nn))^2;
    end
    invSig_vec(pp) = gamrnd(a_sig + 0.5*sum(inds_y(pp,:)),1) / (b_sig + 0.5*sq_err);
end

return;