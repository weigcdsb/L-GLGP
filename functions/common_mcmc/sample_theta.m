function theta = sample_theta(y,eta,invSig_vec,zeta,phi,tau,inds_y)

% to debug
% y
% eta
% invSig_vec
% zeta
% phi
% tau
% inds_y



[p N] = size(y);
L = size(zeta,1);
theta = zeros(p,L);

eta_tilde = zeros(L,N);
for nn=1:N
    eta_tilde(:,nn) = zeta(:,:,nn)*eta(:,nn);
end
eta_tilde = eta_tilde';

for pp=1:p
    inds_y_p = inds_y(pp,:)';
    eta_tilde_p = eta_tilde.*inds_y_p(:,ones(1,L));
    chol_Sig_theta_p_trans = chol(diag(phi(pp,:).*tau) + invSig_vec(pp)*(eta_tilde_p'*eta_tilde_p)) \ eye(L);
    m_theta_p = invSig_vec(pp)*(chol_Sig_theta_p_trans*chol_Sig_theta_p_trans')*(eta_tilde_p'*y(pp,:)');
    theta(pp,:) = m_theta_p + chol_Sig_theta_p_trans*randn(L,1);
end

return;