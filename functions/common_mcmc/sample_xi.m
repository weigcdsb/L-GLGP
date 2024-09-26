function xi = sample_xi(y,theta,invSig_vec,zeta,psi,inds_y)

% Sample latent factors eta_i using standard Gaussian identities based on
% the fact that:
%
% y_i = (theta*zeta)*eta_i + eps_i,   eps_i \sim N(0,Sigma_0),   eta_i \sim N(0,I) 
%
% and using the information form of the Gaussian likelihood and prior.

[p N] = size(y);
[L k] = size(zeta(:,:,1));

xi = zeros(k,N);
for nn=1:N
    theta_zeta_n = theta*zeta(:,:,nn);
    y_tilde_n = y(:,nn)-theta_zeta_n*psi(:,nn);
    
    invSigMat = invSig_vec.*inds_y(:,nn)';
    invSigMat = invSigMat(ones(k,1),:);
    zeta_theta_invSig = theta_zeta_n'.*invSigMat;  % zeta_theta_invSig = zeta(:,:,nn)'*(theta'*invSig);

    cholSig_xi_n_trans = chol(eye(k) + zeta_theta_invSig*theta_zeta_n) \ eye(k);  % Sig_eta_n = inv(eye(k) + zeta_theta_invSig*theta_zeta_nn);   
    m_xi_n = cholSig_xi_n_trans*(cholSig_xi_n_trans'*(zeta_theta_invSig*y_tilde_n));  % m_eta_n = Sig_eta_n*(zeta_theta_invSig*y(:,nn));
    
    xi(:,nn) = m_xi_n + cholSig_xi_n_trans*randn(k,1); % eta(:,nn) = m_eta_n + chol(Sig_eta_n)'*randn(k,1);
end

return;