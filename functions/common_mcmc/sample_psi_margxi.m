function psi = sample_psi_margxi(y,theta,invSig_vec,zeta,psi,invK,inds_y)

[p,L] = size(theta);
[k,N] = size(psi);
Sigma_0 = diag(1./invSig_vec);

% We derive the sequential sampling of the zeta_{ll,kk} by reformulating
% the full regression problem as one that is solely in terms of
% zeta_{ll,kk} having conditioned on the other latent GP functions:
%
% y_i = eta(i,m)*theta(:,ll)*zeta_{ll,kk}(x_i) + tilde(eps)_i,
%

% Initialize the structure for holding the conditional mean for additive
% Gaussian noise term tilde(eps)_i and add values based on previous zeta:
mu_tot = zeros(p,N);
Omega = zeros(p,k,N);
OmegaInvOmegaOmegaSigma0 = zeros(k,p,N);
for nn=1:N
    Omega(inds_y(:,nn),:,nn) = theta(inds_y(:,nn),:)*zeta(:,:,nn);
    temp = (Omega(inds_y(:,nn),:,nn)*Omega(inds_y(:,nn),:,nn)'...
        + Sigma_0(inds_y(:,nn),inds_y(:,nn))) \ eye(sum(inds_y(:,nn)));
    OmegaInvOmegaOmegaSigma0(:,inds_y(:,nn),nn) = Omega(inds_y(:,nn),:,nn)'*temp;
    mu_tot(:,nn) = Omega(:,:,nn)*psi(:,nn);  % terms will be 0 where inds_y(:,nn)=0
end

if sum(sum(mu_tot))==0 % if this is a call to initialize psi
    numTotIters = 50;
else
    numTotIters = 5;
end

for numIter = 1:numTotIters
    
    for kk=randperm(k);  % create random ordering for kk in sampling zeta_{ll,kk} 
        
        Omega_kk = squeeze(Omega(:,kk,:));
        psi_kk = psi(kk,:);
        mu_tot = mu_tot - Omega_kk.*psi_kk(ones(p,1),:);
        
        theta_k = diag(squeeze(OmegaInvOmegaOmegaSigma0(kk,:,:))'*(y-mu_tot));
        Ak_invSig_Ak = diag(squeeze(OmegaInvOmegaOmegaSigma0(kk,:,:))'*Omega_kk);
        
        cholSig_k_trans = chol(invK + diag(Ak_invSig_Ak)) \ eye(N);
        
        % Transform information parameters:
        m_k = cholSig_k_trans*(cholSig_k_trans'*theta_k);  % m_lk = Sig_lk*theta_lk;
        
        % Sample zeta_{ll,kk} from posterior Gaussian:
        psi(kk,:) = m_k + cholSig_k_trans*randn(N,1);  % zeta(ll,kk,:) = m_lk + chol(Sig_lk)'*randn(N,1);
        
        psi_kk = psi(kk,:);
        mu_tot = mu_tot + Omega_kk.*psi_kk(ones(p,1),:);

    end
    
end

return;