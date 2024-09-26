function zeta = sample_zeta(y,theta,eta,invSig_vec,zeta,invK,inds_y)

% to debug
% y
% theta
% eta
% invSig_vec
% zeta
% invK = prior_params.K.invK(:,:,K_ind);
% inds_y


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% The posterior of all latent GPs can be analytically computed in      %%
%%% closed form.  Specifically, it is an NxkxL dimensional Gaussian.     %%
%%% However, for most problems, it is infeasible to sample directly from %%
%%% this joint posterior because of the dimensionality of the Gaussian   %%
%%% parameterization.  Below is the code for sampling from this joint    %%
%%% posterior for reference:                                             %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% [p N] = size(y);
% L = size(theta,2);
% k = size(eta,1);
% 
% invK = prior_params.invK;
% 
% AinvSig = zeros(N*k*L,N*p);
% AinvSigA = zeros(N*k*L);
% AinvSig = sparse(AinvSig);
% AinvSigA = sparse(AinvSigA);
% for nn=1:N
%     tmp1 = kron(theta,eta(:,nn)');
%     tmp2 = tmp1'*invSig;
%     AinvSig((nn-1)*L*k+1:nn*L*k,(nn-1)*p+1:nn*p) = tmp2;
%     AinvSigA((nn-1)*L*k+1:nn*L*k,(nn-1)*L*k+1:nn*L*k) = tmp2*tmp1;
% end
% 
% invbigK = kron(invK,eye(L*k));  % inv(kron(K,eye(L*k)));
% invbigK = sparse(invbigK);
% 
% Sig = (invbigK + AinvSigA) \ eye(N*k*L);
% m = Sig*(AinvSig*y(:));
% zeta_vec = m + chol(Sig)'*randn(N*k*L,1);
% 
% zeta = zeros(L,k,N);
% for ii=1:N
%     zeta_tmp = zeta_vec((ii-1)*k*L + 1: ii*k*L);
%     zeta(:,:,ii) = reshape(zeta_tmp,[k L])';
% end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Instead, we sample the latent GP functions as follows.  For          %%
%%% initialization we sequentially walk through each row sampling        %%
%%% zeta_{ll,kk} assuming zeta_{ll+1:L,unsampled kk for row ll}=0.       %%
%%% We move in this order since in expectation the importance of each    %%
%%% zeta_{ll,kk} decreases with increasing ll due to the sparsity        %%
%%% structure of the weightings matrix theta.  We then reloop through    %%
%%% sampling the zeta_{ll,kk} multiple times in order to improve the     %%
%%% mixing rate given the other currently sampled params.  The other     %%
%%% calls to resampling zeta_{ll,kk} operates in exactly the same way,   %%
%%% but based on the past sample of zeta.                                %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[p L] = size(theta);
[k N] = size(eta);

% We derive the sequential sampling of the zeta_{ll,kk} by reformulating
% the full regression problem as one that is solely in terms of
% zeta_{ll,kk} having conditioned on the other latent GP functions:
%
% y_i = eta(i,m)*theta(:,ll)*zeta_{ll,kk}(x_i) + tilde(eps)_i,
%

% Initialize the structure for holding the conditional mean for additive 
% Gaussian noise term tilde(eps)_i and add values based on previous zeta:
mu_tot = zeros(p,N);
error_nn = zeros(L,N);
for nn=1:N
    mu_tot(:,nn) = theta*zeta(:,:,nn)*eta(:,nn);
    % Store the amount that will be added, but shouldn't be because of
    % missing observations:
    error_nn(:,nn) = (theta.^2)'*(invSig_vec'.*(1-inds_y(:,nn)));
end 

%if sum(zeta(:))==0
%    numiter = 50;
%else
    numiter = 1;
%end
    
for nn=1:numiter
    for ll=1:L  % walk through each row of zeta matrix sequentially
        theta_ll = theta(:,ll);
        for kk=randperm(k);  % create random ordering for kk in sampling zeta_{ll,kk}
            eta_kk = eta(kk,:);
            zeta_ll_kk = squeeze(zeta(ll,kk,:))';
            mu_tot = mu_tot - theta_ll(:,ones(1,N)).*eta_kk(ones(p,1),:).*zeta_ll_kk(ones(p,1),:);
            
            % Using standard Gaussian identities, form posterior of
            % zeta_{ll,kk} using information form of Gaussian prior and likelihood:
            A_lk_invSig_A_lk = (eta(kk,:).^2).*((theta(:,ll).^2)'*invSig_vec'-error_nn(ll,:));
            
            theta_tmp = theta(:,ll)'.*invSig_vec;
            ytilde = (y - mu_tot).*inds_y;  % normalize data by subtracting mean of tilde(eps)
            theta_lk = eta(kk,:)'.*(theta_tmp*ytilde)'; % theta_lk = eta(kk,:)'.*diag(theta_tmp(ones(1,N),:)*ytilde);
            
            % Transform information parameters:
            cholSig_lk_trans = chol(invK + diag(A_lk_invSig_A_lk)) \ eye(N);  % Sig_lk = inv(invK + A_lk_invSig_A_lk);
            m_lk = cholSig_lk_trans*(cholSig_lk_trans'*theta_lk);  % m_lk = Sig_lk*theta_lk;
            
            % Sample zeta_{ll,kk} from posterior Gaussian:
            zeta(ll,kk,:) = m_lk + cholSig_lk_trans*randn(N,1);  % zeta(ll,kk,:) = m_lk + chol(Sig_lk)'*randn(N,1);
            
            zeta_ll_kk = squeeze(zeta(ll,kk,:))';
            mu_tot = mu_tot + theta_ll(:,ones(1,N)).*eta_kk(ones(p,1),:).*zeta_ll_kk(ones(p,1),:);
        end
    end
end

return;