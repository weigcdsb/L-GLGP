function [xy_samp, Y_samp, psi_samp, theta, zeta_samp, eps_samp, eta_samp,...
    psi_x_mu_samp_trace, psi_y_mu_samp_trace,...
    psi_sign_samp, psi_sig2_samp,...
    zeta_x_mu_samp_trace, zeta_y_mu_samp_trace, ...
    zeta_sign_samp, zeta_sig2_samp] =...
    simulateBoxTunnel(N, n_samp, k, L, ...
    s_s, t_h, t_l, n_mu_pos_samp, n_var_pos_samp, seed, sig_eps)


% to debug
% N = 50;
% n_samp = 10000;
% k = 2;
% L = 4;
% s_s = 3;
% t_h = 1;
% t_l = 2;
% n_mu_pos_samp = 5^2;
% n_var_pos_samp = 5^2;
% seed = 1;
% sig_eps = 0.1;

% nX = 100;
% nY = 100;
% y_mesh = zeros(nX*nY, 1);
% x_mesh_raw = linspace(-(t_l/2 + s_s), t_l/2 + s_s, nX)';
% x_mesh = repelem(x_mesh_raw, nX);
% for xx = 1:(nX*nY)
%     y_mesh(xx) = boxTunnel_samp(x_mesh(xx), s_s, t_h, t_l, 1);
% end
% scatter(x_mesh, y_mesh)


rng(seed)

% 1. sample x: 2 squares connected by a tunnel
xy_samp = zeros(n_samp, 2);
for ll = 1:n_samp
    x_tmp = unifrnd(-(t_l/2 + s_s),t_l/2 + s_s);
    xy_samp(ll,1) = x_tmp;
    xy_samp(ll,2) = boxTunnel_samp(x_tmp,s_s, t_h, t_l, 1);
end

% 2. sample latent for psi: for mean
psi_samp = zeros(n_samp, k);
psi_x_mu_samp_trace = zeros(n_mu_pos_samp, k);
psi_y_mu_samp_trace = zeros(n_mu_pos_samp, k);

psi_sign_samp = reshape(randsample([1, -1],...
    n_mu_pos_samp*k, true), n_mu_pos_samp, k);
psi_sig2_samp = unifrnd(1, 1.5, n_mu_pos_samp, k);

for kk = 1:k
    x_mu_samp = unifrnd(-(t_l/2 + s_s),t_l/2 + s_s, n_mu_pos_samp, 1);
    y_mu_samp = zeros(n_mu_pos_samp, 1);
    for mm = 1:n_mu_pos_samp
        y_mu_samp(mm) = boxTunnel_samp(x_mu_samp(mm),s_s, t_h, t_l, 1);
    end
    
    psi_x_mu_samp_trace(:,kk) = x_mu_samp;
    psi_y_mu_samp_trace(:,kk) = y_mu_samp;
    psi_samp(:,kk) = 5*mixMvnpdf(xy_samp, x_mu_samp, y_mu_samp,...
        psi_sign_samp(:,kk), psi_sig2_samp(:,kk));
    
end

% 3. sample latent for zeta: for variance
zeta_samp = zeros(L,k,n_samp);
zeta_x_mu_samp_trace = zeros(L,k,n_var_pos_samp);
zeta_y_mu_samp_trace = zeros(L,k,n_var_pos_samp);

zeta_sign_samp = reshape(randsample([1, -1],...
    n_var_pos_samp*k*L, true), L,k,n_var_pos_samp);
zeta_sig2_samp = unifrnd(1, 1.5, L,k,n_var_pos_samp);

for ll = 1:L
    for kk = 1:k
        x_mu_samp = unifrnd(-(t_l/2 + s_s),t_l/2 + s_s, n_var_pos_samp, 1);
        y_mu_samp = zeros(n_var_pos_samp, 1);
        for mm = 1:n_var_pos_samp
            y_mu_samp(mm) = boxTunnel_samp(x_mu_samp(mm),s_s, t_h, t_l, 1);
        end
        zeta_x_mu_samp_trace(ll,kk,:) = x_mu_samp;
        zeta_y_mu_samp_trace(ll,kk,:) = y_mu_samp;
        zeta_samp(ll,kk,:) = mixMvnpdf(xy_samp, x_mu_samp, y_mu_samp,...
            zeta_sign_samp(ll,kk,:), zeta_sig2_samp(ll,kk,:));
    end
end

% 4. sample theta
theta = randn(N, L);

% 5. combine them together
Y_samp = zeros(N,n_samp);
eps_samp = randn(N,n_samp)*sig_eps;
eta_samp = psi_samp + randn(n_samp, k);


% plotSurf_heat(xy_samp(:,1), xy_samp(:,2),...
%     eta_samp(:,1), s_s, t_h, t_l, false)
% colorbar()

% plotSurf_heat(xy_samp(:,1), xy_samp(:,2),...
%     psi_samp(:,1), s_s, t_h, t_l, false)
% colorbar()


for xx = 1:n_samp
    Y_samp(:,xx) = theta*zeta_samp(:,:,xx)*eta_samp(xx,:)' + eps_samp(:,xx);
end

% max(Y_samp,[],'all')
% min(Y_samp,[],'all')

end