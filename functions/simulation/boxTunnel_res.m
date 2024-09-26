function res = boxTunnel_res(x,y,x_samp, y_samp)

% to debug
% x_samp = x_mu_samp;
% y_samp = y_mu_samp;

res_fun = @(xy) mixMvnpdf(xy, x_samp, y_samp);
res = res_fun([x, y]);

% plot density
% nX = 1000;
% nY = 200;
% y_mesh = zeros(nX*nY, 1);
% x_mesh_raw = linspace(-(t_l/2 + s_s), t_l/2 + s_s, nX)';
% x_mesh = repelem(x_mesh_raw, nY);
% for xx = 1:(nX*nY)
%     y_mesh(xx) = boxTunnel_samp(x_mesh(xx), s_s, t_h, t_l, 1);
% end
% z_mesh = res_fun([x_mesh, y_mesh]);
% scatter3(x_mesh, y_mesh, z_mesh, 1, z_mesh)






end