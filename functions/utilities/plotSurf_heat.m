function plotSurf_heat(x_in, y_in, z_in, s_s, t_h, t_l, plotsurf, step_size)

% to debug
% x_in = xy_samp(:,1);
% y_in = xy_samp(:,2);
% z_in = Y_samp(1,:);

x_in = x_in(:);
y_in = y_in(:);
z_in = z_in(:);

[xq,yq] = meshgrid(-(s_s+t_l/2):step_size:(s_s+t_l/2),...
    -(s_s/2):step_size:(s_s/2));
x_more = (-t_l/2 + step_size):step_size:(t_l/2 - step_size);
y_more_pos = (t_h/2 + step_size):step_size:s_s/2;
y_more_neg = (-s_s/2):step_size:(-t_h/2 - step_size);
y_more = [y_more_neg, y_more_pos];

x_add = repelem(x_more, size(y_more,2))';
y_add = repmat(y_more', size(x_more,2), 1);
z_add = zeros(length(x_add), 1)*nan;

x_plt = [x_in;x_add];
y_plt = [y_in;y_add];
z_plt = [z_in;z_add];

vq = griddata(x_plt,y_plt,z_plt,xq,yq);  %(x,y,v) being your original data for plotting points
if plotsurf
    surf(xq,yq,vq, 'EdgeColor', 'none')
else
    imagesc(-(s_s+t_l/2):.05:(s_s+t_l/2),-(s_s/2):.05:(s_s/2),vq,...
    'AlphaData', ~isnan(vq))
    set(gca,'YDir','normal')
end


end