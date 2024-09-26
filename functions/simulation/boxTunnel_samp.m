function y = boxTunnel_samp(x, s_s, t_h, t_l, nY)

% x_abs_s_bound = t_l/2 + s_s;
x_abs_t_bound = t_l/2;

if (abs(x) < x_abs_t_bound)
    y = unifrnd(-t_h/2,t_h/2, nY, 1);
else
    y = unifrnd(-s_s/2, s_s/2, nY, 1);
    

end