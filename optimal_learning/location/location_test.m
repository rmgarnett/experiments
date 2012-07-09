num_points   = 50;
item_cost    = 1 / 10;
length_scale = 1 / 10;

x = linspace(0, 1, num_points);
[xx, yy] = meshgrid(x, x);

x = [xx(:), yy(:)];

K = covSEiso([log(length_scale), log(1)], x);

pdf = exp(mvnrnd(zeros(num_points^2, 1), K));
pdf = pdf / sum(pdf);
pdf = reshape(pdf, [num_points, num_points]);

clear('x', 'xx', 'yy', 'K');