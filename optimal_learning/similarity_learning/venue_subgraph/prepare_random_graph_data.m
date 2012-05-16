num_classes = 2;

num_stars = 10;
max_size = 20;

sizes = 1 + randi(max_size, num_stars, 1);
star_edges = cumsum(sizes);

num_nodes = sum(sizes);

data = sparse(num_nodes, num_nodes);
responses = zeros(num_nodes, 1);

count = 1;
for i = 1:num_stars
  data(count:star_edges(i), count:star_edges(i)) = 1;
  
  responses(count:star_edges(i)) = randi(num_classes);
  
  count = count + sizes(i);
end

num_connections = 1000;
for i = 1:num_connections
  pair = randi(num_stars, 2);
  data(star_edges(pair(1)), star_edges(pair(2))) = 1;
end

data = min(data, data');
data = data - diag(diag(data));