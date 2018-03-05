features = zeros(50000,8100);

a = imgs(19,:);
b = reshape(a,[28 28]);
% c = rgb2gray(b);
c = b;



[h v] = extractHOGFeatures(c,'CellSize', [4 4]);