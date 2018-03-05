function features = extractFeatures(data)
cell = [2 2];
shape = [32 32 3];
fsz = 8100;

[rows cols] = size(data);
features = zeros(rows,fsz);

for i = 1:rows
    r = data(i,:);
    img = reshape(r,shape);
    imgg = rgb2gray(img);
    
    f = extractHOGFeatures(imgg,'CellSize', cell);
    features(i,:) = f;
end

end