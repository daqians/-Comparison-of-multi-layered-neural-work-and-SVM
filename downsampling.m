function [ndata,nlabels] = downsampling(data,labels)

[sz,wid] = size(data);

count = sum(labels);

data1 = zeros(count,wid);
data2 = zeros(sz-count,wid);

idx = 1;
for i = 1:sz
    if labels(i,:) == 1
        data1(idx,:) = data(i,:);
        idx = idx+1;
    end
end

for i = 1:count
   idx = randi(sz);
   while labels(idx,:) == 1
       idx = randi(sz);
   end
   data2(i,:) = data(idx,:);
end

out = [data1; data2];
outl = [ones(count,1); zeros(count,1)];

nidx = randperm(size(outl,1));

ndata = out(nidx,:);
nlabels = outl(nidx,:);

end