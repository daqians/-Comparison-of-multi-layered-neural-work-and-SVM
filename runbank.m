function [loss confmat] = run(data,labels,folds)

sz = size(labels,1);
len = fix(sz/folds);

loss = zeros(folds,1);
confmat = zeros(2,2,folds);

start = zeros(folds,1);
stop = zeros(folds,1);
for i = 0:(folds-1)
    start(i+1) = i*len+1;
    stop(i+1) = (i+1)*len;
end
stop(folds) = sz;

idx = randperm(sz);

for i = 1:folds
    id = idx(start(i):stop(i));
    
    dtest = data(id,:);
    dtrain = data;
    dtrain(id,:) = [];
    
    ltest = labels(id,:);
    ltrain = labels;
    ltrain(id,:) = [];
    
    tic;
    svm = fitcsvm(dtrain,ltrain);    
    toc;
    
    loss(i) = svm.loss(dtest,ltest);
    
    pred = svm.predict(dtest);
    confmat(:,:,i) = confusionmat(ltest,pred);
end


end

