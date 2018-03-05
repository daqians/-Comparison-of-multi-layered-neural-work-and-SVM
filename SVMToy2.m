classdef SVMToy2
properties
    losses
    confmats
    time
    X
    Y
    sz
    len
    folds
end
methods
    function obj = runsvm(obj, data,labels,folds,ker)
        obj.X = data; 
        obj.Y = labels;
        obj.folds = folds;
        obj.sz = size(labels,1);
        obj.len = fix(obj.sz/folds);

        loss = zeros(folds,1);
        confmat = zeros(2,2,folds);
        tm = zeros(folds,1);

        start = zeros(folds,1);
        stop = zeros(folds,1);
        for i = 0:(folds-1)
            start(i+1) = i*obj.len+1;
            stop(i+1) = (i+1)*obj.len;
        end
        stop(folds) = obj.sz;

        idx = randperm(obj.sz);

        for i = 1:folds
            id = idx(start(i):stop(i));

            dtest = data(id,:);
            dtrain = data;
            dtrain(id,:) = [];

            ltest = labels(id,:);
            ltrain = labels;
            ltrain(id,:) = [];

            ts = tic;
            if ker == 1
                svm = fitcsvm(dtrain,ltrain,'KernelFunction','rbf', ...
                    'KernelScale','auto','standardize',true);    
            elseif ker == 2
                svm = fitcsvm(dtrain,ltrain,'KernelFunction','linear', ...
                    'KernelScale','auto','standardize',true); 
            elseif ker == 3
                svm = fitcsvm(dtrain,ltrain, ...
                    'KernelFunction','polynomial', ...
                    'PolynomialOrder',2, ...
                    'KernelScale','auto','standardize',true);
            elseif ker == 4
                svm = fitcsvm(dtrain,ltrain, ...
                    'KernelFunction','polynomial', ...
                    'PolynomialOrder',3, ...
                    'KernelScale','auto','standardize',true);
            elseif ker == 5
                svm = fitcsvm(dtrain,ltrain, ...
                    'KernelFunction','polynomial', ...
                    'PolynomialOrder',4, ...
                    'KernelScale','auto','standardize',true);
            elseif ker == 6
                svm = fitcsvm(dtrain,ltrain, ...
                    'KernelFunction','polynomial', ...
                    'PolynomialOrder',5, ...
                    'KernelScale','auto','standardize',true);
            elseif ker == 7
                svm = fitcsvm(dtrain,ltrain, ...
                    'KernelFunction','polynomial', ...
                    'PolynomialOrder',6, ...
                    'KernelScale','auto','standardize',true);
            else
                svm = fitcsvm(dtrain,ltrain,'KernelFunction','rbf', ...
                    'KernelScale','auto','standardize',true); 
            end
            te = toc(ts);

            loss(i) = svm.loss(dtest,ltest);
            tm(i) = te;

            pred = svm.predict(dtest);
            confmat(:,:,i) = confusionmat(ltest,pred);
        end
        
        obj.time = tm;
        obj.losses = loss;
        obj.confmats = confmat;
    end
end
end