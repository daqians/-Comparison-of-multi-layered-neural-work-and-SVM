

svm = templateSVM('KernelFunction','rbf',...
    'Standardize',true,'KernelScale','auto');

ts = tic;
mdl = fitcecoc(features1s,labels1,'learners',svm);
te = toc(ts);

mdl.loss(featurest,labelst);