
sz = 100;
ls = zeros(sz,1);
tm = zeros(sz,1);

for i = 1:sz
    [data, labels] = downsampling(cdata,clabels);
    svm = SVMToy2().runsvm(data,labels,10,1);

    ls(i,:) = mean(svm.losses);
    tm(i,:) = sum(svm.time);
end

acc = 1 - mean(ls)
time = mean(tm)