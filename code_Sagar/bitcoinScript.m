% import training and testing data
baseDirData = '/Users/sagarsetru/Documents/Princeton/cos424/hw3/Assignment3_Bitcoin/data/';

tr = csvread([baseDirData,'txTripletsCounts.txt']);
te = csvread([baseDirData,'testTriplets.txt'] );

tr_A = zeros(max(tr(:,1));
