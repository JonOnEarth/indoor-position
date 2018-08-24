% This script is calculating 2D mean square error on position estimation when
% apply long term dataset from UJI
% assuming the floor detection is 100%
clear;
train_rss = csvread('longterm_train_rss.csv');
train_rss(train_rss==100)=-103;
train_location = csvread('longterm_train_coords.csv');
test_rss = csvread('longterm_test_rss.csv');
test_rss(test_rss==100)=-103;
test_location = csvread('longterm_test_coords.csv');

%% third floor
train_rss1 = train_rss(train_location(:,3)==3,:);
train_coord1 = train_location(train_location(:,3)==3,1:2);
test_rss1 = test_rss(test_location(:,3)==3,:);
test_coord1 = test_location(test_location(:,3)==3,:);

% fiveth floor
train_rss2 = train_rss(train_location(:,3)==5,:);
train_coord2 = train_location(train_location(:,3)==5,1:2);
test_rss2 = test_rss(test_location(:,3)==5,:);
test_coord2 = test_location(test_location(:,3)==5,:);

%%
predict1=WKNN(train_rss1,train_coord1,test_rss1,3,'powed','sorensen');
predict2=WKNN(train_rss2,train_coord2,test_rss2,3,'powed','sorensen');

%%
[error1,accuracy1]=calculateAccuracy(predict1,test_coord1(:,1:2)); % 3rd floor 2D error
[error2,accuracy2]=calculateAccuracy(predict2,test_coord2(:,1:2)); % 5th floor 2D error
accuracy=(accuracy1+accuracy2)/2;% total 2D average error