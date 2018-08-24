% This script is doing 2D mean square error on position estimation when 
% apply short term dataset from UJI
%
% for 2D mean square error, we assume the floor detection returns perfect
% result that we extract each floor data before using WKNN and calculate
% the error for each floor

clear;
train_rss = csvread('Training_rss_21Aug17.csv');
train_rss(train_rss==100)=-103;
train_location = csvread('Training_coordinates_21Aug17.csv');
test_rss = csvread('Test_rss_21Aug17.csv');
test_rss(test_rss==100)=-103;
test_location = csvread('Test_coordinates_21Aug17.csv');

%% first floor data
train_rss1 = train_rss(floor(train_location(:,3))==0,:);
train_coord1 = train_location(floor(train_location(:,3))==0,1:2);
test_rss1 = test_rss(floor(test_location(:,3))==0,:);
test_coord1 = test_location(floor(test_location(:,3))==0,:);
% second floor data
train_rss2 = train_rss(floor(train_location(:,3))==3,:);
train_coord2 = train_location(floor(train_location(:,3))==3,1:2);
test_rss2 = test_rss(floor(test_location(:,3))==3,:);
test_coord2 = test_location(floor(test_location(:,3))==3,:);
% third floor data
train_rss3 = train_rss(floor(train_location(:,3))==7,:);
train_coord3 = train_location(floor(train_location(:,3))==7,1:2);
test_rss3 = test_rss(floor(test_location(:,3))==7,:);
test_coord3 = test_location(floor(test_location(:,3))==7,:);
% forth floor data
train_rss4 = train_rss(floor(train_location(:,3))==11,:);
train_coord4 = train_location(floor(train_location(:,3))==11,1:2);
test_rss4 = test_rss(floor(test_location(:,3))==11,:);
test_coord4 = test_location(floor(test_location(:,3))==11,:);
% fifth floor data
train_rss5 = train_rss(floor(train_location(:,3))==14,:);
train_coord5 = train_location(floor(train_location(:,3))==14,1:2);
test_rss5 = test_rss(floor(test_location(:,3))==14,:);
test_coord5 = test_location(floor(test_location(:,3))==14,:);

%%
predict1=WKNN(train_rss1,train_coord1,test_rss1,3,'powed','sorensen');
result1=cat(2,predict1,repmat(0.0,length(predict1),1));
predict2=WKNN(train_rss2,train_coord2,test_rss2,3,'powed','sorensen');
result2=cat(2,predict2,repmat(3.7,length(predict2),1));
predict3=WKNN(train_rss3,train_coord3,test_rss3,3,'powed','sorensen');
result3=cat(2,predict3,repmat(7.4,length(predict3),1));
predict4=WKNN(train_rss4,train_coord4,test_rss4,3,'powed','sorensen');
result4=cat(2,predict4,repmat(11.1,length(predict4),1));
predict5=WKNN(train_rss5,train_coord5,test_rss5,3,'powed','sorensen');
result5=cat(2,predict5,repmat(14.8,length(predict5),1));
%%
prediction=[result1;result2;result3;result4;result5];
coordinate=[test_coord1;test_coord2;test_coord3;test_coord4;test_coord5];
[error1,accuracy1]=calculateAccuracy(result1,test_coord1);
[error2,accuracy2]=calculateAccuracy(result2,test_coord2);
[error3,accuracy3]=calculateAccuracy(result3,test_coord3);
[error4,accuracy4]=calculateAccuracy(result4,test_coord4);
[error5,accuracy5]=calculateAccuracy(result5,test_coord5);
[error,accuracy]=calculateAccuracy(prediction,coordinate);
