% This script is doing floor detection by choosing the nearest neigbor when
% apply short term & long term data

clear;
% load short term data
short_train_rss = csvread('Training_rss_21Aug17.csv');
short_train_rss(short_train_rss==100)=-103;
short_train_location = csvread('Training_coordinates_21Aug17.csv');
short_test_rss = csvread('Test_rss_21Aug17.csv');
short_test_rss(short_test_rss==100)=-103;
short_test_location = csvread('Test_coordinates_21Aug17.csv');

% load long term data
long_train_rss = csvread('longterm_train_rss.csv');
long_train_rss(long_train_rss==100)=-103;
long_train_location = csvread('longterm_train_coords.csv');
long_test_rss = csvread('longterm_test_rss.csv');
long_test_rss(long_test_rss==100)=-103;
long_test_location = csvread('longterm_test_coords.csv');

%%
short_FD_result = WKNN(short_train_rss,short_train_location(:,3),short_test_rss,1,'powed','sorensen');
long_FD_result = WKNN(long_train_rss,long_train_location(:,3),long_test_rss,1,'powed','sorensen');

%% compare long term to short term 
short_FD_rate = floorDetectionRate(short_FD_result,short_test_location(:,3));
long_FD_rate = floorDetectionRate(long_FD_result,long_test_location(:,3));