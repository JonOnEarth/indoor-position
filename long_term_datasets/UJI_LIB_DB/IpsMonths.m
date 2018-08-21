% Copyright (c) 2017 Universitat Jaume I (UJI)
%
% Permission is hereby granted, free of charge, to any person obtaining a copy of
% this software and associated documentation files (the "Software"), to deal in
% the Software without restriction, including without limitation the rights to
% use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
% of the Software, and to permit persons to whom the Software is furnished to do
% so, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.

% Script to generate the paper's figures "ipsError" and "ipsFSR"

close all;

addpath('db','files','ids','ips','shelves');

% For Cse method
shelfPolys = loadShelves();

% For reproducibilty in the random method
rng('default');

% Common to all methods
mounthAmount = 15;
notDetected = 100;
monthRange = (1:mounthAmount);

% Storage for 2D error
metricRand = zeros(1, mounthAmount);
metricProb = zeros(1, mounthAmount);
metricKnn = zeros(1, mounthAmount);
metricStg = zeros(1, mounthAmount);
metricCse = zeros(1, mounthAmount);
metricGk = zeros(1, mounthAmount);

% Storage for floor detection rate
rateRand = zeros(1, mounthAmount);
rateProb = zeros(1, mounthAmount);
rateKnn = zeros(1, mounthAmount);
rateStg = zeros(1, mounthAmount);
rateCse = zeros(1, mounthAmount);
rateGk = zeros(1, mounthAmount);

for month = monthRange
    % load current month data
    dataTrain = loadContentSpecific('db', 1, 1, month);
    dataTest = loadContentSpecific('db', 2, [1,2,3,4,5], month);
    
    % deal with not seen AP
    dataTrain.rss(dataTrain.rss==100) = -105;
    dataTest.rss(dataTest.rss==100) = -105;
    
    % random location estimation
    kAmount = 1;    % Single Point
    [predictionRandom] = randomEstimation(dataTrain.rss, dataTest.rss, dataTrain.coords, kAmount);
    [errorRandom,fsrR] = customError(predictionRandom, dataTest.coords, 0);
    metricRand(1, month) = getMetric(errorRandom);
    rateRand(1, month) = fsrR;
    
    % Probabilistic method estimation
    kValue = 1;    % Single Point
    [predictionProb] = probEstimation(dataTrain.rss, dataTest.rss, dataTrain.coords, kValue, floor(dataTrain.ids./100));
    [errorProb,fsrP] = customError(predictionProb, dataTest.coords, 0);
    metricProb(1, month) = getMetric(errorProb);
    rateProb(1, month) = fsrP;
    
    % kNN method estimation
    knnValue = 9;    % Number of neighbors
    predictionKnn = kNNEstimation(dataTrain.rss, dataTest.rss, dataTrain.coords, knnValue);
    [errorKnn,fsrK] = customError(predictionKnn, dataTest.coords, 0);
    metricKnn(1, month) = getMetric(errorKnn);
    rateKnn(1, month) = fsrK;
    
    % Stg method estimation
    stgValue = 3;    % AP filtering value
    kValue = 5;    % Number of neighbors
    predictionStg = stgKNNEstimation(dataTrain.rss, dataTest.rss, dataTrain.coords, stgValue, kValue);
    [errorStg,fsrS] = customError(predictionStg, dataTest.coords, 0);
    metricStg(month) = getMetric(errorStg);
    rateStg(1, month) = fsrS;
    
    % Cse method estimation
	% THIS METHOD IS NOT PROVIDED NOW
    % margin = 4;     % margin parameter
    % predictionCSE = cseEstimation(dataTrain.rss, dataTest.rss, dataTrain.coords, 2, margin, shelfPolys);
    % [errorCSE,fsrC] = customError(predictionCSE, dataTest.coords, 0);
    % metricCse(month) = getMetric(errorCSE);
    % rateCse(1, month) = fsrC;
    
    % Gk method estimation
    std_dB = 4; % (has almost no effect in this scenario)
    kValue = 12;
    predictionGk = gaussiankernelEstimation(dataTrain.rss, dataTest.rss, dataTrain.coords, std_dB, kValue);
    [errorGk,fsrGk] = customError(predictionGk, dataTest.coords, 0);
    metricGk(1, month) = getMetric(errorGk);
    rateGk(1, month) = fsrGk;

    disp(month);
end


% Display figure "ipsError" and "ipsFSR"
figure('PaperUnits','centimeters','PaperSize',[40,20],'PaperPosition',[0 0 40 20]); hold on;
plot(monthRange, metricRand, 'Color', [1 0 1], 'LineWidth', 2);
plot(monthRange, metricProb, 'Color', [0 1 1], 'LineWidth', 2);
plot(monthRange, metricKnn, 'Color', [1 0 0], 'LineWidth', 2);
plot(monthRange, metricStg, 'Color', [0 0 1], 'LineWidth', 2);
plot(monthRange, metricCse, 'Color', [0 1 0], 'LineWidth', 2);
plot(monthRange, metricGk, 'Color', [0 0 0], 'LineWidth', 2); hold off;
legend({'Rand','Prob','kNN','Stg', 'Cse', 'Gk'},'fontsize',16,'Location','eastoutside');
axis([1,mounthAmount,0,12]);
xticks((1:1:mounthAmount));
xlabel('month number','fontsize',16);
ylabel('75 percentile error (m)','fontsize',16);
grid on;

% Display figure "ipsFSR"
figure('PaperUnits','centimeters','PaperSize',[40,20],'PaperPosition',[0 0 40 20]); hold on;
plot(monthRange, rateRand, 'Color', [1 0 1], 'LineWidth', 2);
plot(monthRange, rateProb, 'Color', [0 1 1], 'LineWidth', 2);
plot(monthRange, rateKnn, 'Color', [1 0 0], 'LineWidth', 2);
plot(monthRange, rateStg, 'Color', [0 0 1], 'LineWidth', 2);
plot(monthRange, rateCse, 'Color', [0 1 0], 'LineWidth', 2);
plot(monthRange, rateGk, 'Color', [0 0 0], 'LineWidth', 2); hold off;
legend({'Rand','Prob','kNN','Stg', 'Cse', 'Gk'},'fontsize',16,'Location','eastoutside');
axis([1,mounthAmount,0.4,1]);
xticks((1:1:mounthAmount));
xlabel('month number','fontsize',16);
ylabel('Rate of Floor Detection Success','fontsize',16);
grid on;


function [metric] = getMetric(errors)
    metric = prctile(errors, 75);
end