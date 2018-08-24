function [prediction] = WKNN(train_rss, train_location, test_rss, k_value, datarep, distance_metric)
% weighted K nearest neighbors algorithm which uses training coordinates, 
% training rss, test rss, and distance metric to calculate the prediction coordinates

fingerprintNum = size(train_location,1);
dimension      = size(train_location,2);
testNum        = size(test_rss,1);
prediction     = zeros(testNum,dimension);
k=k_value;

% Apply new data representation                 
if  strcmp(datarep,'positive') % Convert negative values to positive by subtracting minimum value
    [train_rss,test_rss] = datarepPositive(train_rss,test_rss);
elseif strcmp(datarep,'exponential') % Convert to exponential data representation
    [train_rss,test_rss] = datarepExponential(train_rss,test_rss);
elseif strcmp(datarep,'powed') % Convert to powed data representation
    [train_rss,test_rss] = datarepPowed(train_rss,test_rss);    
end % Default, no conversion

% calculate similarity between test point and fingerprint, simulate prediction coordinates
for i=1:testNum
    % get distance between test data and fingerprint data
    if strcmp(distance_metric,'euclidean')
        distances = sqrt(sum((train_rss - repmat(test_rss(i,:),fingerprintNum,1)).^2,2));
    elseif strcmp(distance_metric,'manhattan')
        distances = sum(abs(train_rss - repmat(test_rss(i,:),fingerprintNum,1)),2);
    elseif strcmp(distance_metric,'sorensen')
        distances = sum(abs(train_rss - repmat(test_rss(i,:),fingerprintNum,1)),2)./sum(abs(train_rss + repmat(test_rss(i,:),fingerprintNum,1)),2);
    elseif strcmp(distance_metric,'neyman')
        distances = sum(((train_rss - repmat(test_rss(i,:),fingerprintNum,1)).^2)./(abs(train_rss)+(1e-6)*(train_rss==0)),2);
    elseif strcmp(distance_metric,'cosine')
        distances = sum(train_rss.*repmat(test_rss(i,:),fingerprintNum,1),2)./sqrt((sum(train_rss.^2,2).*sum(repmat(test_rss(i,:),fingerprintNum,1).^2,2)));
    end
    
    % sort distance
    [d,index] = sort(distances);
    d(d==0) = 1e-6;
    % calculate weight 
    weight=1./d(1:k);
    prediction(i,:)=weight'*train_location(index(1:k),:)/sum(weight);
    %prediction(i,:)=mean(train_location(index(1:k),:));
end

end

% function for converting raw data to positive data representation
function [x1,y1] = datarepPositive(x0,y0)
minValue = min([x0(:)',y0(:)']);
x1 = x0 - minValue;
y1 = y0 - minValue;
end

% function for converting raw data to exponential data representation
function [x1,y1] = datarepExponential(x0,y0)
minValue = min([x0(:)',y0(:)']);
[x2,y2] = datarepPositive(x0,y0);
alpha = 15;
x1 = exp(x2/alpha)/exp(-minValue/alpha);
y1 = exp(y2/alpha)/exp(-minValue/alpha);
end

% function for converting raw data to power data representation
function [x1,y1] = datarepPowed(x0,y0)
minValue = min([x0(:)',y0(:)']);
[x2,y2] = datarepPositive(x0,y0);
beta = exp(1)/2;
x1 = (x2.^beta)/((-minValue)^beta);
y1 = (y2.^beta)/((-minValue)^beta);
end


