function [error,accuracy] = calculateAccuracy(prediction, test_location)

% compare the prediction coordinates to test coordinates and get diffrence 
error=(sqrt(sum((prediction - test_location).^2,2)));
accuracy=mean(error);

end