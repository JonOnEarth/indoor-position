function [acc]=floorDetectionRate(predict,label)

% calculate floor detection rate
count = 0;
for i=1:length(predict)
    if predict(i)==label(i)
        count=count+1;
    end
end

acc = count/length(predict);