I = imread('C:\Users\user\Desktop\Github Projects\face-recognition-matlab\test_data\Person L\d8a21545-0e2a-11ef-9eef-9c2976f172db.jpg');
I = imresize(I, [224, 224]);

[label, prob] = classify(trainedNetwork_1, I);

fig = figure;
imshow(I);
title({char(label), num2str(max(prob),4)});

% Resize the figure window
set(gcf, 'Position', [100, 100, 500, 500]);
