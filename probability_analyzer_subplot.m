% Specify the directory containing your test data
testDataDir = 'C:\Users\user\Desktop\Github Projects\face-recognition-matlab\test_data';

% Get a list of all subdirectories (each representing a class)
classList = dir(testDataDir);
classList = classList([classList.isdir]);
classList = classList(~ismember({classList.name}, {'.', '..'}));

% Initialize variables to store true labels and predicted labels
trueLabels = {};
predictedLabels = {};
probabilities = zeros(0, numel(classList));

% Load the trained network
%load('C:\Users\user\Desktop\Github Projects\face-recognition-matlab\resnet50\resnet50_transfer_learning.mat');
load("C:\Users\user\Desktop\Github Projects\face-recognition-matlab\custom_cnn\custom_net.mat")

numCols = 20; 
figure('WindowState','maximized');

% Loop through each class directory
for classIdx = 1:numel(classList)
    currentClassDir = fullfile(testDataDir, classList(classIdx).name);
    
    % Get a list of all image files in the current class directory
    imageFiles = dir(fullfile(currentClassDir, '*.jpg'));
    
    % Determine the number of subplots to create
    numSubplots = min(numel(imageFiles), numCols);
    
    % Loop through each image in the current class directory
    for imgIdx = 1:numSubplots
        subplot(numel(classList), numCols, (classIdx - 1) * numCols + imgIdx);
        
        % Load the test image
        img = imread(fullfile(currentClassDir, imageFiles(imgIdx).name));
        img = imresize(img, [224, 224]);
        
        % Classify the image using the trained network
        [label, scores] = classify(trainedNetwork_1, img);
        
        % Display the image
        imshow(img);
        title(sprintf('%s\n%.4f', label, max(scores)));
        
        % Store true label (folder name)
        trueLabels = [trueLabels; classList(classIdx).name];
        
        % Store predicted label
        predictedLabels = [predictedLabels; label];
        
        % Store probabilities
        probabilities(end+1, :) = scores;
    end
end

% Convert cell array of labels to categorical array
trueLabels = categorical(trueLabels);
predictedLabels = categorical(predictedLabels);

% Generate confusion matrix
figure;
plotconfusion(trueLabels, predictedLabels);

% Calculate classification metrics
numClasses = numel(unique(trueLabels));
classLabels = categories(trueLabels);

confMat = confusionmat(trueLabels, predictedLabels);
accuracy = sum(diag(confMat)) / sum(confMat(:));

precision = zeros(numClasses, 1);
recall = zeros(numClasses, 1);
f1Score = zeros(numClasses, 1);

for i = 1:numClasses
    precision(i) = confMat(i,i) / sum(confMat(:,i));
    recall(i) = confMat(i,i) / sum(confMat(i,:));
    f1Score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
end

% Display classification report
fprintf('\nClassification Report:\n');
fprintf('---------------------------------------------------\n');
fprintf(' Precision   Recall    F1 Score     Class\n');
fprintf('---------------------------------------------------\n');
for i = 1:numClasses
    fprintf('  %-9.2f  %-9.2f  %-9.2f  %s\n', ...
        precision(i)*100, recall(i)*100, f1Score(i)*100, classLabels{i});
end
fprintf('===================================================\n');
fprintf('Accuracy:               %.2f%%\n', accuracy * 100);
fprintf('===================================================\n\n');