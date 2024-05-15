% Initialize variables to store true labels and predicted labels
trueLabels = {};
predictedLabels = {};

% Specify the directory containing your test data
testDataDir = 'C:\Users\user\Desktop\Github Projects\face-recognition-matlab\test_data';

% Get a list of all subdirectories (each representing a class)
classList = dir(testDataDir);
classList = classList([classList.isdir]);
classList = classList(~ismember({classList.name}, {'.', '..'}));

% Load the trained network
load("C:\Users\user\Desktop\Github Projects\face-recognition-matlab\resnet50\trained_model.mat"); % Assuming you saved your trained network in a .mat file

% Loop through each class directory
for classIdx = 1:numel(classList)
    currentClassDir = fullfile(testDataDir, classList(classIdx).name);
    
    % Get a list of all image files in the current class directory
    imageFiles = dir(fullfile(currentClassDir, '*.jpg')); % Assuming images are in jpg format
    
    % Loop through each image in the current class directory
    for imgIdx = 1:numel(imageFiles)
        % Load the test image
        img = imread(fullfile(currentClassDir, imageFiles(imgIdx).name));
        img = imresize(img, [224, 224]);
        
        % Classify the image using the trained network
        label = classify(trainedNetwork_1, img);
        
        % Store true label (folder name)
        trueLabels = [trueLabels; classList(classIdx).name];
        
        % Store predicted label
        predictedLabels = [predictedLabels; label];
    end
end

% Convert cell array of labels to categorical array
trueLabels = categorical(trueLabels);
predictedLabels = categorical(predictedLabels);

% Generate confusion matrix
figure;
plotconfusion(trueLabels, predictedLabels);
