close all
clear all
clc
% This is code to accompany the tutorial video at
% https://www.youtube.com/watch?v=lK9YyX-q32k
% Author: CodingLikeMad, 10/30/2020.
% A similar tutorial using the same dataset is available from Mathworks at:
% https://www.mathworks.com/help/deeplearning/ug/create-simple-deep-learning-network-for-classification.html


% Begin by loading the dataset. To adapt this to your own dataset, you can
% just follow the structure in the example folder - you don't need the CSV
% files they have there.
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% Create your test/train/validation splits. There is no gold rule for this.
% The decision to randomize the order here is somewhat context dependent.
% Time series data (for instance, climate data) should NOT be randomized,
% for example.
fracTrainFiles = 0.6;
fracValFiles = 0.2;
fracTestFiles = 0.2;
[imdsTrain, imdsValidation, imdsTest ] = splitEachLabel(imds, ...
    fracTrainFiles, fracValFiles, fracTestFiles, 'randomize');

% The network is defined as a series of layers. This is similar to the
% format used by Keras, and may even be directly importable into Python if
% needed via a plugin.
%
% Something important here that I did not discuss in my tutorial is how to
% select the parameters in this structure. These are the model
% hyperparameters. I do discuss this briefly in some of my other tutorials,
% but I don't really have a good explanation anywhere. Basically you want
% to try many different values of them, and check the validation
% performance. The best validation performance model is then reported. The
% search itself is honestly just brute force. That sounds bad, but it is
% where the field is today. A monte-carlo style search is generally
% considered state of the art with substantial advantages over a grid
% search. Fine tuning it by hand (called "grad-student decent" sometimes)
% is often seen in small projects or academic work, but in industry it is
% just brute forced. Fancy methods which try to estimate hyperparameters
% with adaptive function fitting and whatnot have so far not panned out
% well, despite seeming promising to me, but perhaps a universal search
% algorithm will be found in the future. 
layers = [
    % Setup your inputs
    imageInputLayer([28 28 1])
    
    % Create your first convolutional set with activation function. 10 3x3
    % filters - I've chosen 10 to match the number of classes, but it isn't
    % necesarry. The choice to use padding "same" is largely irrelevent,
    % FOR THIS PROBLEM and just determines how the edges are treated.
    convolution2dLayer(3,10,'Padding','same')
    batchNormalizationLayer % This helps deal with variations in batch properties, doesn't matter much here.
    reluLayer % Relu layer choice is largely up to the user, pretty much any would work here.
    
    % The choice of stride DOES matter here - it controls how much
    % translational invariance is present, while the equivariance is
    % introduced by the CNNs. This network is actually not very robust to
    % translation right now.
    maxPooling2dLayer(2,'Stride',2)
    
    % Repeat the above, but note that the convolutional size is 3x3x10,
    % since it treats all channels from the above layer as seperate image
    % channels.
    convolution2dLayer(3,10,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    % You would ideally include a global pool here, like
    % globalAveragePooling2dLayer, this would DRASTICALLY decrease the
    % complexity of the model. In retrospect, I should have included one.
    % My bad. This layer would establish proper translational invariance in
    % the model as well. It basically does the maxPooling above, but over
    % the entire remaining image. 
    
    % The last fully connected layer is used to take the features we
    % identified above, and post process them to assign probibilities to
    % the final digits.
    fullyConnectedLayer(10)
    softmaxLayer % The results from the fully connected layer are converted to explicit probabilities by the soft max - it normalizes them to between 0 and 1
    classificationLayer]; % The classification layer just tells matlab to use a cross-entropy function - see my classification tutorial.

%%

% In addition to the choice of max Epochs mentioned in the video, the
% 'sgdm' and 'initialLearnRate' parameters are actually very important.
% They set which optimizer is used (sgdm in this case) and what learn rate
% to use (one of the parameters for the optimizer). Optimizers tend to be
% somewhat problem specific, and SGDM is not the most popular model in my
% experience. See the ADAM optimizer as well for instance, to name just
% one.
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',100, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

% Surprisingly, the training is just calling this method.
net = trainNetwork(imdsTrain,layers,options);

%%

YPred = classify(net,imdsTest);
YTest = imdsTest.Labels;

accuracy = sum(YPred == YTest)/numel(YTest)

%%
ind = find(YPred ~= YTest);
figure; 
for ii = 1:9
    subplot(3,3,ii);
    imagesc(imdsValidation.readimage(ind(ii)));
    title([num2str(double(YPred(ind(ii)))-1), ' predicted, ', ... 
        num2str(double(YTest(ind(ii)))-1), ' actual'])
end