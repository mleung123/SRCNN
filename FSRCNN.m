%% SuperResolution: FSRCNN

%% Parameters/Hyperparameters

scalingFactor = 2;
trainSubImgSize = 7;
trainResponseDim = (trainSubImgSize * scalingFactor) - scalingFactor + 1;

f1 = 5;
f2 = 1;
f3 = 3;
f5 = 9;

dimension = 48;
shrink = 12;
channels = 1; 
numSubImgs=64;
a = 0.25; % Weights initial value

%Data Pre-Processing
validationDatasetPath = fullfile('SuperResolution', 'set5HighRes','*.png');
inputPath = fullfile('SuperResolution', 'Urban100','*HR.png');
testPath1 = fullfile('SuperResolution',  'BSD100','*HR.png');
testPath2 = fullfile('SuperResolution', 'OST100','*.png');
testPath3 = fullfile('SuperResolution', 'set14HighRes','*.png');

inputDs = imageDatastore(inputPath);
validDs = imageDatastore(validationDatasetPath);
testDs1 = imageDatastore(testPath1);
testDs2 = imageDatastore(testPath2);
testDs3 = imageDatastore(testPath3);

% Patch Initialization
[inputPatches, responseInputPatches] = subImgExtract(inputDs,numSubImgs,trainSubImgSize, trainResponseDim, channels,scalingFactor);
[validPatches, responseValidPatches] = subImgExtract(validDs,numSubImgs, trainSubImgSize, trainResponseDim, channels,scalingFactor);

options = trainingOptions('adam', ...
               plots = 'training-progress', ...
               Verbose = 1, ...
               ValidationData = {validPatches responseValidPatches}, ...
               ValidationFrequency = 50, ...
               InitialLearnRate = 10^-3, ...
               LearnRateSchedule = 'piecewise', ...
               LearnRateDropPeriod =50, ...
               MaxEpochs=300);

%Layer Initialization
conv1 =  convolution2dLayer(f1,dimension,'name','conv1','Padding','same','WeightsInitializer', @(sz) leakyHe(sz,a));

conv2 =  convolution2dLayer(f2,shrink,'name','conv2','Padding', 'same','WeightsInitializer', @(sz) leakyHe(sz,a));

map1 =  convolution2dLayer(f3,shrink,'name','map1', 'Padding', 'same','WeightsInitializer', @(sz) leakyHe(sz,a));
map2 =  convolution2dLayer(f3,shrink,'name','map2', 'Padding', 'same','WeightsInitializer', @(sz) leakyHe(sz,a));
map3 =  convolution2dLayer(f3,shrink,'name','map3', 'Padding', 'same','WeightsInitializer', @(sz) leakyHe(sz,a));
map4 =  convolution2dLayer(f3,shrink,'name','map4', 'Padding', 'same','WeightsInitializer', @(sz) leakyHe(sz,a));

prelu1 = preluLayer(dimension, "prelu1");
prelu2 = preluLayer(shrink, "prelu2");
prelu3 = preluLayer(shrink, "prelu3");
prelu4 = preluLayer(shrink, "prelu4");
prelu5 = preluLayer(shrink, "prelu5");
prelu6 = preluLayer(shrink, "prelu6");
prelu7 = preluLayer(dimension, "prelu7");

conv3 =  convolution2dLayer(channels,dimension,'name','conv3','Padding', 'same','WeightsInitializer', @(sz) leakyHe(sz,a));

transConv =  transposedConv2dLayer(f5,channels,'name','transConv','stride',scalingFactor, 'Cropping', 4);

% Layer Assembly
layers = [imageInputLayer([trainSubImgSize, trainSubImgSize, channels],'name','input')
          conv1 %Feature Extraction
          prelu1
          conv2 %Shrink
          prelu2
          map1 %Non-linear Mapping of m times
          prelu3
          %map2 %Non-linear Mapping of m times
          %prelu4
          %map3 %Non-linear Mapping of m times
          %prelu5
          map4 %Non-linear Mapping of m times
          prelu6
          conv3 %Expanding
          prelu7
          transConv %Transposed Convolutional Layer
          regressionLayer('name','regression')];
      
lgraph = layerGraph(layers);

[net, info] = trainNetwork(inputPatches,responseInputPatches, lgraph,options);
newMeans = net.Layers(1).Mean;


%% Net Surgery / Test Set Creation
testSize = 80;
testResponseDim = (testSize * scalingFactor - scalingFactor + 1); 
%Replace the input layer to allow for larger patch inputs
newInputLayer = imageInputLayer([testSize, testSize, channels],'name', 'newInput',Mean = newMeans);
lgraph = layerGraph(net);
lgraph = replaceLayer(lgraph,'input',newInputLayer);
testNet = assembleNetwork(lgraph);

%Test Set Creation
[testPatches1, responseTestPatches1] = subImgExtract(testDs1,1, testSize, testResponseDim,channels,scalingFactor);
[testPatches2, responseTestPatches2] = subImgExtract(testDs2,1, testSize, testResponseDim,channels,scalingFactor);
[testPatches3, responseTestPatches3] = subImgExtract(testDs3,1, testSize, testResponseDim,channels,scalingFactor);

%Predictions
prediction1 = predict(testNet,testPatches1);
psnrCalc1 = psnr(prediction1, single(responseTestPatches1));

prediction2 = predict(testNet,testPatches2);
psnrCalc2 = psnr(prediction2, single(responseTestPatches2));

prediction3 = predict(testNet,testPatches3);
psnrCalc3 = psnr(prediction3, single(responseTestPatches3));

% Computes average PSNR using bi-cubic interpolation for comparison with the
% trained network's PSNR
biSumPsnr1 = 0;
biSumPsnr2 = 0;
biSumPsnr3 = 0;


for i = 1 : size(testDs1.Files,1)
    currImg = readimage(testDs1,i);
    currImg = im2double(currImg);
    % Convert to single channel luminance images
    
    if(channels == 1)
        currImg = rgb2ycbcr(currImg);
        currImg = currImg(:,:,1);
    end
    
    % Shrink image by scaling factor
    shrinkImg= imresize(currImg,1/scalingFactor);
    upscale = imresize(shrinkImg, [size(currImg,1) size(currImg,2)], 'bicubic');
    biSumPsnr1 = biSumPsnr1 + psnr(currImg,upscale);
end
avgBiPsnr1 = biSumPsnr1/ size(testDs1.Files,1);

for i = 1 : size(testDs2.Files,1)
    currImg = readimage(testDs2,i);
    currImg = im2double(currImg);
    % Convert to single channel luminance images
    if(channels == 1)
        currImg = rgb2ycbcr(currImg);
        currImg = currImg(:,:,1);
    end
    
    % Shrink image by scaling factor
    shrinkImg= imresize(currImg,1/scalingFactor);
    upscale = imresize(shrinkImg, [size(currImg,1) size(currImg,2)], 'bicubic');
    biSumPsnr2 = biSumPsnr2 + psnr(currImg,upscale);
end
avgBiPsnr2 = biSumPsnr2/ size(testDs2.Files,1);

for i = 1 : size(testDs3.Files,1)
    currImg = readimage(testDs3,i);
    currImg = im2double(currImg);
    % Convert to single channel luminance images
    if(channels == 1)
            currImg = rgb2ycbcr(currImg);
            currImg = currImg(:,:,1);
    end
    % Shrink image by scaling factor
    shrinkImg= imresize(currImg,1/scalingFactor);
    upscale = imresize(shrinkImg, [size(currImg,1) size(currImg,2)], 'bicubic');
    biSumPsnr3 = biSumPsnr3 + psnr(currImg,upscale);
end
avgBiPsnr3 = biSumPsnr3/ size(testDs3.Files,1);

%% Data Visualization
for i = 1:9
    figure(1);   
    subplot(3,3,i);
    imshow(testPatches3(:,:,:,i),[]);
    title('Low-Res Input');
    figure(2);   
    subplot(3,3,i);
    imshow(prediction3(:,:,:,i),[]);
    title('Network Output');

    pause(0.1);
    figure(3);
    subplot(3,3,i);
    imshow(responseTestPatches3(:,:,:,i),[]);
    title('Ground Truth');

end