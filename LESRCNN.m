%% SuperResolution: LESRCNN

%% Parameters/Hyperparameters

scalingFactor = 3;

oddKernel = 3;
evenKernel = 1;

outputChannels = 64;
channels = 3; 
std=0.5 ;
numSubImgs=128;
trainSubImgSize = 12;
trainResponseDim = trainSubImgSize * scalingFactor;
a = 0.25; % Weights initial value

%Path Initialization
validationDatasetPath = fullfile('SuperResolution', 'set5HighRes','*.png');
inputPath = fullfile('SuperResolution', 'Urban100','*HR.png');

testPath1 = fullfile('SuperResolution',  'BSD100','*HR.png');
testPath2 = fullfile('SuperResolution', 'OST100','*.png');
testPath3 = fullfile('SuperResolution', 'set14HighRes','*.png');


testDs1 = imageDatastore(testPath1);
testDs2 = imageDatastore(testPath2);
testDs3 = imageDatastore(testPath3);
inputDs = imageDatastore(inputPath);
validDs = imageDatastore(validationDatasetPath);

% Patch Initialization
[inputPatches, responseInputPatches] = subImgExtract(inputDs,numSubImgs,trainSubImgSize, trainResponseDim, channels,scalingFactor);
[validPatches, responseValidPatches] = subImgExtract(validDs,numSubImgs, trainSubImgSize, trainResponseDim, channels,scalingFactor);

%Network Options
options = trainingOptions('adam', ...
               plots = 'training-progress', ...
               Verbose = 1, ...
               ValidationData = {validPatches responseValidPatches}, ...
               ValidationFrequency = 50, ...
               InitialLearnRate = 10^-3, ...
               LearnRateSchedule = 'piecewise', ...
               LearnRateDropPeriod = 2, ...
               MaxEpochs=6, ...
               MiniBatchSize = 5);

%Layer Initialization
conv1 =  convolution2dLayer(oddKernel,outputChannels,'name','conv1','Padding','same','WeightsInitializer', @(sz) leakyHe(sz,a));
conv2 =  convolution2dLayer(evenKernel,outputChannels,'name','conv2','Padding', 'same','WeightsInitializer', @(sz) leakyHe(sz,a));
conv3 =  convolution2dLayer(oddKernel,outputChannels,'name','conv3','Padding','same','WeightsInitializer', @(sz) leakyHe(sz,a));
conv4 =  convolution2dLayer(evenKernel,outputChannels,'name','conv4','Padding', 'same','WeightsInitializer', @(sz) leakyHe(sz,a));
conv5 =  convolution2dLayer(oddKernel,outputChannels,'name','conv5','Padding','same','WeightsInitializer', @(sz) leakyHe(sz,a));
conv6 =  convolution2dLayer(evenKernel,outputChannels,'name','conv6','Padding', 'same','WeightsInitializer', @(sz) leakyHe(sz,a));
conv7 =  convolution2dLayer(oddKernel,outputChannels,'name','conv7','Padding','same','WeightsInitializer', @(sz) leakyHe(sz,a));
conv8 =  convolution2dLayer(evenKernel,outputChannels,'name','conv8','Padding', 'same','WeightsInitializer', @(sz) leakyHe(sz,a));
conv9 =  convolution2dLayer(oddKernel,outputChannels,'name','conv9','Padding','same','WeightsInitializer', @(sz) leakyHe(sz,a));
conv10 =  convolution2dLayer(evenKernel,outputChannels,'name','conv10','Padding', 'same','WeightsInitializer', @(sz) leakyHe(sz,a));
conv11 =  convolution2dLayer(oddKernel,outputChannels,'name','conv11','Padding','same','WeightsInitializer', @(sz) leakyHe(sz,a));
conv12 =  convolution2dLayer(evenKernel,outputChannels,'name','conv12','Padding', 'same','WeightsInitializer', @(sz) leakyHe(sz,a));
conv13 =  convolution2dLayer(oddKernel,outputChannels,'name','conv13','Padding','same','WeightsInitializer', @(sz) leakyHe(sz,a));
conv14 =  convolution2dLayer(evenKernel,outputChannels,'name','conv14','Padding', 'same','WeightsInitializer', @(sz) leakyHe(sz,a));
conv15 =  convolution2dLayer(oddKernel,outputChannels,'name','conv15','Padding','same','WeightsInitializer', @(sz) leakyHe(sz,a));
conv16 =  convolution2dLayer(evenKernel,outputChannels,'name','conv16','Padding', 'same','WeightsInitializer', @(sz) leakyHe(sz,a));
conv17 =  convolution2dLayer(oddKernel,outputChannels,'name','conv17','Padding','same','WeightsInitializer', @(sz) leakyHe(sz,a));

convRB1 =  convolution2dLayer(oddKernel,outputChannels*scalingFactor^2,'name','convRB1','Padding','same','WeightsInitializer', @(sz) leakyHe(sz,a));
convRB2 =  convolution2dLayer(oddKernel,outputChannels*scalingFactor^2,'name','convRB2','Padding','same','WeightsInitializer', @(sz) leakyHe(sz,a));


IRB_conv1 =  convolution2dLayer(oddKernel,outputChannels,'name','IRB_conv1', 'Padding', 'same','WeightsInitializer', @(sz) leakyHe(sz,a));
IRB_conv2 =  convolution2dLayer(oddKernel,outputChannels,'name','IRB_conv2', 'Padding', 'same','WeightsInitializer', @(sz) leakyHe(sz,a));
IRB_conv3 =  convolution2dLayer(oddKernel,outputChannels,'name','IRB_conv3', 'Padding', 'same','WeightsInitializer', @(sz) leakyHe(sz,a));
IRB_conv4 =  convolution2dLayer(oddKernel,outputChannels,'name','IRB_conv4', 'Padding', 'same','WeightsInitializer', @(sz) leakyHe(sz,a));
IRB_conv5 =  convolution2dLayer(oddKernel,channels,'name','IRB_conv5', 'Padding', 'same','WeightsInitializer', @(sz) leakyHe(sz,a));

relu1 = reluLayer('Name','relu1');
relu2 = reluLayer('Name','relu2');
relu3 = reluLayer('Name','relu3');
relu4 = reluLayer('Name','relu4');
relu5 = reluLayer('Name','relu5');
relu6 = reluLayer('Name','relu6');
relu7 = reluLayer('Name','relu7');
relu8 = reluLayer('Name','relu8');
relu9 = reluLayer('Name','relu9');
relu10 = reluLayer('Name','relu10');
relu11 = reluLayer('Name','relu11');
relu12 = reluLayer('Name','relu12');
relu13 = reluLayer('Name','relu13');
relu14 = reluLayer('Name','relu14');
relu15 = reluLayer('Name','relu15');
relu16 = reluLayer('Name','relu16');
relu17 = reluLayer('Name','relu17');

shuffle1 =  shuffleLayer('shuffle1',scalingFactor);
shuffle2 =  shuffleLayer('shuffle2',scalingFactor);

RB_relu = reluLayer('Name','RB_relu');
IRB_relu1 = reluLayer('Name','IRB_relu1');
IRB_relu2 = reluLayer('Name','IRB_relu2');
IRB_relu3 = reluLayer('Name','IRB_relu3');
IRB_relu4 = reluLayer('Name','IRB_relu4');

%transConv =  transposedConv2dLayer(f5,channels,'name','transConv','stride',scalingFactor);

addLayer3 = additionLayer(2,'Name','add_3');
addLayer5 = additionLayer(2,'Name','add_5');
addLayer7 = additionLayer(2,'Name','add_7');
addLayer9 = additionLayer(2,'Name','add_9');
addLayer11 = additionLayer(2,'Name','add_11');
addLayer13 = additionLayer(2,'Name','add_13');
addLayer15 = additionLayer(2,'Name','add_15');
addLayer17 = additionLayer(2,'Name','add_17');

add_RB= additionLayer(2,'Name','add_RB');

%Layer Assembly 
layers = [imageInputLayer([trainSubImgSize, trainSubImgSize, channels],'name','input')
          %IEEB
          conv1 
          relu1
          conv2 
          relu2
          conv3
          addLayer3
          relu3
          conv4 
          relu4
          conv5 
          addLayer5
          relu5
          conv6
          relu6
          conv7
          addLayer7
          relu7
          conv8 
          relu8
          conv9
          addLayer9
          relu9
          conv10 
          relu10
          conv11
          addLayer11
          relu11
          conv12 
          relu12
          conv13
          addLayer13
          relu13
          conv14 
          relu14
          conv15
          addLayer15
          relu15
          conv16 
          relu16
          conv17
          addLayer17
          relu17
          
          
          %RB
          convRB1
          shuffle1
          
          % add second sub-pixel conv after this via net surgery
          
          add_RB
          RB_relu
         
          %IRB
          IRB_conv1
          IRB_relu1
          IRB_conv2
          IRB_relu2
          IRB_conv3
          IRB_relu3
          IRB_conv4
          IRB_relu4
          IRB_conv5
          
          regressionLayer('name','regression')];
      
lgraph = layerGraph(layers);

%Residual learning network connects( IEEB)
lgraph = connectLayers(lgraph, 'conv1', 'add_3/in2');

lgraph = connectLayers(lgraph, 'conv3', 'add_5/in2');

lgraph = connectLayers(lgraph, 'conv5', 'add_7/in2');

lgraph = connectLayers(lgraph, 'conv7', 'add_9/in2');

lgraph = connectLayers(lgraph, 'conv9', 'add_11/in2');

lgraph = connectLayers(lgraph, 'conv11', 'add_13/in2');

lgraph = connectLayers(lgraph, 'conv5', 'add_15/in2');

lgraph = connectLayers(lgraph, 'conv15', 'add_17/in2');

%Residual learning network connects(RB)
lgraph = addLayers(lgraph,convRB2);

lgraph = addLayers(lgraph,shuffle2);
lgraph = connectLayers(lgraph, 'relu1', 'convRB2');

lgraph = connectLayers(lgraph, 'convRB2', 'shuffle2');

lgraph = connectLayers(lgraph, 'shuffle2', 'add_RB/in2');



%% Training

[net, info] = trainNetwork(inputPatches,responseInputPatches, lgraph,options);
newMeans = net.Layers(1).Mean;


%% Net Surgery / Test Set Creation
testSize = 50;
testResponseDim = scalingFactor * testSize;
%Replace the input layer to allow for larger patch inputs for testing
newInputLayer = imageInputLayer([testSize, testSize, channels],'name', 'newInput',Mean = newMeans); 
lgraph = layerGraph(net);
lgraph = replaceLayer(lgraph,'input',newInputLayer);

testNet = assembleNetwork(lgraph);

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