%% SuperResolution: SRCNN


%% Parameters/Hyperparameters

scalingFactor = 3;

%Filter sizes
f1 = 9;
f2 = 5;
f3 = 5;

%NumFilters
n1 = 64;
n2 = 32;

c = 3; 
std=0.5 ;

patchDim = 33;
patchesPerImg = 256;


trainResponseDim = (patchDim(1) - f1 -f2 -f3 +3);

%Data Pre-Processing
validationDatasetPath = fullfile('SuperResolution', 'set5HighRes','*.png');
%inputPath = fullfile('SuperResolution', 'set14HighRes','*.png');
%testPath = fullfile('SuperResolution', 'Urban100', 'image_SRF_2','*HR.png');

inputPath = fullfile('SuperResolution', 'Urban100','*HR.png');
testPath1 = fullfile('SuperResolution',  'BSD100','*HR.png');
testPath2 = fullfile('SuperResolution', 'OST100','*.png');
testPath3 = fullfile('SuperResolution', 'set14HighRes','*.png');

testDs1 = imageDatastore(testPath1);
testDs2 = imageDatastore(testPath2);
testDs3 = imageDatastore(testPath3);

inputDs = imageDatastore(inputPath);
validDs = imageDatastore(validationDatasetPath);

%Patch Extraction
[inputPatches, responseInputPatches] = patchExtract(inputDs,patchesPerImg,patchDim, trainResponseDim, c,scalingFactor,std);
[validPatches, responseValidPatches] = patchExtract(validDs,patchesPerImg, patchDim, trainResponseDim, c,scalingFactor,std);


options = trainingOptions('adam', ...
               plots = 'training-progress', ...
               Verbose = 1, ...
               ValidationData = {validPatches responseValidPatches}, ...
               ValidationFrequency = 50, ...
               InitialLearnRate = 10^-3, ...
               LearnRateSchedule = 'piecewise', ...
               LearnRateDropPeriod = 70, ...
               MaxEpochs=300);
% Layer Initialization
conv1 =  convolution2dLayer(f1,n1,'name','conv1');
%conv1.Weights = randn( f1, f1, c, n1, 'single');
% conv1.Bias = gpuArray(single(randn([1 n1]) * 0.001));

conv2 =  convolution2dLayer(f2,n2,'name','conv2');
%conv2.Weights = randn(f2, f2, n1, n2, 'single');
% conv2.Bias = gpuArray(single(randn([1 n2]) * 0.001));

conv3 =  convolution2dLayer(f3,c,'name','conv3');
%conv3.Weights = randn(f3, f3, n2, c, 'single');
% conv3.Bias = gpuArray(single(randn([1 c]) * 0.001));

layers = [imageInputLayer([patchDim, patchDim, c],'name','input')
          conv1
          reluLayer('name','relu1')
          conv2
          reluLayer('name','relu2')
          conv3
          regressionLayer('name','reconstruct')];
      
lgraph = layerGraph(layers);
%% Training
[net, info] = trainNetwork(inputPatches,responseInputPatches, lgraph,options);
newMeans = net.Layers(1).Mean;

%% Net Surgery / Test Set Creation
testSize = 200;
%Replace the input layer to allow for larger patch inputs
newInputLayer = imageInputLayer([testSize, testSize, c],'name', 'newInput',Mean = newMeans);
testResponseDim = (testSize - f1 -f2 -f3 +3);
lgraph = layerGraph(net);
lgraph = replaceLayer(lgraph,'input',newInputLayer);

testNet = assembleNetwork(lgraph);

%Test Set Creation
[testPatches1, responseTestPatches1] = patchExtract(testDs1,1, testSize, testResponseDim,c,scalingFactor,std);
[testPatches2, responseTestPatches2] = patchExtract(testDs2,1, testSize, testResponseDim,c,scalingFactor,std);
[testPatches3, responseTestPatches3] = patchExtract(testDs3,1, testSize, testResponseDim,c,scalingFactor,std);

%Predictions
prediction1 = predict(testNet,testPatches1);
psnrCalc1 = psnr(prediction1, single(responseTestPatches1));

prediction2 = predict(testNet,testPatches2);
psnrCalc2 = psnr(prediction2, single(responseTestPatches2));

prediction3 = predict(testNet,testPatches3);
psnrCalc3 = psnr(prediction3, single(responseTestPatches3));


%Computes average PSNR using bi-cubic interpolation for comparison with the
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
randLoc = round((size(testPatches3,4)-9)*rand());
for i = 1:9
    figure(1);
    subplot(3,3,i);
    imshow(testPatches1(:,:,:,i+randLoc),[]);
    title('Low-Res Input');
    
    figure(2);   
    subplot(3,3,i);
    imshow(prediction1(:,:,:,i+randLoc),[]);
    title('Network Output');

    pause(0.1);
    figure(3);
    subplot(3,3,i);
    imshow(responseTestPatches1(:,:,:,i+randLoc),[]);
    title('Ground Truth');
end

