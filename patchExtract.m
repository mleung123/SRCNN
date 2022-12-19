function [inputPatches, responsePatches] = patchExtract(images,patchesPerImg,patchDim, responseDim, channels,scalingFactor,std)
% PATCHEXTRACT Will extract Input and Response patches from an
% imageDatastore
%   IMAGES is a imageDatastore type containing high-resolution images that
%   the patches will be extracted from. PATCHESPERIMG is an integer
%   specifying the number of patches to be extracted from each image. It
%   must be divisible by 4, as we augment our data with rotated patches
%   PATCHDIM is an integer describing the length of one side of the square
%   patches, that must be divisible by SCALINGFACTOR. RESPONSEDIM is a 2D
%   matrix describing the dimension of the extracted Response patches.
%   CHANNELS is an integer describing the number of channels in the images
%   that IMAGES contains. SCALINGFACTOR is the factor by which the patches
%   should be shrunk, blurred, and expanded by.


%   INPUTPATCHES is a [patchSize patchSize channels numPatches] matrix
%   containing the input sub-images extracted from images. 
%   RESPONSEPATCHES is a [responseSize responseSize channels numPatches] matrix
%   containing the response sub-images extracted from images.
imgCount = size(images.Files,1);
numPatches = patchesPerImg * imgCount;
inputPatches = ones(patchDim, patchDim, channels, numPatches);
responsePatches = ones(responseDim, responseDim, channels, numPatches);

for i = 1 : imgCount % For every image
    
    currImg = readimage(images,i);
    currImg = im2double(currImg);
    if(channels == 1)
        currImg = rgb2ycbcr(currImg);
        currImg = currImg(:,:,1);
    end
    % Convert to single channel luminance images

    for j = 1 : 4 : patchesPerImg %For every patch        
        % Randomly select Upper Left hand corner of input patch coords
        inpXMin = randi(size(currImg,2)-patchDim); 
        inpYMin = randi(size(currImg,1)-patchDim); 
        
        % Select Upper Left hand corner of response patch coords
        respXMin = inpXMin + (patchDim-responseDim)/2; 
        respYMin = inpYMin + (patchDim-responseDim)/2;
        
        
        %Patch Initialization
        inpCroppedPatch = imcrop(currImg, [inpXMin inpYMin patchDim-1 patchDim-1]); 
        respCroppedPatch = imcrop(currImg, [respXMin respYMin responseDim-1 responseDim-1]);
        

        %Shrink, blur, and scale back up to the original patch size to
        %obtain lowRes images
        shrinkPatch = imresize(inpCroppedPatch,1/scalingFactor);
        blurredPatch = imgaussfilt(shrinkPatch,std); 
        finalPatch=imresize(blurredPatch, [patchDim patchDim],"bicubic");
        
       
        for k = 0 : 3
            % Add each patch orientation to their matrices
            responsePatches(:,:, :,j+((i-1)*patchesPerImg)+k) = imrotate(respCroppedPatch,90*k);
            inputPatches(:,:, :, j+((i-1)*patchesPerImg)+k) = imrotate(finalPatch,90*k);

        end
        
    end
end

end

