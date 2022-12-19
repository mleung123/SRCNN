function [inputSubImgs, responseSubImgs] = subImgExtractNonRotate(images,patchesPerImg,patchSize, responseSize, channels,scalingFactor)

% SUBIMGEXTRACT Will extract Input and Response patches from an
% imageDatastore
%   IMAGES is a imageDatastore type containing high-resolution images that
%   the patches will be extracted from. PATCHESPERIMG is an integer
%   specifying the number of patches to be extracted from each image.
%   PATCHSIZE is an integer describing the length of one side of the square
%   patches.
%   RESPONSESIZE is a 2D matrix describing the dimension of the
%   extracted Respone patches. CHANNELS is an integer describing the number
%   of channels in the images that IMAGES contains. SCALINGFACTOR is the
%   factor by which the patches should be shrunk, blurred, and expanded by.

%   INPUTSUBIMGS is a [patchSize patchSize channels numPatches] matrix
%   containing the input sub-images extracted from images. 
%   responseSUBIMGS is a [responseSize response Size channels numPatches] matrix
%   containing the reponse sub-images extracted from images. 

imgCount = size(images.Files,1);
numPatches = patchesPerImg * imgCount;
inputSubImgs = ones(patchSize, patchSize, channels, numPatches);

responseSubImgs = ones(responseSize, responseSize, channels, numPatches);


for i = 1 : imgCount % For every image
    
    currImg = readimage(images,i);
    currImg = im2double(currImg);
    % Convert to single channel luminance images
    if(channels == 1)
            currImg = rgb2ycbcr(currImg);
            currImg = currImg(:,:,1);
    end
    % Shrink image by scaling factor
    shrinkImg= imresize(currImg,1/scalingFactor);
    
    for j = 1 : patchesPerImg % We add copies of the selected patch rotated in 90 degree increments
        % Randomly select upper Left hand corner of input patch coords
        inpXMin = randi(size(shrinkImg,2)-patchSize);
        inpYMin = randi(size(shrinkImg,1)-patchSize);
        
        %Select upper Left hand corner of response patch coords    
        respXMin = (inpXMin * scalingFactor); 
        respYMin = (inpYMin * scalingFactor); 
                   
        inpCroppedPatch = imcrop(shrinkImg, [inpXMin inpYMin patchSize-1 patchSize-1]); 
        respCroppedPatch = imcrop(currImg, [respXMin respYMin responseSize-1 responseSize-1]);
            % Add each patch orientation to their matrices
        responseSubImgs(:,:, :,j+((i-1)*patchesPerImg)) = respCroppedPatch;
        inputSubImgs(:,:, :, j+((i-1)*patchesPerImg)) = inpCroppedPatch;

        
    end
end
end