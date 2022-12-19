%% Pixel-Shuffle Layer
% Code in predict layer inspired by
% https://www.mathworks.com/matlabcentral/fileexchange/95228-srgan-vgg54-single-image-super-resolution-matlab-port
% 

classdef shuffleLayer < nnet.layer.Layer ...
        & nnet.layer.Formattable %... % (Optional) 
        % & nnet.layer.Acceleratable % (Optional)

    properties
        % (Optional) Layer properties.

        % Declare layer properties here.
        Scale
    end
    
  
    methods
        function layer = shuffleLayer(name,scale)
            % (Optional) Create a shuffleLayer.
            layer.Name = name;
            layer.Scale = scale;  
            layer.Description  = 'shuffle';
        end
        
        function Z = predict(layer, X)
            % Rearranges an array of shape (patchSize,patchSize,
            % ChannelSize*upscaleFactor^2, BatchSize) to an array of shape
            % (patchSize*upScaleFactor,patchSize*upScaleFactor, ChannelSize, BatchSize)
            %
            % Inputs:
            %         layer - Layer to forward propagate through 
            %         X     - An array of shape (patchSize,patchSize,
            %         ChannelSize*upscaleFactor^2, BatchSize)
            % Outputs:
            %         Z     - An array of shape
            %         (patchSize*upScaleFactor,patchSize*upScaleFactor,
            %         ChannelSize, BatchSize)

            %
            X_shape = size(X);
            patchSize = X_shape(1);
            channels = X_shape(3);
            batches = X_shape(4);
            
            newChannels = channels/(layer.Scale^2);
            newPatchSize = patchSize*layer.Scale;
            Z = zeros(newPatchSize,newPatchSize,newChannels,batches,'like',X);
           
            %shuffle with structure
            %first column
            Z(1 : layer.Scale:patchSize*layer.Scale, 1 : layer.Scale:patchSize*layer.Scale, 1:newChannels,:) = ...
                X(1:patchSize,1:patchSize,1:layer.Scale^2:channels,:);

            Z(2:layer.Scale:patchSize*layer.Scale, 1:layer.Scale:patchSize*layer.Scale, 1:newChannels,:) = ...
                X(1:patchSize,1:patchSize,2:layer.Scale^2:channels,:);

            Z(3:layer.Scale:patchSize*layer.Scale, 1:layer.Scale:patchSize*layer.Scale, 1:newChannels,:) = ...
                X(1:patchSize,1:patchSize,3:layer.Scale^2:channels,:);
                
            %second column
            Z(1:layer.Scale:patchSize*layer.Scale, 2:layer.Scale:patchSize*layer.Scale, 1:newChannels,:) = ...
                X(1:patchSize,1:patchSize,4:layer.Scale^2:channels,:);

            Z(2:layer.Scale:patchSize*layer.Scale, 2:layer.Scale:patchSize*layer.Scale, 1:newChannels,:) = ...
                X(1:patchSize,1:patchSize,5:layer.Scale^2:channels,:);

            Z(3:layer.Scale:patchSize*layer.Scale, 2:layer.Scale:patchSize*layer.Scale, 1:newChannels,:) = ...
                X(1:patchSize,1:patchSize,6:layer.Scale^2:channels,:);
            
            %third column
            Z(1:layer.Scale:patchSize*layer.Scale, 3:layer.Scale:patchSize*layer.Scale, 1:newChannels,:) = ...
                X(1:patchSize,1:patchSize,7:layer.Scale^2:channels,:);

            Z(2:layer.Scale:patchSize*layer.Scale, 3:layer.Scale:patchSize*layer.Scale, 1:newChannels,:) = ...
                X(1:patchSize,1:patchSize,8:layer.Scale^2:channels,:);

            Z(3:layer.Scale:patchSize*layer.Scale, 3:layer.Scale:patchSize*layer.Scale, 1:newChannels,:) = ...
                X(1:patchSize,1:patchSize,9:layer.Scale^2:channels,:);
                
            %functional shuffling using for loops
%             for i = 1:scalingFactor
%                 for j = 1:scalingFactor
%                     Z(j : layer.Scale:patchSize*layer.Scale, i : layer.Scale:patchSize*layer.Scale, 1:newChannels,:) = ...
%                 X(1:patchSize,1:patchSize,i*j:layer.Scale^2:channels,:);
%                 end
%             end
        end
        
    end
end

