%% Obtained from https://www.mathworks.com/help/deeplearning/ug/define-custom-deep-learning-layer.html
% Modifications made to constructor method. 
classdef preluLayer < nnet.layer.Layer
    % Example custom PReLU layer.
    
    properties (Learnable)
        % Layer learnable parameters.
        
        % Scaling coefficient.
        Alpha
    end
    
    methods



        function layer = preluLayer(numChannels, name)
            % layer =  creates a PReLU layer
            % with numChannels channels and specifies the layer name.
            
            % Set layer name.
            layer.Name = name;
            
            % Set layer description.
            layer.Description = "Prelu";
            
            
            
            szAlpha = ones(1,3);
            szAlpha(end) = numChannels;
            layer.Alpha = rand(szAlpha);
        end
        
        
        function Z = predict(layer, X)
            % Z = predict(layer, X) forwards the input data X through the
            % layer and outputs the result Z.
            
            Z = max(0, X) + layer.Alpha .* min(0, X);
        end
    end
end