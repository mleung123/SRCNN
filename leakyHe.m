function [weights] = leakyHe(sz,a)
%Code used from Matlab's custom weight initialization function
%   Detailed explanation goes here
if nargin<2
    a = 0.25;
end

n = sz(1) * sz(2) * sz(3);

varWeights = 2 / ((1 + a^2) * n);
weights = randn(sz) * sqrt(varWeights);

end
