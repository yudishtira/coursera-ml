function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


bestError = 99999999;

conditions = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

i = 1;
while (i <= length(conditions)),

	tempC = conditions(i);

	j = 1;
	while (j <= length(conditions)),

		tempSigma = conditions(j);

		model = svmTrain(X, y, tempC, @(x1,x2)gaussianKernel(x1, x2, tempSigma), tempSigma, 5);
		predictions = svmPredict (model, Xval);

		currentError = mean(double(predictions ~= yval));
		
		if (currentError < bestError)
			bestError = currentError;
			C = tempC;
			sigma = tempSigma;
		endif

		j=j+1;

	end

i=i+1;

end



% =========================================================================

end
