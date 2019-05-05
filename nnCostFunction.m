function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% To account for the bias unit, we add a bias unit to every training example
X = [ones(m,1) X]; 
yMatrix = zeros(num_labels, m);

for i=1:m

	aOne = X(i,:)';
	size(aOne);
	
	zTwo = Theta1*aOne;

	%aTwo = [1; sigmoid(zTwo)];
	aTwo = [1; sigmoid(zTwo)];

	zThree = Theta2*aTwo; 

	aThree = sigmoid(zThree);
	
	h = aThree;
	
	%Let's construct our y vector
	yVec = (1:num_labels)' == y(i);
	
	term1 = (1/m)*(-1*yVec).*log(h);
	term2 = (1/m)*(1-yVec).*log(1-h);
	
	J = J + sum(term1 - term2);
	
	
	%backpropagation
	delta3 = aThree - yVec; 	
	delta2 = (Theta2'*delta3).*(aTwo.*(1-aTwo));
	%We want to get rid of the bias term from our calculation
	delta2 = delta2(2:end);
	
	Theta2_grad = Theta2_grad + (delta3*aTwo');
	Theta1_grad = Theta1_grad + (delta2*aOne');
end 



theta1RegTerm = Theta1(:, 2:end);
theta2RegTerm = Theta2(:, 2:end);


regTerm = (lambda/(2*m)) * (sum(sum(theta1RegTerm.^2)) + (sum(sum(theta2RegTerm.^2))));

J = J+regTerm;

Theta1_grad = Theta1_grad./m;
Theta2_grad = Theta2_grad./m;

% Implement for Theta1 and Theta 2 when l > 0
Theta1_grad = Theta1_grad + (lambda / m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = Theta2_grad + (lambda / m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
