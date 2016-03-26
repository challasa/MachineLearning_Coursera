function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X*theta)
v1 = -y' * log(h)
v2 = (1-y)' * log(1-h)
unreg_cost = (1/m) * (v1 - v2)

theta(1) = 0

sum_sq_theta = theta'*theta
reg_cost = (lambda/(2*m)) * sum_sq_theta

J = unreg_cost + reg_cost

unreg_grad = ((h-y)' * X) .* (1/m)
reg_grad = (lambda/m) * theta

grad = unreg_grad' + reg_grad




% =============================================================

end
