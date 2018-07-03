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


h = sigmoid(X * theta);
n = size(theta);

temp1 = (-y(1) * log(h(1)) - (1 - y(1)) * log(1 - h(1)))/m;

thet = theta(2:n);
J = temp1 + (sum((-y(2:m) .* log(h(2:m))) - ((1-y(2:m)) .* log(1-h(2:m))))/m) + (lambda * sum(thet .^ 2))/(2 *m);

grad(1) = sum(h - y)/m;

grad(2:n) = (( (X(:,2:n))' * (h-y) ) + lambda .* thet) ./ m;




% =============================================================

end
