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

v1=log(sigmoid(X*theta));
v2=log(1-sigmoid(X*theta));
%J= -1/m*((y'*v1)+((1-y)'*v2))+((lambda/m)*theta.^2);
%k=length(theta);
%for k=1:length(theta),
%    t=t+theta(k).^2;
%end;
J= -1/m*((y'*v1)+((1-y)'*v2))+((lambda/(2*m))*sum(theta(2:size(theta)).^2));
%grad(0)=1/m*((sigmoid(X*theta)-y)*X(1,1);
%grad(2:size(theta))= 1/m*(X'*(sigmoid(X*theta)-y));
for j=1:length(grad),
    if j==1,
       grad(1)=1/m*(X(:,1)'*(sigmoid(X*theta)-y)); 
    else,
       grad(j)=1/m*(X(:,j)'*(sigmoid(X*theta)-y))+(lambda/m)*theta(j);
    end;
end;




% =============================================================

end
