clear all;
close all;
load('dataset.mat');
load('target.mat');
[row column]=size(X);
%the number of digits to classify is 10
num=10;
temp=ones(row,1);
weight=zeros([num column+1]);
%By setting the bias parameter W0=1, we can construct a design matrix
designmat=horzcat(temp,X);
%setting random number of iterations
%The iterations will help us in gradient descent. 
%Rough hueristics have been chosen here
iter=randi([500,2000],1);
if iter<550
    iter=iter+600;
end
error=zeros([1 iter]);
% Gradient Descent Approach
for i=1:iter
    %Applying the logistic sigmoid function with parameters a=0 
    %and c=1, we have
    %Logistic function
    yn=sigmf(designmat*weight',[1 0]);
    %Entropy error given by E(w)=-sum(target*ln*yn+(1-tn)*ln(1-yn))
    entropyerr=(target.*log(yn)+(1-target).*log(1-yn));
    error(1,i)= -sum(entropyerr(:));
    wnext = weight' -0.00001*(designmat'*(yn-target));
    weight=wnext'; %simultaneuos update
end
%Normalizing Error  values in the range of (1,10)
%in order to easily plot
error = error - min(error(:));
error = (error/range(error(:)))*(14-1);
error = error + 1; 
plot(1:iter, error, '-r');
xlabel('Number of Iterations');
ylabel('Normalized Error');
save weight.mat weight

