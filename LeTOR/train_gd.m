function[rms_gdtrain,rms_gdvalid,rms_gdtest]=train_gd(complexity)
%gradient descent
%function [rms_train,rms_validate,rms_test]= train_gd(complexity)

load project1_data.mat
%Linear Regression with basis function
format long  %Format all the computation in long
[rn,nc]=size(A);
%Init
%Number of features is 46
%complexity=20;
%Split Dataset into training, validation and testing set
train_rn = floor(0.8*rn); % 80% Training data
validation_rn=floor(0.1*rn); %10% Validation data
test_rn=floor(0.1*rn);
%Seperate out the target vectors
train_targetdata=A(1:train_rn,1); % Store the target vector for training data
validation_targetdata=A(train_rn+1:train_rn+1+validation_rn,1);
test_targetdata=A(train_rn+validation_rn+1:train_rn+validation_rn+test_rn,1);

%Seperate out the Training,Validation and Testing data for 46 features
train_data = A(1:train_rn,2:47);  % Load the training data
validate_data= A(train_rn+1:train_rn+1+validation_rn,2:47); %Load the validation data
test_data=A(train_rn+validation_rn+1:train_rn+validation_rn+test_rn,2:47); % Load the test data

%Calculating mean and variance mu(i) and sigma(i)
%Compute Mu and Sigma in a random fashion
eta=0.000000053;
for l=1:complexity
for i=1:complexity     
    for j=1:46  %randomly compute the columns and store the mean and variance of the column
      mu(j)=mean(train_data(:,randi(46,1))); 
      sigma(j)=var(train_data(:,randi(46,1)));      
    end
   coeffs(i,1)=mean(mu)+0.001; % compute the mean(mean(randomly chosen column for each row))) 
   coeffs(i,2)=mean(sigma)+0.001;  
end
a1=coeffs(:,1);
a2=coeffs(:,2);
save mu_cfs a1;
save s_gd a2;
%Construct a design matrix
c=((complexity-1)*(46))+1;
designmat=zeros(train_rn,c); %Populate the design matrix
designmat(:,1)=1; % initialize phi0

for i = 1:complexity-1
    for j =1:length(train_data)
        for k =((i-1)*46+2):((i-1)*46+47)
            designmat(j,k) =(exp(-(((train_data(j,k-(1+(i-1)*46))- coeffs(i,1))^2)/(2*(coeffs(i,2)^2)))));
        end
    end
end

%Training 
features=(complexity-1)*46+1;
num_samples=length(train_data);
weight=zeros(features,1);

for i=1:features-1         
        weight(i+1)=weight(i)+ eta*(train_targetdata(i)-(transpose(weight(i))*(designmat(i,:))))*transpose(designmat(i,:));
end

Ermstrain= sqrt(transpose(designmat*weight-train_targetdata)*(designmat*weight-train_targetdata)/train_rn);
rms_gdtrain(l)=Ermstrain;
save w_cfs weight;
  
%Validation
for i=1:features-1         
        weight(i+1)=weight(i)+ eta*(validation_targetdata(i)-(transpose(weight(i))*(designmat(i,:))))*transpose(designmat(i,:));
end

designmatt=designmat(1:validation_rn,:);
Ermsvalid= sqrt(transpose(designmatt*weight- validation_targetdata(1:validation_rn,:))*(designmatt*weight-validation_targetdata(1:validation_rn,:))/validation_rn);
rms_gdvalid(l)=Ermsvalid;

rms_gdtest(l)= test_gd(test_rn,complexity,test_data,test_targetdata,coeffs,l,features,eta);
end


%end


