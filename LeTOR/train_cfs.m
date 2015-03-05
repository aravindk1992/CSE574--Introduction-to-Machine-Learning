
function[rms_train,rms_valid,rms_test]=  train_cfs(complexity,lambda)
load project1_data.mat
%Linear Regression with basis function
format long  %Format all the computation in long
[rn,nc]=size(A);
%Init
%Number of features is 46

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
save('mu_cfs','a1');
save('s_cfs','a2');
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

%Design matrix has been computed now we need to compute
%the W*(min) using w*=(designmat'designmat+LI)^-1* designmat'*t


designmatt=zeros(c,train_rn);
designmatt=transpose(designmat);
weight=pinv(designmatt*designmat+lambda*eye(c,c))*designmatt*train_targetdata;
save('w_cfs','weight');
%Calculate Squared Error
%Ed= (designmat*w-t)'*(designmat*w-t)+(1/2)*L*w'*w
%ERMS=sqrt(Ed/N)

Ermstrain= sqrt(transpose(designmat*weight-train_targetdata)*(designmat*weight-train_targetdata)/train_rn);
rms_train=Ermstrain;
%---------------------------------------------------------------------%
%Validation Phase

designmat_validate=ones(validation_rn,c);
designmatt_validate=ones(c,validation_rn);
for i = 1:complexity-1
    for j =1:length(validate_data)
        for k =((i-1)*46+2):((i-1)*46+47)
            designmat_validate(j,k) =(exp(-(((validate_data(j,k-(1+(i-1)*46))- coeffs(i,1))^2)/(2*(coeffs(i,2)^2)))));
        end
    end
end

designmatt_validate=transpose(designmat_validate);
weight=pinv(designmatt_validate*designmat_validate+lambda*eye(c,c))*designmatt_validate*validation_targetdata;

Ermsvalidate= sqrt(transpose(designmat_validate*weight-validation_targetdata)*(designmat_validate*weight-validation_targetdata)/validation_rn);
rms_valid=Ermsvalidate;
%-----------------------------------------------------------------------

rms_test= test_cfs(c,test_rn,complexity,lambda,test_data,test_targetdata,coeffs);
end