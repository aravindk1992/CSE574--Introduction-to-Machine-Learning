clear all
close all
load('dataset.mat');
load('target.mat');
[row column]=size(X);
%the number of digits to classify is 10
temp=ones(row,1);
%number of neurons in the hidden layer
numOfNodes= 100;
%number of neurons in the output layer 
numOfOutput=10;
%By setting the bias parameter W0=1, we can construct a design matrix
designmat=horzcat(temp,X);
%weights of hidden layer (randomly assigned)
[r c]= size(designmat);
wh= rand(numOfNodes,c)*(rand(1)/10);
%weights of Output layer (randomly assigned)
wo= rand(numOfOutput,numOfNodes+1)*(rand(1)/10); 
ah=zeros(numOfNodes,1);
zh= zeros(numOfNodes,1);
zh_bias= zeros(numOfNodes+1,1);

for a=1:1000
    Error =0;
    delh=zeros(numOfNodes,c);
    delo=zeros(numOfOutput,numOfNodes+1);   
    for i=1:row
        %choose random row number
        k=randi([1,row]);
        %Extract the input data of that row
        xi=designmat(k,:)';
        %Extract the corresponding target
        ti=target(k,:)';
        % a= wi' *xi;
        ah=wh*xi;
        %zi=h(ai)
        zh=sigmf(ah, [1 0]);
        %adding bias
        zh_bias=vertcat(1,zh);
        % Output
        ao=wo*zh_bias;
        % Non-linear transformation of the output
        zo= sigmf(ao, [1 0]);
        % Calculate do and back-propogate
        do=(zo-ti);               
        timepass= sigmf(zh_bias,[1 0]).*sigmf((1-zh_bias),[1 0]);
        dh= (wo'*do).* timepass;
        dh([1], :) = [];
        delh=delh + dh *xi';
        delo= delo + do*zh_bias';
        %Entropy error
        loss = ti .* log(zo)+(1-ti) .* log(1-zo); 
        Error = Error + sum(loss);        
    end
    %Hessian Computation
    Error = Error - (0.001/2) * (sum(sum(wh.* wh )) + sum(sum(wo.*wo )));
    Error = -1*( Error / row);
    Errors(1,a) = Error;
    DA = ( delh ./ row  ) + 0.001.*wh;
    DB = ( delo ./ row ) + 0.001.*wo;
    wh = wh - DA;
    wo = wo - DB;
end

%Normalize error
Errors = Errors - min(Errors(:));
Errors = (Error/range(Error(:)))*(14-1);
Error = Error + 1; 
% Apply input xn to the network and forward propagate through
plot(1:1000, Errors, '-b.');
xlabel('Number of Iterations');
ylabel('Normalized Error');
save w_hidden.mat wh
save w_output.mat wo
