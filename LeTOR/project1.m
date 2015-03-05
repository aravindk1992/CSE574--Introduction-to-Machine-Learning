clear all
clc

fprintf('My ubit name is %s\n','aravindk');
fprintf('My student number is %d\n',50132610);


ra=30;
ca=3;

answers=zeros(ra,ca);
fprintf('Training Error     Validation Error    Testing Error\n');
lambda_cfs=0.001;
for i=1:ra
    [rms_train,rms_validate,rms_test]=train_cfs(i+1,lambda_cfs);
    answers(i,1)=rms_train;
    answers(i,2)=rms_validate;
    answers(i,3)=rms_test;
    fprintf('M= %d\t \t',i+1);
    fprintf('%f\t',answers(i,:));
    fprintf('\n');
end

xaxis=1:ra;
plot(xaxis,answers(:,1),'k');
grid on;
hold on;
plot(xaxis,answers(:,3),'r');
hold off;
legend('Training Error','Testing Error','Location','northoutside','Orientation','horizontal');
xlabel('Complexity');
ylabel('ERMS');
rms_cfs=min(answers(:,3));
[row, col] = find(ismember(answers, rms_cfs));
lambda_gd=lambda_cfs; 
M_cfs=row+1;

[rms_gdtrain,rms_gdvalid,rms_gdtest]=train_gd(25);
rms_gd=min(rms_gdtest);
[rowgd, colgd]=find(ismember(rms_gdtest,rms_gd));
M_gd=colgd;
fprintf('the model complexity M cfs is %d\n', M_cfs);
fprintf('the model complexity M gd is %d\n', M_gd);
fprintf('the regularization parameters lambda cfs is %f\n', lambda_cfs);
fprintf('the regularization parameters lambda gd is %f\n', lambda_gd);
fprintf('the root mean square error for the closed form solution is %f\n', rms_cfs);
fprintf('the root mean square error for the gradient descent method is %f\n', rms_gd);
