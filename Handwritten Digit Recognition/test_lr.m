load('testdata.mat');
load('testtarget.mat');
load('weight.mat');
[row column]=size(testdata);
temp=ones(row,1);
A=zeros([1500 10]);
designmattest=horzcat(temp,testdata);
%Computing Yn= 1/1+e^(W'*designmat)
Yntest=sigmf(designmattest*weight',[1 0]);
count=zeros([1 10]);
for i=1:10
    for j=(i-1)*150+1:(i*150)
        if(Yntest(j,i)>0.5)
            A(j,i)=1;
        end
    end
end
dAB= testtarget-A;
%TotalErrors = sum(Errors(:));
%PercentageError = TotalErrors*100/1500

for i=1:10
    for j=(i-1)*150+1:i*150
        if(dAB(j,i)==1)
            count(1,i)=count(1,i)+1;
        end
    end
end

Errors=sum(count)
Error_Rate= (Errors/1500)*100

bar(1:10,count(:),'k');

