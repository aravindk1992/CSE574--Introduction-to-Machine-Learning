load('testdata.mat');
load('testtarget.mat');
load('w_hidden.mat');
load('w_output.mat');
[row column]=size(testdata);
temp=ones(row,1);
A=zeros([1500 10]);
designmattest=horzcat(temp,testdata);
Yntest=zeros(1500,10);
count=zeros([1 10]);
i=0;
wo=wo(:,2:101);

for i=1:1500
    Yt = designmattest(i,:);
    aH = Yt*wh';
    zH = sigmf(aH, [1 0]);
    aO = wo*zH';
    zO = sigmf(aO, [1 0]);
    Yntest(i,:) = zO(:);
end

for i=1:10
    for j=(i-1)*150+1:i*150
        if(Yntest(j,i)>0.5)
            A(j,i)=1;
        end
    end
end
dAB= testtarget-A;

for i=1:10
    for j=(i-1)*150+1:i*150
        if(dAB(j,i)==1)
            count(1,i)=count(1,i)+1;
        end
    end
end

Error_Rate= (sum(count(:))/1500)*100



