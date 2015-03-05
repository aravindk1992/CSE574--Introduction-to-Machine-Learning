function [d]= test_gd(test_rn,complexity,test_data,test_targetdata,coeffs,l,features,eta)

designmat_test=ones(test_rn,(complexity-1)*46+1);
designmatt_test=ones((complexity-1)*46+1,test_rn);
weight=zeros(features,1);
for i = 1:complexity-1
    for j =1:length(test_data)
        for k =((i-1)*46+2):((i-1)*46+47)
            designmat_test(j,k) =(exp(-(((test_data(j,k-(1+(i-1)*46))- coeffs(i,1))^2)/(2*(coeffs(i,2)^2)))));
        end
    end
end

for i=1:features-1         
        weight(i+1)=weight(i)+ eta*(test_targetdata(i)-(transpose(weight(i))*(designmat_test(i,:))))*transpose(designmat_test(i,:));
end

designmatt=designmat_test(1:test_rn,:);
Ermstest= sqrt(transpose(designmatt*weight- test_targetdata(1:test_rn,:))*(designmatt*weight-test_targetdata(1:test_rn,:))/test_rn);
d=Ermstest;
end