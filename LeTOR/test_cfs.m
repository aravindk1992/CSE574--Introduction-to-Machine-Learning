%Closed Form Solution Testing

function [d]= test_cfs(c,test_rn,complexity,lambda,test_data,test_targetdata,coeffs)
designmat_test=ones(test_rn,c);
designmatt_validate=ones(c,test_rn);
for i = 1:complexity-1
    for j =1:length(test_data)
        for k =((i-1)*46+2):((i-1)*46+47)
            designmat_test(j,k) =(exp(-(((test_data(j,k-(1+(i-1)*46))- coeffs(i,1))^2)/(2*(coeffs(i,2)^2)))));
        end
    end
end

designmatt_test=transpose(designmat_test);
weight=pinv(designmatt_test*designmat_test+lambda*eye(c,c))*designmatt_test*test_targetdata;

Ermstest= sqrt(transpose(designmat_test*weight-test_targetdata)*(designmat_test*weight-test_targetdata)/test_rn);
d=Ermstest;

end
