function rho = RecoverRho(t, step, len)
rho = zeros(len,1);

for i=2:len
    rho(i) = t(i)*step + rho(i-1);
end