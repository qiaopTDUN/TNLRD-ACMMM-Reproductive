function showMFsFilters(pbins, q,KernelPara)
fsz   = KernelPara.fsz;
filtN = KernelPara.filtN;
basis = KernelPara.basis;
vcof = KernelPara.cof;

m = fsz^2 - 1;
part1 = vcof(1:filtN*m);
cof_beta = reshape(part1,m,filtN);
% weight theta
part2 = vcof(filtN*m+1:end-1);
theta = exp(part2);
% construct filters
filters = basis * cof_beta;
%% show MFs
for i=1:filtN
    sfigure(i);
    plot(q,pbins(:,i));hold on;
    plot(q,2*q./(1+q.^2),'r');
    drawnow;hold off;
end
figure;DisplayFilters(filters',2,12,fsz,theta);drawnow;