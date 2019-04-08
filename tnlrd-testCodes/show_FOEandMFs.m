function show_FOEandMFs(KernelPara, mfs)
means = mfs.means;
precision = mfs.precision;
NumW = mfs.NumW;

basis = KernelPara.basis;
vcof = KernelPara.cof;
% tic;
filter_size = KernelPara.fsz;
m = filter_size^2 - 1;
filtN = KernelPara.filtN;
part1 = vcof(1:filtN*m);
cof_beta = reshape(part1,m,filtN);
% weight theta
% part2 = vcof(filtN*m+1:filtN*(m+1));
% theta = exp(part2);
part3 = vcof(filtN*m+1);
p = exp(part3);
part4 = vcof(filtN*m+2:end-1);
weights = reshape(part4,NumW,filtN);
part5 = vcof(end);
beta = exp(part5);

% construct filters
% filters = basis*cof_beta;
% sfigure(1);
% DisplayFilters(filters',2,12,filter_size,theta);drawnow;

%% unit norm filters
filters = [];
parfor i = 1:filtN
    x_cof = cof_beta(:,i);
    filter = basis*x_cof;
    filter = filter/norm(filter);    
    filters = [filters, filter];
end

cmap = hsv(filtN);
if 1
%     close all;
    x = mfs.D;
    x_mu = bsxfun(@minus, x, means);
    t = bsxfun(@times, x_mu.^2, -0.5*precision);
    gw = exp(t);
    sfigure(100);
    hold on;
    for i=1:filtN
        w = weights(:,i);
        q = bsxfun(@times, gw, w);
        p = sum(q,1);
        plot(x,p,'Color',cmap(i,:));drawnow;
    end
    grid on;
%     plot(x,2*x./(1+x.^2),'k.')
    hold off;
    sfigure(101);
    DisplayFilters(filters',2,12,filter_size);
    drawnow;
end