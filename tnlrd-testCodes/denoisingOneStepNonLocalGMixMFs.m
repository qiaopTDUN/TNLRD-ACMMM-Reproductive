function x_star = denoisingOneStepNonLocalGMixMFs(noisy, input, model, ne, we)
MFsALL = model.MFsALL;
K      = model.K;
p      = model.p;
nlcoef = model.nlcoef;
mfs    = model.mfs;
filtN  = length(K);
fnlsz  = size(nlcoef,1);
[r,c]  = size(input);
%% do a gradient descent step for all samples
u      = input;
f      = noisy;
g      = (u - f)*p;
for i=1:filtN
    ca  = nlcoef(:,i);
    a   = sparse(repmat(1:numel(u),fnlsz,1), ne, diag(ca)*we);   
    Ku  = imfilter(u,K{i},'symmetric');   
    Ku  = a*Ku(:);    
    Ne1 = lut_eval(Ku(:)', mfs.offsetD, mfs.step, MFsALL{i}.P, 0, 0, 0);
    Ne1 = reshape(Ne1,r,c);
    Ne1 = reshape(a'*Ne1(:),r,c);
    g   = g + imfilter(Ne1,rot90(rot90(K{i})),'symmetric');
end
x_star = u - g;
% x_star = max(0, min(x_star, 255));