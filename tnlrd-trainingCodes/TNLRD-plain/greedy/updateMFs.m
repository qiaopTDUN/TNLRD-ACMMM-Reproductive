function MFsALL = updateMFs(mfs, weights, filtN)
% means = mfs.means;
% precision = mfs.precision;
% D_mu = bsxfun(@minus, mfs.D, mfs.means(:));
% mfs.G = exp(-0.5*mfs.precision*D_mu.^2);
MFsALL = cell(filtN,1);
for i=1:filtN
    w = weights(:,i);
    Q = bsxfun(@times, mfs.G, w);
    MFsALL{i}.P = sum(Q,1);
    MFsALL{i}.GX = -mfs.precision * sum(bsxfun(@times, Q, mfs.D_mu),1);
end
% [p] = lut_eval(x, this.offsetD, this.step, this.P, this.G, this.GX, this.Q);