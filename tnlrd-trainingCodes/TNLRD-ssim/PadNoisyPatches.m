function [PT, T, samples, neig, wei] = PadNoisyPatches(TP, R)
psz     = sqrt(size(TP.G_TRUTH,1));
bsz     = TP.fsz+1;
bndry   = [bsz,bsz];
pad     = @(x) padarray(x,bndry,'symmetric','both');

imdims  = [psz+2*bsz,psz+2*bsz];
npixels = prod(imdims);

samples = zeros(npixels,R);
neig    = cell(R,1);
wei     = cell(R,1);
noisy   = TP.F_NOISE;
fsz3D   = TP.fsz3D;
wsz     = TP.wsz;
bd      = TP.bd;

parfor i=1:R
    y            = reshape(noisy(:,i),psz,psz);
    y            = pad(y);
    [ne,~,we]    = NLMatching(y, fsz3D, wsz, bd);
    neig{i}      = ne+1;
%    wei{i}       = we;
    wei{i}       = ones(size(ne));
    samples(:,i) = y(:);
end

% truncation/cropping matrix T
t          = bsz;
[r,c]      = ndgrid(1+t:imdims(1)-t, 1+t:imdims(2)-t);
ind_int    = sub2ind(imdims, r(:), c(:));
d          = zeros(imdims); d(ind_int) = 1;
T          = spdiags(d(:),0,npixels,npixels);
T          = T(ind_int,:);

% padding matrix P that mirror boundary pixels, symmetric boundary
% condition
idximg     = reshape(1:prod(imdims-2*t),imdims-2*t);
pad_idximg = padarray(idximg,[t t],'symmetric','both');
P          = sparse((1:npixels)',pad_idximg(:),ones(npixels,1),npixels,prod(imdims-2*t));

% first truncation, then padding
PT         = P*T;