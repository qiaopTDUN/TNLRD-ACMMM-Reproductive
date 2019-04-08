function [mssim, d_ssim_x] = ssim_index(img1, img2,  BSZ, PSZ)
psz = PSZ;
bsz = BSZ;
bndry = [bsz,bsz];

crop = @(x) x(1+bndry(1):end-bndry(1),1+bndry(2):end-bndry(2));
pad   = @(x) padarray(x,bndry,0,'both');

%added by fws
img1 = pad(img1);
img2 = pad(img2);

right = ones(psz(1),psz(2));
right = pad(right);

[M N] = size(img1);
    
window = fspecial('gaussian', 11, 1.5);	%
K(1) = 0.01;								      % default settings
K(2) = 0.03;								      %
L = 255;                                  %

C1 = (K(1)*L)^2;
C2 = (K(2)*L)^2;
window = window/sum(sum(window));

mu1   = imfilter(img1,window, 'symmetric');
mu2   = imfilter(img2,window, 'symmetric');  % kg
mu1_sq = mu1.*mu1;
mu2_sq = mu2.*mu2;
mu1_mu2 = mu1.*mu2;
sigma1_sq = imfilter( img1.*img1, window, 'symmetric') - mu1_sq;
sigma2_sq = imfilter(img2.*img2, window, 'symmetric') - mu2_sq;
sigma12 = imfilter(img1.*img2, window, 'symmetric') - mu1_mu2;


v1 = 2*mu1_mu2 + C1;
v2 = (2*sigma12 + C2);
v3 = (mu1_sq + mu2_sq + C1);
v4 = (sigma1_sq + sigma2_sq + C2);

ssim_map = (v1.*v2)./(v3.*v4);

minus_v3_v4=1./(v3.*v4);
temp2 = ssim_map.*minus_v3_v4;
temp3 = v3.*temp2;
temp4 = mu2.*minus_v3_v4;
temp = (v2   -  v1).*temp4...
    -mu1.*v4.* temp2+mu1.*temp3;

sec_part = imfilter(temp.*right, window, 'symmetric');
fir_part = img2.*imfilter(right.*minus_v3_v4.*v1, window, 'symmetric')...
            - img1.*imfilter(right.*temp2.*v3, window, 'symmetric');

cropssim=crop(ssim_map);
mssim = sum(cropssim(:), [], 'double') /psz(1)/psz(2);

% d_ssim_x=2*(sec_part+fir_part) /psz(1)/psz(2);%original
d_ssim_x=2*(crop(sec_part+fir_part)) /psz(1)/psz(2);%modified by fws
d_ssim_x=d_ssim_x(:);
return