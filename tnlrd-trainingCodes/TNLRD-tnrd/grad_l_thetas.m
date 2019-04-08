function grad = grad_l_thetas(AllNetWorks, trained_model, input, s, TP)
mfs      = TP.MFS;
NumW     = mfs.NumW;
fsz      = TP.fsz;
m        = fsz^2 - 1;
filtN    = TP.filtN;
fnlsz    = TP.fnlsz;
basis    = TP.basis;
nbasis   = TP.nbasis;
neig     = TP.NEIG;
wei      = TP.WEI;
noisy    = TP.F_NOISE;
R        = size(noisy,2);
pd       = (fsz-1)/2;
ndims    = size(noisy,1);
r        = sqrt(ndims);
c        = r;

grad_l_u = AllNetWorks{s}.grad_l_u;
model    = trained_model{s};
MFsALL   = model.MFsALL;
K        = model.K;
p        = model.p;
cof_beta = model.cof_beta;
f_norms  = model.f_norms;
nlcoef   = model.nlcoef;
nlweights = model.nlweights;
nlf_norms = model.nlf_norms;

grad_loss_beta      = 0;
grad_loss_weights   = 0;
grad_loss_nlweights = 0;

parfor samp = 1:R
    grad_ln_beta         = zeros(m,filtN);
    grad_ln_ws           = zeros(NumW,filtN);
    grad_ln_nlws         = zeros(fnlsz,filtN);
    x                    = reshape(input(:,samp),r,c);
    xl                   = padarray(x, [pd,pd], 'both', 'symmetric');
    v                    = reshape(grad_l_u(:,samp),r,c);
    ane                  = neig{samp};
    we                   = wei{samp};
    for i=1:filtN
        k                 = K{i};
        ca                = nlcoef(:,i);
        a                 = sparse(repmat(1:ndims,fnlsz,1), ane, diag(ca)*we);
        %% part 1
        kx1               = imfilter(x,k,'symmetric');
        kx                = a*kx1(:);
        
        [Nkx,GW,N2kx]     = lut_eval(kx(:)', mfs.offsetD, mfs.step, MFsALL{i}.P, mfs.G, MFsALL{i}.GX, 0);
        Nkx               = reshape(Nkx,r,c);
        N2kx              = reshape(N2kx,r,c);
        
        t1                = convolution_transposeOMP(v,rot90(rot90(k)),r,c);
        t                 = a*t1;
        temp1             = N2kx.*reshape(t,r,c);
        temp              = reshape(a'*temp1(:),r,c);
        p1                = conv2(xl,rot90(rot90(temp)),'valid');
        %% part 2
        Nkx1              = reshape(a'*Nkx(:),r,c);
        Nkxl              = padarray(Nkx1, [pd pd], 'both', 'symmetric');
        p2                = conv2(Nkxl,rot90(rot90(v)),'valid');

        gk                = p1 + rot90(rot90(p2));
        
        grad_ln_beta(:,i) = -(eye(m) - cof_beta(:,i)*cof_beta(:,i)'/f_norms(i)^2)/f_norms(i)*basis'*gk(:);
        
        grad_ln_ws(:,i)   = -GW*t(:);

        p1                = repmat(Nkx(:),1,fnlsz).*t1(ane)';
        kx1               = kx1(:);
        p2                = repmat(t(:).*N2kx(:),1,fnlsz).*kx1(ane)';
 
        gnlk              = - (diag(we*p1) + diag(we*p2));
        grad_ln_nlws(:,i) = (eye(fnlsz) - nlweights(:,i)*nlweights(:,i)'/nlf_norms(i)^2)/nlf_norms(i)*nbasis'*gnlk(:);
    end
    grad_loss_beta        = grad_loss_beta + grad_ln_beta;
    grad_loss_weights     = grad_loss_weights + grad_ln_ws;
    grad_loss_nlweights   = grad_loss_nlweights + grad_ln_nlws;
end

grad_loss_p = -p*sum(sum((input - noisy).*grad_l_u));
grad        = [grad_loss_beta(:); grad_loss_p; grad_loss_weights(:); grad_loss_nlweights(:)];