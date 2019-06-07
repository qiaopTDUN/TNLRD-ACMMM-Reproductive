function [loss, grad] = loss_with_gradient_unit_filters_LUT(vcof)
global TP

TP.iter = TP.iter + 1;

tic;
noisy     = TP.F_NOISE;
clean     = TP.G_TRUTH;
input     = TP.INPUT;
basis     = TP.basis;
nbasis    = TP.nbasis;
neig      = TP.NEIG;
wei       = TP.WEI;
fsz       = TP.fsz;
m         = fsz^2 - 1;
filtN     = TP.filtN;
fnlsz     = TP.fnlsz;
T         = TP.T;
mfs       = TP.MFS;
NumW      = mfs.NumW;

R         = size(noisy,2);
r         = sqrt(size(clean,1)) + (fsz + 1)*2;
c         = r;

%% intialize models parameters
cof       = vcof(:);
part1     = cof(1:filtN*m);
cof_beta  = reshape(part1,m,filtN);
part2     = cof(filtN*m+1);
p         = exp(part2);
part3     = cof(filtN*m+2:filtN*m+1+NumW*filtN);
weights   = reshape(part3,NumW,filtN);
part4     = cof(filtN*m+1+NumW*filtN+1:end);
nlweights = reshape(part4,fnlsz,filtN);

K         = cell(filtN,1);
f_norms   = zeros(filtN,1);
nlcoef    = zeros(fnlsz,filtN);
nlf_norms = zeros(filtN,1);
for i = 1:filtN
    x_cof        = cof_beta(:,i);
    filter       = basis*x_cof;
%         f_norms(i)   = norm(x_cof);
    f_norms(i)   = norm(filter);
    filter       = filter/(norm(filter) + eps);
    K{i}         = reshape(filter,fsz,fsz);

    x_cof        = nlweights(:,i);
    filter       = nbasis*x_cof;
%         nlf_norms(i) = norm(x_cof);
    nlf_norms(i) = norm(filter);
    filter       = filter/(norm(filter) + eps);
    nlcoef(:,i)  = filter(:);
end
%% update mfs
MFsALL = updateMFs(mfs, weights, filtN);    

%% do a gradient descent step for all samples
x = zeros(size(input));
parfor samp = 1:R
    u   = input(:,samp);
    f   = noisy(:,samp);
    g   = (u - f)*p;
    g   = reshape(g,r,c);
    ane = neig{samp};
    we  = wei{samp};
    for i=1:filtN
        ca  = nlcoef(:,i);
        a   = sparse(repmat(1:numel(u),fnlsz,1), ane, diag(ca)*we);

        Ku  = imfilter(reshape(u,r,c),K{i},'symmetric');
        Ku  = a*Ku(:);
        Ne1 = lut_eval_one_variable(Ku(:)', mfs.offsetD, mfs.step, MFsALL{i}.P);
        Ne1 = reshape(Ne1,r,c);
        Ne1 = reshape(a'*Ne1(:),r,c);
        g   = g + imfilter(Ne1,rot90(rot90(K{i})),'symmetric');
    end
    x(:,samp) = u - g(:);
end
loss          = sum(sum((T*x - clean).^2))/R;
fprintf('Iter: %03d\tLoss: %.6f\t', TP.iter, loss);

%% derivative of loss wrt x
grad_l_x      = 2/R*T'*(T*x-clean);

%% caluculate gradients of loss w.r.t us of each stage
grad_loss_beta            = 0;
grad_loss_weights         = 0;
grad_loss_nlweights       = 0;

pd                        = (fsz-1)/2;
ndims                     = size(noisy, 1);
parfor samp = 1:R
    grad_ln_beta          = zeros(m,filtN);
    grad_ln_ws            = zeros(NumW,filtN);
    grad_ln_nlws          = zeros((fnlsz-1),filtN);
    x                     = reshape(input(:,samp),r,c);
    xl                    = padarray(x, [pd,pd], 'both', 'symmetric');
    v                     = reshape(grad_l_x(:,samp),r,c);
    ane                   = neig{samp};
    we                    = wei{samp};
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
grad_loss_p = -p*sum(sum((input-noisy).*grad_l_x));
grad = [grad_loss_beta(:);grad_loss_p;grad_loss_weights(:);grad_loss_nlweights(:)];
T=toc;
fprintf('Runtime: %.6f seconds.\n',T);
