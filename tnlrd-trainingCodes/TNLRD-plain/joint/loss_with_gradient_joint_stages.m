function [loss, grad] = loss_with_gradient_joint_stages(vcof)
global TP

TP.iter = TP.iter + 1;

tic;
noisy          = TP.F_NOISE;
clean          = TP.G_TRUTH;
basis          = TP.basis;
nbasis         = TP.nbasis;
neig           = TP.NEIG;
wei            = TP.WEI;
stage          = TP.stage;
fsz            = TP.fsz;
m              = fsz^2 - 1;
filtN          = TP.filtN;
fnlsz          = TP.fnlsz;
mtxPT          = TP.PT;
PT             = TP.PT;
T              = TP.T;
mfs            = TP.MFS;
NumW           = mfs.NumW;

R              = size(noisy,2);
r              = sqrt(size(clean,1)) + (fsz + 1)*2;
c              = r;

trained_model  = cell(stage,1);
len_cof        = (filtN)^2 + 1 + NumW*filtN + (fnlsz-1)*filtN;
vcof           = reshape(vcof,len_cof,stage);

%% intialize models parameters
for s = 1:stage
    cof       = vcof(:,s);
    part1     = cof(1:filtN*m);
    cof_beta  = reshape(part1,m,filtN);
    part2     = cof(filtN*m+1);
    p         = exp(part2);
    part3     = cof(filtN*m+2:filtN*m+1+NumW*filtN);
    weights   = reshape(part3,NumW,filtN);
    part4     = cof(filtN*m+1+NumW*filtN+1:end);
    nlweights = reshape(part4,(fnlsz-1),filtN);
    
    K         = cell(filtN,1);
    f_norms   = zeros(filtN,1);
    nlK       = zeros(fnlsz,filtN);
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
        nlK(:,i)     = filter(:);
    end
    %% update mfs
    MFsALL = updateMFs(mfs, weights, filtN);    
    %% construct model for one stage
    model.cof_beta   = cof_beta;
    model.MFsALL     = MFsALL;
    model.K          = K;
    model.p          = p;
    model.mfs        = mfs;
    model.f_norms    = f_norms;
    model.nlweights  = nlweights;
    model.nlcoef     = nlK;
    model.nlf_norms  = nlf_norms;
    trained_model{s} = model;
end

% make sure not to use temporary variables
clear cof_beta MFsALL K p f_norms nlweights nlK nlf_norms;

%% do a gradient descent step for all samples
AllNetWorks = cell(stage,1);
for s = 1:stage
    result.u = zeros(size(noisy));
    result.PTu = zeros(size(noisy));
    result.grad_l_u = zeros(size(noisy));
    AllNetWorks{s} = result;
end

%% forward step
input = noisy;
for s = 1:stage
    model  = trained_model{s};
    MFsALL = model.MFsALL;
    K      = model.K;
    p      = model.p;  
    nlcoef = model.nlcoef;
    out_u  = zeros(size(noisy));
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
        out_u(:,samp) = u - g(:);
    end
    AllNetWorks{s}.u   = out_u;
    AllNetWorks{s}.PTu = PT*out_u;
    input              = AllNetWorks{s}.PTu;
end
x_s  = AllNetWorks{stage}.u;
loss = sum(sum((T*x_s - clean).^2))/R;
fprintf('Iter: %03d\tLoss: %.6f\t', TP.iter, loss);

%% derivative of loss wrt x
AllNetWorks{stage}.grad_l_u = 2/R*T'*(T*x_s-clean);

%% caluculate gradients of loss w.r.t us of each stage
for s = stage-1:-1:1
    model       = trained_model{s+1};
    MFsALL      = model.MFsALL;
    K           = model.K;
    p           = model.p;
    nlcoef      = model.nlcoef;
    grad_l_usp1 = AllNetWorks{s+1}.grad_l_u;
    PTu_s       = AllNetWorks{s}.PTu;
    
    grad_l_us   = zeros(size(noisy));
    parfor samp = 1:R
        u       = PTu_s(:,samp);
        ane     = neig{samp};
        we      = wei{samp};
        % part 1
        part1 = (1-p)*grad_l_usp1(:,samp);
        % part 2
        part2 = 0;
        for i=1:filtN
            ca    = nlcoef(:,i);
            a     = sparse(repmat(1:numel(u),fnlsz,1), ane, diag(ca)*we);
            
            k     = K{i};
            t     = convolution_transposeOMP(grad_l_usp1(:,samp),rot90(rot90(k)),r,c);
            t     = a*t;
            Ku    = imfilter(reshape(u,r,c),k,'symmetric');
            Ku    = a*Ku(:);
            Ne2   = lut_eval_one_variable(Ku(:)', mfs.offsetD, mfs.step, MFsALL{i}.GX);
            t     = a'*(t(:).*Ne2(:));
            part2 = part2 + convolution_transposeOMP(t,k,r,c);
        end
        grad_l_us(:,samp) = mtxPT'*(part1 - part2);
    end
    AllNetWorks{s}.grad_l_u = grad_l_us;
end

%% calculate gradients of loss w.r.t training parameters
grad_l_paras = zeros(len_cof,stage);
for s = 1:stage
    if s == 1
        input = noisy;
    else
        input = AllNetWorks{s-1}.PTu;
    end
    grad_l_paras(:,s) = ...
        grad_l_thetas(AllNetWorks, trained_model, input, s, TP);
end
grad = grad_l_paras(:);
T=toc;
fprintf('Runtime: %.6f seconds.\n',T);