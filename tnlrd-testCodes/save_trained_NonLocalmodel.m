function trained_model = save_trained_NonLocalmodel(TP)
basis  = TP.basis;
nbasis = TP.nbasis;
fsz    = TP.fsz;
m      = fsz^2 - 1;
filtN  = TP.filtN;
fnlsz  = TP.fnlsz;
stage  = TP.stage;
mfs    = TP.MFS;
NumW   = mfs.NumW;

trained_model = cell(stage,1);

for s = 1:stage
    vcof      = TP.cof(:,s);
    part1     = vcof(1:filtN*m);
    cof_beta  = reshape(part1,m,filtN);
    part2     = vcof(filtN*m+1);
    p         = exp(part2);
    part3     = vcof(filtN*m+2:filtN*m+1+NumW*filtN);
    weights   = reshape(part3,mfs.NumW,filtN);
    part4     = vcof(filtN*m+1+NumW*filtN+1:end);
    nlweights = reshape(part4,fnlsz,filtN);
    
    K = cell(filtN,1);
    f_norms   = zeros(filtN,1);
    nlK       = zeros(fnlsz,filtN);
    nlf_norms = zeros(filtN,1);
    for i = 1:filtN
        x_cof      = cof_beta(:,i);
        filter     = basis*x_cof;
        f_norms(i) = norm(filter);
        filter     = filter/(norm(filter) + eps);
        K{i}       = reshape(filter,fsz,fsz);
        
        x_cof        = nlweights(:,i);
        filter       = nbasis*x_cof;
%         nlf_norms(i) = norm(x_cof);
        nlf_norms(i) = norm(filter);
        filter       = filter/(norm(filter) + eps);
        nlK(:,i)     = filter(:);
    end
    %% update mfs
    MFsALL = cell(filtN,1);
    for i=1:filtN
        w           = weights(:,i);
        Q           = bsxfun(@times, mfs.G, w);
        MFsALL{i}.P = sum(Q,1);
    end
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
