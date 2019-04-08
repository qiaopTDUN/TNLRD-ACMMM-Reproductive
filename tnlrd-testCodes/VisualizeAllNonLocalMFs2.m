clear all; 

% load JointTraining_7x7x5_400_stage=5.mat;
load JointTraining_7x7x5_7x7_400_stage=5_sigma=25.mat;

%% MFs means and precisions
trained_model = save_trained_NonLocalmodel(TrainedModels);
stage = TrainedModels.stage;
fsz = TrainedModels.fsz;
m = fsz^2 - 1;
filtN = TrainedModels.filtN;
BASIS   = TrainedModels.basis;
KernelPara.fsz = fsz;
KernelPara.filtN = filtN;
KernelPara.basis = BASIS;
MFS = TrainedModels.MFS;
for s = 1:stage
	model = trained_model{s};
    K     = model.K;
    nlK   = model.nlcoef;
    K1    = zeros(fsz^2, filtN);
    
    for i = 1:filtN
        k = K{i};
        K1(:,i) = k(:);
    end
    
    map  = colormap(gray(256));
    imgk = dictionary2imageColor(K1, nlK, map);
    imwrite(imgk,['s' num2str(s) '.png'])
%      for i=1:filtN
%         sfigure(s*100+i);
%         subplot(1,2,1);
%         plot(MFS.D, mfsAll{i}.P);grid on;drawnow;
%         
%         rho = RecoverRho(mfsAll{i}.P, MFS.step, length(MFS.D));
%         subplot(1,2,2);
%         plot(MFS.D, rho);grid on;drawnow;
%     end
end