This is TNLRD-ACMMM-Reproductive for MM'17 'Learning Non-local Image Diffusion for Image Denoising'
----FoETrainingSets180: training images
----testCodes: contains test codes for experiments described in [1]. Run test68tnlrd.m
----trainingCodes: contains training codes for experiments described in [1].
--------TNLRD-plain: greedy training + joint training, parameters are initialized randomly.
------------greedy : greedy training, run GreedyTraining.m
------------joint  : joint training, run JointTraining.m
--------TNLRD-tnrd : joint training using the parameters in TNRD [2] models. Run JointTraining.m
--------TNLRD-ssim : training with ssim-like loss instead of l2 loss. Run JointTrainingwithSSIM.m
               
              
[1] Peng Qiao, Yong Dou, Wensen Feng, Rongchun Li, and Yunjin Chen. 2017. Learning Non-local Image Diﬀusion for Image Denoising. In ACM Multimedia. 1847–1855.
[2] Yunjin Chen, Wei Yu, and Thomas Pock. 2015. On learning optimized reaction diﬀusion processes for eﬀective image restoration. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 5261–5269.
