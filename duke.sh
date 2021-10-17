# train CAP model on DukeMTMC
CUDA_VISIBLE_DEVICES=6 python train_cap.py --target 'dukemtmcreid' --data_dir '/data0/ReIDData/' --logs_dir 'Supervised_Duke_logs_lr_5'

# test model on Market1501
CUDA_VISIBLE_DEVICES=6 python train_cap.py --target 'market1501' --data_dir '/data0/ReIDData/' --logs_dir 'Supervised_DukeTrainMarketTest_lr_5' --evaluate --load_ckpt 'Supervised_Duke_logs_lr_5/final_model_epoch_50.pth'

# test model on MSMT
CUDA_VISIBLE_DEVICES=6 python train_cap.py --target 'MSMT17_V1' --data_dir '/data0/ReIDData/' --logs_dir 'Supervised_DukeTrainMSMTTest_lr_5' --evaluate --load_ckpt 'Supervised_Duke_logs_lr_5/final_model_epoch_50.pth'