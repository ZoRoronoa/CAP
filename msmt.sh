# train CAP model on MSMT17
CUDA_VISIBLE_DEVICES=7 python train_cap.py --target 'MSMT17_V1' --data_dir '/data0/ReIDData/' --logs_dir 'Supervised_MSMT_logs_lr_5'

# test model on DukeMTMC
CUDA_VISIBLE_DEVICES=7 python train_cap.py --target 'dukemtmcreid' --data_dir '/data0/ReIDData/' --logs_dir 'Supervised_MSMTTrainDukeTest_lr_5' --evaluate --load_ckpt 'Supervised_MSMT_logs_lr_5/final_model_epoch_50.pth'

# test model on Market
CUDA_VISIBLE_DEVICES=7 python train_cap.py --target 'market1501' --data_dir '/data0/ReIDData/' --logs_dir 'Supervised_MSMTTrainMarketTest_lr_5' --evaluate --load_ckpt 'Supervised_MSMT_logs_lr_5/final_model_epoch_50.pth'