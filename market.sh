# train CAP model on Market-1501
CUDA_VISIBLE_DEVICES=0 python train_cap.py --target 'market1501' --data_dir '/data0/ReIDData/' --logs_dir 'Supervised_Market_logs_lr_5'

# test model on DukeMTMC
CUDA_VISIBLE_DEVICES=0 python train_cap.py --target 'dukemtmcreid' --data_dir '/data0/ReIDData/' --logs_dir 'Supervised_MarketTrainDukeTest_lr_5' --evaluate --load_ckpt 'Supervised_Market_logs_lr_5/final_model_epoch_50.pth'

# test model on MSMT
CUDA_VISIBLE_DEVICES=0 python train_cap.py --target 'MSMT17_V1' --data_dir '/data0/ReIDData/' --logs_dir 'Supervised_MarketTrainDukeMSMT_lr_5' --evaluate --load_ckpt 'Supervised_Market_logs_lr_5/final_model_epoch_50.pth'