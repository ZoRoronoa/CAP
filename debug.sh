# train CAP model on Market-1501
CUDA_VISIBLE_DEVICES=0 python train_cap.py --target 'market1501' --data_dir '/data0/ReIDData/' --logs_dir 'Market_debug'