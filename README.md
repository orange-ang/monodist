# monodist
based on litemono

first download the pretrained-weights

training like:

python train.py  --mypretrain pretrain_weights/lite-mono-pretrain.pth --teacher_da3_path pretrained/model.safetensors --teacher_unidepth_path pretrained/pytorch_model.bin --model_name monodist_640_0926

change the data_path,file_dir,log_dir in option.py
