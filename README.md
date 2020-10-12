# Pytorch-Utils
Pytorch utils for computer vision tasks.



### Useful Commands
```bash
# Test on custom model
python test.py --weights_path checkpoints/yolov3_ckpt_66.pth --model_def config/yolov3-custom.cfg --data_config config/custom.data --batch_size 8

# Detect on custom model
python detect.py --image_folder data/samples/ --weights_path checkpoints/yolov3_ckpt_66.pth --model_def config/yolov3-custom.cfg
```

