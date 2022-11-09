# Faster RCNN Detection example.

``` 
Dependence:
torch, torchvision, pycocotools, einops
```
### Dependence Install.
```
python==3.8.0
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
# linux
pip install pycocotools==2.0.0
# windows
pip install pycocotools-windows
pip install einops
```

The tickets of to train a detection model are follows:
1. Using the weights of backbone pretrained by IMAGENET-1K, It is good to fix the problem of the AP is 0 in testing model. 
2. Training the CNNs backbone need freeze the first stage.
But Training the Swin backbone doesn't have to freeze any weight.
3. Setting the anchor's aspect ratio of anchor generator need as far as possible to include all aspect of target. 
If your 'loss_rpn_box_reg' is much bigger than other losses in training, you need think to find a more suitable parameter.