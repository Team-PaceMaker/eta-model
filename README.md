# ETA - Model

## PyTorch model on Flask

### versions
```
blinker==1.7.0
certifi==2023.7.22
charset-normalizer==3.3.2
click==8.1.7
Flask==3.0.0
idna==3.4
importlib-metadata==6.8.0
itsdangerous==2.1.2
Jinja2==3.1.2
MarkupSafe==2.1.3
numpy==1.26.1
Pillow==10.1.0
pytorch-pretrained-vit==0.0.7
requests==2.31.0
torch==1.13.0
torchvision==0.14.0
typing_extensions==4.8.0
urllib3==2.0.7
Werkzeug==3.0.1
zipp==3.17.0
```

### Port Number
`5001`

### How to use
Please Check [inference.ipynb](./inference.ipynb)

### Pretrained Weight

Model was trained with `python 3.8.x`, `pytoch 1.14`

```
$ git clone https://github.com/Team-PaceMaker/eta-model.git
$ conda create -n ETA_model python=3.8
$ conda activate ETA_model
$ pip install -r requirements.txt 
```

You can download pre-trained weight this [link](https://drive.google.com/file/d/1vbggo2VEdDI-L5q8aGCn3408sLTSYr3X/view?usp=drive_link)

### Model Output
![result](./img/output.gif)
