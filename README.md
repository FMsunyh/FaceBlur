
# 图片/视频人脸模糊

该仓库是做一个图片和视频的人脸模糊应用。在原有的两个仓库基础上做了修改：
原有的仓库：
- [yolov7-face-blur](https://github.com/FareedKhan-dev/yolov7-face-blur) 人脸检测
- [face-alignment](https://github.com/1adrianb/face-alignment) 关键点检测



## 机器
zjai-dev-011

## 环境搭建
```sh
conda create -n faceblur python==3.9.0 -y
```

```sh
conda activate faceblur
```

```sh
pip install -r requirements
pip install -r requirements_gpu.txt
```

```sh
pip install face-alignment
```

## 运行
1. `--weights` contains your downloaded model from the above table
2. `--bluurratio` to determine the strength of blurring, default value is `20`
3. `--device` either `cpu` or `0` in case of `gpu`.
4. `--source` containes your images (png, jpg etc) or video (mp4 etc)

图片
```sh
python detect.py --weights weights/yolov7-tiny.pt --blurratio 50 --device 0 --source data/images/22_Picnic_Picnic_22_10.jpg

```
视频
```sh
python detect.py --weights weights/yolov7-tiny.pt  --hidedetarea --blurratio 50 --device 0 --source data/JIA/1.mp4
```
