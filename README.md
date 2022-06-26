# segmentation-horse
semantic segmentationによる馬の領域抽出

PyTorch Hubで配布されている**DEEPLABV3**を使用。
semantic segmentationを用いてマスクを生成して馬の領域の抽出を行う。

<img src="./assets/desc.jpg" style="height:200px"></img>

## 仮想環境
```
conda env create --file env.yaml
```
- python:3.8.13
- pytorch:1.11.0
- numpy:1.22.3
- pillow:9.0.1
- opencv:4.0.1

## 参考サイト
- [DEEPLABV3](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/)
- [Deeplabv3-ResNet101でセマンティック・セグメンテーションをやってみる](https://kikaben.com/deeplabv3-resnet101-segmentation/)
- [OpenCV4Python10:OpenCV(numpy.ndarray)とPyTorch Hubで画像分類](https://note.com/fz5050/n/nfe3e087a8949)
- [Pythonでマスク画像を作る方法（３選）](https://water2litter.net/rum/post/python_image_mask/)
- [Semantic Segmentationの実装](https://qiita.com/MMsk0914/items/2f64a741e04b36cd1c76)