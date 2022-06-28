# segmentation-horse

## 非深層学習ベース
### 二値化・輪郭抽出による馬の領域抽出
[ソースコード：extract_contour.py](./extract_contour.py)

トラックバーによって手動でマスクを設定

<img src="./assets/desc.jpg" style="height:200px"></img><br>
図１：非深層学習手法による出力結果比較

【トラックバー説明】<br>
**マスクを生成**

パラメーターによって二値化の範囲を設定
- 画面上部に操作パラメーター、画面下部に生成されるマスク
- HSVパラメーターと"モード"のパラメーターが存在
- "モード"は二値化の上限値or下限値の設定の2種類
- 二値化上限値設定モードがデフォルト
- Enterキーを押すとマスクを確定
- imageウィンドウに描画される図を**ダブルクリック**すると該当ピクセルの色を取得して二値化のパラメーターに反映

【注意】
マスクは被写体が白、それ以外は黒になるように生成する必要

<img src="./assets/trackbar.jpg" style="height:400px"></img><br>
図２：トラックバー

<img src="./assets/getcolor.jpg" style="height:300px"></img><br>
図３：imageウィンドウにおける色取得(表示される画像はHSV画像)


### GrabCutによる馬の領域抽出
[ソースコード：grabcut.py](./grabcut.py)

<img src="./assets/desc2.jpg" style="height:200px"></img><br>
図４：GrabCutアルゴリズムによる出力結果比較

1. バウンディングボックスによる領域選択

    <img src="./assets/cutimg.jpg" style="height:300px"></img><br>
    図５：アルゴリズム適用領域選択

2. ペイント操作によって領域修正<br>
    白色：未検出部分、黒色：誤検出部分<br>
    マウス左クリックで黒色、マウス右クリックで白色
    マウスホイール部をクリックして全体の処理を終了

    <img src="./assets/paint.jpg" style="height:300px"></img><br>
    図６：GrabCutアルゴリズムの出力に対する領域修正

【注意】全体の処理を終了させるにはペイント操作時にマウスホイール部をクリックする必要


## 深層学習ベース
[ソースコード：deep.py](./deep.py)

semantic segmentationによる馬の領域抽出

PyTorch Hubで配布されている**DEEPLABV3**を使用。
semantic segmentationを用いてマスクを生成して馬の領域の抽出を行う。

<img src="./assets/desc3.jpg" style="height:200px"></img><br>
図３：深層学習手法による出力結果比較

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
- [GrabCutを使った対話的前景領域抽出](http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_grabcut/py_grabcut.html)
- [DEEPLABV3](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/)
- [Deeplabv3-ResNet101でセマンティック・セグメンテーションをやってみる](https://kikaben.com/deeplabv3-resnet101-segmentation/)
- [OpenCV4Python10:OpenCV(numpy.ndarray)とPyTorch Hubで画像分類](https://note.com/fz5050/n/nfe3e087a8949)
- [Pythonでマスク画像を作る方法（３選）](https://water2litter.net/rum/post/python_image_mask/)
- [Semantic Segmentationの実装](https://qiita.com/MMsk0914/items/2f64a741e04b36cd1c76)