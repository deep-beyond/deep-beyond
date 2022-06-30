import cv2
import torch
import urllib
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms
from utils import loadImg, displayImg

# 前処理設定
preprocess = transforms.Compose(
    [
        transforms.ToPILImage(),  # Pillowオブジェクトに変換
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

class DeepSegmentation:
    def __init__(self,colorPath,imgData):
        """
        :param colorPath (type:string) 分割色を記載したテキストファイルのパス
        :param imgData (type:numpy.ndarray) 画像情報
        """
        # カラーパレット(21クラス分): 馬のみ白、それ以外は黒
        color = open(colorPath).read().strip().split("\n")
        color = [np.array(c.split(",")).astype("int") for c in color]
        self.color = np.array(color, dtype="uint8")

        # 画像情報
        self.img =imgData   # (H,W,3)

        # モデルをダウンロード
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0", "deeplabv3_resnet50", pretrained=True
        )
        # モデルを推論モードにする
        self.model.eval()
    
    def __call__(self):
        """
        :return resultimg (type:numpy.ndarray) マスクで切り抜かれた入力画像
        :return contours (type:list) マスクの輪郭情報
        """
        # 前処理
        inputTensor = preprocess(self.img)  # (3,H,W)

        # 形状変更(バッチを考慮)
        inputBatch = inputTensor.unsqueeze(0)  # (1,3,H,W)

        # gpuがあるならば適用
        if torch.cuda.is_available():
            device = torch.device("cuda")
            inputBatch = inputBatch.to(device)
            self.model.to(device)

        # 推論：モデル実行
        with torch.no_grad():
            outputs = self.model(inputBatch)  # [out],[aux]
        out = outputs["out"][0]  # (21,H,W) : 21 = クラス数

        # 予測確率が最大のものをmaskにする
        mask = out.argmax(0).byte().cpu().numpy()  # (H,W)

        # ノイズ除去
        mask = cv2.medianBlur(mask,7)

        # マスクの輪郭を計算
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # numpy -> PIL
        maskPIL = Image.fromarray(mask)

        # 着色(colorで決定した色にする)
        maskPIL.putpalette(self.color)

        # PILをRGBモードにする
        maskPIL = maskPIL.convert('RGB')

        # PIL-> numpy
        mask = np.asarray(maskPIL)

        # マスク処理
        resultImg = cv2.bitwise_and(self.img, mask)

        return resultImg, contours

def main(args):
    # 画像読み込み
    img = loadImg(args)
    
    # インスタンス生成(クラスの__init__メソッドを実行)
    ds = DeepSegmentation(args.colorPath,img)
    # クラスの__call__メソッドを実行
    resultImg, contours = ds()

    displayImg(resultImg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation based Deep Learning")
    parser.add_argument("--mode",type=str,choices=['net', 'local'],default='local',help="入力画像先")
    parser.add_argument(
        "--imgUrl",
        type=str,
        default="https://prtimes.jp/i/21266/9/resize/d21266-9-237312-1.jpg",
        help="入力画像URL",
    )
    parser.add_argument("--imgPath",type=str,default="./img/tokara_horse.jpg",help="ローカル上の画像パス")
    parser.add_argument("--colorPath",type=str,default="./color.txt",help="色情報ファイルのパス")
    args = parser.parse_args()
    main(args)
