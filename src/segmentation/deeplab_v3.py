import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
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
    def __init__(self, img, color_path, transparent):
        """
        :param img (type:numpy.ndarray) shape=(H,W,C) 画像情報
        :param color_path (type:string) 分割色を記載したテキストファイルのパス
        """
        # カラーパレット(21クラス分): 馬のみ白、それ以外は黒
        color = open(color_path).read().strip().split("\n")
        color = [np.array(c.split(",")).astype("int") for c in color]
        self.color = np.array(color, dtype="uint8")

        # 画像情報
        self.img = img  # (H,W,3)

        # モデルを読み込み
        self.model = torch.jit.load("./assets/deeplabv3_scripted.pt")   # deeplabv3_resnet101

        # モデルを推論モードにする
        self.model.eval()

        # 透過処理フラグ
        self.transparent = transparent

    def __call__(self):
        """
        :return resultimg (type:numpy.ndarray) shape=(H,W,C) or shape=(H,W,C,A) GrabCutを適用した画像情報
        :return contours (type:list) マスクの輪郭情報
        """

        # 前処理
        inputTensor = preprocess(self.img)  # (3,H,W)

        # 形状変更(バッチを考慮)
        inputBatch = inputTensor.unsqueeze(0)  # (1,3,H,W)

        with torch.no_grad():
            outputs = self.model(inputBatch)  # [out],[aux]
        out = outputs["out"][0]  # (21,H,W) : 21 = クラス数

        # 予測確率が最大のものをmaskにする
        mask = out.argmax(0).byte().cpu().numpy()  # (H,W)

        # ノイズ除去
        mask = cv2.medianBlur(mask, 7)

        # マスクの輪郭を計算
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # numpy -> PIL
        maskPIL = Image.fromarray(mask)

        # 着色(colorで決定した色にする)
        maskPIL.putpalette(self.color)

        # PILをRGBモードにする
        maskPIL = maskPIL.convert("RGB")

        # PIL-> numpy
        mask = np.asarray(maskPIL)

        # マスク処理
        resultimg = cv2.bitwise_and(self.img, mask)

        # 透過処理
        if self.transparent:
            # RGBA形式に変更
            resultimg = cv2.cvtColor(resultimg, cv2.COLOR_BGR2BGRA)
            # resultimgのアルファチャンネルを上書き
            resultimg[:, :, 3] = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        return resultimg, contours


def main(args):
    # 画像読み込み
    img = loadImg(mode=args.mode, img_url=args.img_url, img_path=args.img_path)

    # インスタンス生成(クラスの__init__メソッドを実行)
    ds = DeepSegmentation(img, args.color_path, args.transparent)
    # クラスの__call__メソッドを実行
    resultimg, contours = ds()

    if args.display:
        displayImg(resultimg)

    if args.save:
        if args.save_format == "png":
            cv2.imwrite("./result.png", resultimg)
        else:
            cv2.imwrite("./result.jpg", resultimg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation")
    parser.add_argument(
        "--mode", type=str, choices=["net", "local"], default="local", help="入力画像先"
    )
    parser.add_argument(
        "--img_url",
        type=str,
        default="https://blogimg.goo.ne.jp/user_image/29/58/45dc07ba6673ee855e23253d6ff78098.jpg",
        help="入力画像URL",
    )
    parser.add_argument(
        "--img_path", type=str, default="./assets/tokara_horse.jpg", help="ローカル上の画像パス"
    )
    parser.add_argument(
        "--color_path", type=str, default="./assets/color.txt", help="色情報ファイルのパス"
    )
    parser.add_argument("--display", action="store_false", help="表示フラグ")
    parser.add_argument("--save", action="store_true", help="保存フラグ")
    parser.add_argument("--transparent", action="store_false", help="透過フラグ")
    parser.add_argument(
        "--save_format", choices=["jpg", "png"], default="png", help="保存形式"
    )
    args = parser.parse_args()

    assert (
        args.transparent and args.save_format == "png"
    ), "jpg format can't transparent"

    main(args)
