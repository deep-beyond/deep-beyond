import torch
import urllib
import argparse
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2

# 前処理設定
preprocess = transforms.Compose(
    [
        transforms.ToPILImage(),  # Pillowオブジェクトに変換
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# カラーパレット(21クラス分): 馬のみ白、それ以外は黒
color = [
    0,  0,   0,
    0,  0,   0,
    0,  0,   0,
    0,  0,   0,
    0,  0,   0,
    0,  0,   0,
    0,  0,   0,
    0,  0,   0,
    0,  0,   0,
    0,  0,   0,
    0,  0,   0,
    0,  0,   0,
    0,  0,   0,
    255, 255, 255,   # 馬
    0,  0,   0,
    0,  0,   0,
    0,  0,   0,
    0,  0,   0,
    0,  0,   0,
    0,  0,   0,
    0,  0,   0,
]

def displayImg(img):
    """
    画像を表示
    """
    cv2.imshow("display image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def getImg(args):
    """
    :return 画像データ(ndarray)
    """
    if args.mode == "net":
        # インターネットから画像をダウンロード
        urllib.request.urlretrieve(args.img_url, args.fname)
        # 画像を読み込み
        img = cv2.imread(args.fname)  # (H,W,3)

    elif args.mode == "local":
        img = cv2.imread(args.fpath)
    
    return img


def main(args):
    # モデルをダウンロード
    model = torch.hub.load(
        "pytorch/vision:v0.10.0", "deeplabv3_resnet50", pretrained=True
    )

    # モデルを推論モードにする
    model.eval()

    # 画像読み込み
    img = getImg(args)

    # 前処理
    input_tensor = preprocess(img)  # (3,H,W)

    # 形状変更(バッチを考慮)
    input_batch = input_tensor.unsqueeze(0)  # (1,3,H,W)

    # gpuがあるならば適用
    if torch.cuda.is_available():
        device = torch.device("cuda")
        input_batch = input_batch.to(device)
        model.to(device)

    # 推論：モデル実行
    with torch.no_grad():
        outputs = model(input_batch)  # [out],[aux]
    out = outputs["out"][0]  # (21,H,W) : 21 = クラス数

    # 予測確率が最大のものをmaskにする
    mask = out.argmax(0).byte().cpu().numpy()  # (H,W)

    # numpy -> PIL
    maskPIL = Image.fromarray(mask)

    # 着色(colorで決定した色にする)
    maskPIL.putpalette(color)

    # PILをRGBモードにする
    maskPIL = maskPIL.convert('RGB')

    # PIL-> numpy
    mask = np.asarray(maskPIL)

    # マスク画像を表示
    if args.display:
        displayImg(mask)

    # マスク処理
    result = cv2.bitwise_and(img, mask)

    # マスクで切り抜いた画像を表示
    if args.display:
        displayImg(result)
    
    # 保存
    if args.save:
        cv2.imwrite("./result.jpg", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation")
    parser.add_argument("--mode",type=str,choices=['net', 'local'],default='local',help="入力画像先")
    parser.add_argument(
        "--img_url",
        type=str,
        default="https://prtimes.jp/i/21266/9/resize/d21266-9-237312-1.jpg",
        help="入力画像URL",
    )
    parser.add_argument("--fpath",type=str,default="./img/tokara_horse.jpg",help="ローカル上の画像パス")
    parser.add_argument("--fname", type=str, default="horse.jpg", help="保存ファイル名")
    parser.add_argument("--display", action="store_false", help="表示フラグ")
    parser.add_argument("--save", action="store_true", help="保存フラグ")
    args = parser.parse_args()
    main(args)
