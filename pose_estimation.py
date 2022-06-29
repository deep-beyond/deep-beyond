import cv2
import urllib
import argparse
import numpy as np

# deep.pyのDeepSementationクラス
from deep import DeepSegmentation


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


def searchCnt(contours):
    """
    輪郭情報から頂点10以上の近似輪郭の座標を調査
    :param contours (type:list) マスクの輪郭情報
    :return cntPos (type:list) 頂点10以上の近似輪郭の座標リスト
    """
    # 輪郭の座標リスト
    cntPos = []

    for cnt in contours:
        # 輪郭線の長さ
        arclen = cv2.arcLength(cnt, True)

        # 輪郭線の近似
        approx = cv2.approxPolyDP(cnt, 0.01 * arclen, True)

        # 10頂点以上の輪郭に対して処理: hard code !
        if len(approx) > 9:
            # 近似輪郭線の各座標を調査
            for app in approx:
                x = app[0][0]
                y = app[0][1]
                pos = (x, y)
                cntPos.append(pos)

    return cntPos


def getWithers(cntPos, contours, img):
    """
    キ甲の座標情報を取得 & キ甲を描画
    :param cntPos (type:list) 頂点10以上の近似輪郭の座標リスト
    :param contours (type:list) マスクの輪郭情報
    :param img (type:numpy.ndarray) 入力画像と同じサイズの空の画像
    """
    # 画像の高さと幅を取得
    h, w = img.shape[0], img.shape[1]

    # 隣り合う座標のy座標の距離を記録
    distanceY = []

    for i in range(len(cntPos) - 1):
        _, y1 = cntPos[i]
        _, y2 = cntPos[i + 1]
        d = abs(y1 - y2)
        distanceY.append((d, i))

    # y座標の距離を大きい順にソート
    distanceY.sort(reverse=True)

    # 前足を探索するため画像の「左半分」だけを見る
    half_w = w // 2

    wither_posX = 0  # キ甲のX座標
    toesY = 0  # 足先の座標

    # y座標の距離が大きいもの上位3つを探索 hard code !!
    for i in range(3):
        _, idx = distanceY[i]
        pos1 = cntPos[idx]
        pos2 = cntPos[idx + 1]

        # 2点のx座標がどちらも画像の「左半分」の位置にある場合
        if pos1[0] <= half_w and pos2[0] <= half_w:
            toesY = max(pos1[1],pos2[1],toesY)  # 画像が下なほどYの値は大きい
            # pos1とpos2のどちらが上で下なのか
            if pos1[1] < pos2[1]:
                wither_posX += pos1[0]
            else:
                wither_posX += pos2[0]

            # 条件を満たす2点が形成する辺＝前足を沿う辺    を描画(赤色)
            # cv2.line(img, pos1, pos2, (0, 0, 255), thickness=4, lineType=cv2.LINE_AA)

    wither_posX //= 2  # キ甲のX座標：前足の辺の中点
    wither_pos = [] # キ甲を形成する座標

    # 輪郭とキ甲の直線の交点を探索
    for cnt in contours[0]:
        if wither_posX == cnt[0][0]:
            wither_pos.append([cnt[0][0], cnt[0][1]])
    
    # 足先に線分の頂点が位置していない場合、足先の座標に変更
    # 2点の中でY座標が大きい方の頂点がtoesYよりも小さければ変更
    if wither_pos[0][1] > wither_pos[1][1]:
        if wither_pos[0][1] < toesY:
            wither_pos[0][1] = toesY
    else:
        if wither_pos[1][1] < toesY:
            wither_pos[1][1] = toesY

    cv2.line(
        img,
        (wither_pos[0][0],wither_pos[0][1]),
        (wither_pos[1][0],wither_pos[1][1]),
        (0, 255, 255),
        thickness=3,
        lineType=cv2.LINE_AA,
    )

    cv2.putText(img,
                text=str(wither_pos[0]),
                org=(wither_pos[0][0], wither_pos[0][1]),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 255, 0),
                thickness=2,
                lineType=cv2.LINE_4)

    cv2.putText(img,
                text=str(wither_pos[1]),
                org=(wither_pos[1][0], wither_pos[1][1]),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 255, 0),
                thickness=2,
                lineType=cv2.LINE_4)

    return 


def main(args):
    # 画像読み込み
    img = getImg(args)

    # インスタンス生成(クラスの__init__メソッドを実行)
    ds = DeepSegmentation(args.colorPath, img)

    # クラスの__call__メソッドを実行
    resultImg, contours = ds()

    # 特定の条件を満たす輪郭の座標を取得
    cntPos = searchCnt(contours)

    # キ甲のx座標を探索
    getWithers(cntPos, contours, resultImg)

    # 画像を表示
    if args.display:
        ds.displayImg(resultImg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation")
    parser.add_argument(
        "--mode", type=str, choices=["net", "local"], default="net", help="入力画像先"
    )
    parser.add_argument(
        "--img_url",
        type=str,
        default="https://blogimg.goo.ne.jp/user_image/29/58/45dc07ba6673ee855e23253d6ff78098.jpg",
        help="入力画像URL",
    )
    parser.add_argument(
        "--fpath", type=str, default="./img/tokara_horse.jpg", help="ローカル上の画像パス"
    )
    parser.add_argument("--fname", type=str, default="horse.jpg", help="ダウンロードファイル名")
    parser.add_argument("--display", action="store_false", help="表示フラグ")
    parser.add_argument("--save", action="store_true", help="保存フラグ")
    parser.add_argument(
        "--colorPath", type=str, default="./color.txt", help="色情報ファイルのパス"
    )
    args = parser.parse_args()
    main(args)
