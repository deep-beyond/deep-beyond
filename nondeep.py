import cv2
import argparse
import urllib.request
import numpy as np

# 二値化 上限/下限
upper = (180, 255, 255)
lower = (0, 0, 0)

# 色定義
colors = np.zeros(3)


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
    # インターネットから画像をダウンロード
    urllib.request.urlretrieve(args.img_url, args.fname)
    # 画像を読み込み
    img = cv2.imread(args.fname)  # (H,W,3)

    return img


def maskgenerator(img, h, s, v, flg, originimg):
    """
    マスク生成を行う
    :param img 画像データ
    :param h 色相
    :param s 彩度
    :param v 明度
    :param flg 上限設定 or 下限設定
    """

    global upper
    global lower
    if flg:
        upper = (h, s, v)
    else:
        lower = (h, s, v)

    # 二値化
    img = cv2.inRange(img, lower, upper)  # [480,640]

    #### memo:ゴマ塩ノイズ除去(メディアン処理)　-> マスクが曖昧になり必要がなかった

    # 輪郭抽出
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 空のマスク生成
    mask = np.zeros_like(originimg)

    # もし輪郭が存在すれば
    if len(contours) > 0:
        # 輪郭の中で最大の面積のものを取得
        contour = max(contours, key=lambda x: cv2.contourArea(x))

        # マスクを描画
        cv2.drawContours(mask, [contour], -1, color=(255, 255, 255), thickness=-1)

    return mask


def dummy(x):
    """ダミー関数"""
    pass


def trackbar(bgrimg, hsvimg):
    """
    マスクを生成するためのトラックバー
    """

    ENTER_KEY = 13  # Enterキーコード

    # 'window'ウィンドウでパラメーター処理
    cv2.namedWindow("window")
    cv2.createTrackbar("H", "window", 0, 180, dummy)
    cv2.createTrackbar("S", "window", 0, 255, dummy)
    cv2.createTrackbar("V", "window", 0, 255, dummy)
    cv2.createTrackbar("L(0),U(1)", "window", 0, 1, dummy)  # 上限(1)下限(0)

    cv2.namedWindow("image")
    global colors

    def getColor(event, x, y, flags, param):
        """
        画像の色を取得
        """
        # ダブルクリックした場合
        if event == cv2.EVENT_LBUTTONDBLCLK:
            global colors
            colors = hsvimg[y, x, :]  # 色取得

    # 'image'ウィンドウでマウス処理
    cv2.setMouseCallback("image", getColor)

    upper_defo = [180, 255, 255]
    lower_defo = [0, 0, 0]
    flg_defo = 1    
    
    # 上限設定をデフォルト
    cv2.setTrackbarPos("L(0),U(1)", "window",flg_defo)
    cv2.setTrackbarPos("H", "window", upper_defo[0])
    cv2.setTrackbarPos("S", "window", upper_defo[1])
    cv2.setTrackbarPos("V", "window", upper_defo[2])

    showimg = maskgenerator(hsvimg, 0, 0, 0, 0, bgrimg)

    while 1:
        # 画像表示
        cv2.imshow("window", showimg)
        cv2.imshow("image", hsvimg)

        # Enterキーで処理終了
        if cv2.waitKey(1) == ENTER_KEY:
            break

        # パラメーター値取得
        h = cv2.getTrackbarPos("H", "window")
        s = cv2.getTrackbarPos("S", "window")
        v = cv2.getTrackbarPos("V", "window")
        flg = cv2.getTrackbarPos("L(0),U(1)", "window")

        # フラグが変更された場合:上限または下限の値をセット
        if not flg_defo == flg:
            if flg:
                cv2.setTrackbarPos("H", "window", upper_defo[0])
                cv2.setTrackbarPos("S", "window", upper_defo[1])
                cv2.setTrackbarPos("V", "window", upper_defo[2])
            else:
                cv2.setTrackbarPos("H", "window", lower_defo[0])
                cv2.setTrackbarPos("S", "window", lower_defo[1])
                cv2.setTrackbarPos("V", "window", lower_defo[2])
            flg_defo = flg

        # パラメーターの値が変更された場合、二値化処理
        if flg:
            if not upper_defo == [h, s, v]:
                showimg = maskgenerator(hsvimg, h, s, v, flg, bgrimg)
                upper_defo = [h, s, v]
        else:
            if not lower_defo == [h, s, v]:
                showimg = maskgenerator(hsvimg, h, s, v, flg, bgrimg)
                lower_defo = [h, s, v]

        # マウスがダブルクリックされた場合：該当ピクセルの値に置換
        if colors.all():
            h, s, v = int(colors[0]), int(colors[1]), int(colors[2])
            if flg:
                upper_defo = [h, s, v]
                cv2.setTrackbarPos("H", "window", upper_defo[0])
                cv2.setTrackbarPos("S", "window", upper_defo[1])
                cv2.setTrackbarPos("V", "window", upper_defo[2])
            else:
                lower_defo = [h, s, v]
                cv2.setTrackbarPos("H", "window", lower_defo[0])
                cv2.setTrackbarPos("S", "window", lower_defo[1])
                cv2.setTrackbarPos("V", "window", lower_defo[2])

            print("取得値：",colors)
            showimg = maskgenerator(hsvimg, h, s, v, flg, bgrimg)
            colors = np.zeros(3)

    cv2.destroyAllWindows()
    return showimg


def main(args):
    # 画像読み込み
    bgrimg = getImg(args)

    # HSV化
    hsvimg = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2HSV)

    # マスク生成（トラックバーによる手動生成）
    mask = trackbar(bgrimg, hsvimg)

    # マスク処理
    result = cv2.bitwise_and(bgrimg, mask)

    if args.display:
        displayImg(result)
    
    if args.save:
        cv2.imwrite("./result.jpg", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation")
    parser.add_argument(
        "--img_url",
        type=str,
        default="https://blogimg.goo.ne.jp/user_image/29/58/45dc07ba6673ee855e23253d6ff78098.jpg",
        help="入力画像URL",
    )
    parser.add_argument("--fname", type=str, default="deepbond.jpg", help="保存ファイル名")
    parser.add_argument("--display", action="store_false", help="表示フラグ")
    parser.add_argument("--save", action="store_true", help="保存フラグ")
    args = parser.parse_args()

    main(args)
