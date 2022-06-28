from copy import deepcopy
import cv2
import argparse
import urllib.request
import numpy as np

# 切り抜き用の左上位置・右下位置・フラグ
upleft = None
lowright = None
cutflg = False

# Enterキーコード
ENTER_KEY = 13  

# マウスが押されているかフラグ(ペイント時に使用)
mouseLpush = False
mouseRpush = False

# ループ終了フラグ
loopend = False


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

    img = cv2.imread("./img/tokara_horse.jpg")

    return img


def cutImg(img):
    """
    マウス操作によって画像を矩形に切り抜き
    """
    cv2.namedWindow("cutting")

    def mouseposition(event, x, y, flags, param):
        """
        マウスの位置座標を取得
        """
        global upleft
        global lowright
        global cutflg
        # 左マウスが押された時
        if event == cv2.EVENT_LBUTTONDOWN:
            upleft = (x, y)
        # 左マウスが離された時
        elif event == cv2.EVENT_LBUTTONUP:
            cutflg = True
        # 　左マウスが離される前まで
        elif not event == cv2.EVENT_LBUTTONUP and not cutflg:
            lowright = (x, y)

    cv2.setMouseCallback("cutting", mouseposition)
    showimg = img

    while 1:
        cv2.imshow("cutting", showimg)
        # Enterキーで処理終了
        if cv2.waitKey(1) == ENTER_KEY or cutflg:
            cv2.destroyAllWindows()
            break

        if (not upleft is None) and (not lowright is None) and not cutflg:
            showimg = deepcopy(img)  # 画像の上書きを防ぐ
            # バウンディングボックス選択中の描画
            cv2.rectangle(
                showimg,
                (upleft[0], upleft[1]),
                (lowright[0], lowright[1]),
                (0, 255, 0),
                2,
            )
    
    return img[upleft[1] : lowright[1], upleft[0] : lowright[0]]  # 切り抜き


def paint(img, originimg, mask):
    """
    GUIを用いて描画
    マウス左ボタン：黒色描画（誤検出部分をマスク）
    マウス右ボタン：白色描画（未検出部分をマスク）
    """

    # 画像合成(切り抜かれる部分とそうでない部分を明瞭化)
    showimg = cv2.addWeighted(src1=img, alpha=0.6, src2=originimg, beta=0.3, gamma=0)

    cv2.namedWindow("painting")

    def draw_circle(event, x, y, flags, param):
        global mouseLpush
        global mouseRpush
        global loopend

        # マウス左ボタンが押されている場合
        if mouseLpush:
            cv2.circle(showimg, (x, y), 10, (0, 0, 0), -1)
            cv2.circle(mask, (x, y), 10, (0, 0, 0), -1)
        # マウス右ボタンが押されている場合
        elif mouseRpush:
            cv2.circle(showimg, (x, y), 10, (255, 255, 255), -1)
            cv2.circle(mask, (x, y), 10, (255, 255, 255), -1)

        # マウス左ボタンが押されたら
        if event == cv2.EVENT_LBUTTONDOWN:
            mouseLpush = True
        # マウス右ボタンが押されたら
        elif event == cv2.EVENT_RBUTTONDOWN:
            mouseRpush = True
        elif event == cv2.EVENT_MBUTTONDOWN:
            loopend = True
            
        # それ以外
        elif flags == 0:
            mouseLpush = False
            mouseRpush = False

    cv2.setMouseCallback("painting", draw_circle)

    while 1:
        cv2.imshow("painting", showimg)
        # Enterキーで処理終了
        if cv2.waitKey(1) == ENTER_KEY or loopend:
            cv2.destroyAllWindows()
            break

    return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)


def grabcut(img, newmask, rect, bgdModel, fgdModel, mask_base):
    """
    GrabCutアルゴリズム実行
    :param img 入力画像
    :param newmask 適用するマスク
    :param rect 対象範囲
    :param bdgModel 背景配列(お約束)
    :param fgdModel 前景配列(お約束)
    :param mask_base 空のマスク(2回目のみ使用)
    """

    if not rect is None:
        cv2.grabCut(img, newmask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
    else:
        mask_base[newmask == 0] = 0     # 黒色は背景とする
        mask_base[newmask == 255] = 1   # 白色は前景とする
        newmask, bgdModel, fgdModel = cv2.grabCut(
            img, mask_base, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK
        )

    # 画素値が0と2の画素は全て0(背景)に，画素値が1と3の画素は全て1(前景)に変更
    mask = np.where((newmask == 2) | (newmask == 0), 0, 1).astype("uint8")

    # マスクを用いて切り抜き
    img = img * mask[:, :, np.newaxis]

    return img


def main(args):
    # 画像読み込み
    img = getImg(args)

    # 画像切り抜き
    croppedimg = cutImg(img)

    # 画像の高さ・幅を取得
    h, w = croppedimg.shape[:2]

    # 初期マスクを生成
    mask_init = np.zeros((h, w), dtype=np.uint8)
    newmask = mask_init

    # 対象矩形範囲を設定
    rect = (1, 1, w, h)

    # 背景/前景に関する配列(お約束)
    bgdModel = np.zeros((1, 65), np.float64)  
    fgdModel = np.zeros((1, 65), np.float64)


    while 1:
        # GrabCutアルゴリズム実行
        result = grabcut(croppedimg, newmask, rect, bgdModel, fgdModel, mask_init)

        # 新規空マスクを生成
        empty_mask = np.zeros((h, w, 3), np.uint8)
        empty_mask += 125  # 灰色

        # マスクを手作業で作成(白：未検出部分、黒:誤検出部分)
        newmask = paint(result, croppedimg, empty_mask)

        # ペイント時にマウス中ボタンを押すと終了
        if loopend:
            break

        # ２回目以降、rectは不要なため
        rect = None

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
    parser.add_argument("--fname", type=str, default="deepbond.jpg", help="ダウンロードファイル名")
    parser.add_argument("--display", action="store_false", help="表示フラグ")
    parser.add_argument("--save", action="store_true", help="保存フラグ")
    args = parser.parse_args()

    main(args)
