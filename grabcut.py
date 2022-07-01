import cv2
import argparse
import numpy as np
from copy import deepcopy

# utils.pyの関数
from utils import loadImg, displayImg


class GrabcutSegmentation:
    def __init__(self, img, transparent=True):
        """
        :param img (type:numpy.ndarray) shape=(H,W,C) 画像情報
        :param transparent (type:bool) 透過処理フラグ
        """
        # 切り抜き用の左上位置・右下位置・フラグ
        self.top_left_pos = None
        self.btm_right_pos = None
        self.release_left_click_flg = False

        # Enterキーコード
        self.ENTER_KEY = 13

        # マウスが押されているかフラグ(ペイント時に使用)
        self.mouse_left_click_flg = False
        self.mouse_right_click_flg = False

        # ループ終了フラグ
        self.loop_end_flag = False

        # 画像情報
        self.img = img

        # 初期状態のマスク
        self.init_mask = None

        # 透過処理フラグ
        self.transparent = transparent

    def __call__(self):
        """
        :return resultimg (type:numpy.ndarray) shape=(H,W,C) or shape=(H,W,C,A) GrabCutを適用した画像情報
        :return contours (type:list) マスクの輪郭情報
        """
        # 画像切り抜き
        self.cropImg()

        displayImg(self.img)

        # 画像の高さ・幅を取得
        h, w = self.img.shape[:2]

        # 初期マスクを生成
        self.init_mask = np.zeros((h, w), dtype=np.uint8)
        mask = self.init_mask

        # 対象矩形範囲を設定
        rect = (1, 1, w, h)

        # 背景/前景に関する配列(お約束)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        while 1:
            # GrabCutアルゴリズム実行
            resultimg = self.grabcut(self.img, mask, rect, bgdModel, fgdModel)

            # 新規空マスクを生成(画像合成のため入力画像の形状と同じにする)
            empty_mask = np.zeros(self.img.shape, np.uint8)
            empty_mask += 125  # 下地を灰色とする

            # マスクを手作業で作成(白：未検出部分、黒:誤検出部分)
            mask = self.paint(resultimg, empty_mask)

            # ペイント時にマウスホイールを押すと終了
            if self.loop_end_flag:
                break

            # 2回目以降、rectは不要なためNone
            rect = None

        # 透過処理
        if self.transparent:
            # RGBA形式に変更
            resultimg = cv2.cvtColor(resultimg, cv2.COLOR_BGR2BGRA)
            # resultimgのアルファチャンネルを上書き
            resultimg[:, :, 3] = mask

        # マスクの輪郭を計算
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return resultimg, contours


    def cropImg(self):
        """
        マウス操作によって画像を矩形に切り抜き
        [操作方法]
            マウス左ボタンを押し続けて切り抜き範囲を指定
            マウス左ボタンを離すと処理終了
        """
        cv2.namedWindow("cropping")

        def mouseposition(event, x, y, flags, param):
            """
            マウスの位置座標を取得
            """
            # マウス左ボタンが押された時
            if event == cv2.EVENT_LBUTTONDOWN:
                self.top_left_pos = (x, y)
            # マウス左ボタンが離された時
            elif event == cv2.EVENT_LBUTTONUP:
                self.release_left_click_flg = True
            # マウス左ボタンが離される前まで
            elif not event == cv2.EVENT_LBUTTONUP and not self.release_left_click_flg:
                self.btm_right_pos = (x, y)

        cv2.setMouseCallback("cropping", mouseposition)
        showimg = self.img

        while 1:
            cv2.imshow("cropping", showimg)
            # エンターキーが押された場合、マウス左ボタンが離された場合に処理終了
            if cv2.waitKey(1) == self.ENTER_KEY or self.release_left_click_flg:
                break

            # 選択されていない状態もしくは選択している状態ならば
            if not self.top_left_pos is None:
                showimg = deepcopy(self.img)  # 画像の上書きを防ぐ
                # バウンディングボックス選択中の描画
                cv2.rectangle(
                    showimg,
                    (self.top_left_pos[0], self.top_left_pos[1]),
                    (self.btm_right_pos[0], self.btm_right_pos[1]),
                    (0, 255, 0),
                    2,
                )
        cv2.destroyAllWindows()

        # 切り抜き
        self.img = self.img[
            self.top_left_pos[1] : self.btm_right_pos[1],
            self.top_left_pos[0] : self.btm_right_pos[0],
        ]


    def grabcut(self, img, mask, rect, bgdModel, fgdModel):
        """
        GrabCutアルゴリズム実行
        :param img (type:numpy.ndarray) shape=(H,W,C) 入力画像情報
        :param mask (type:numpy.ndarray) shape=(H,W) 適用するマスク
        :param rect (type:tuple) 対象領域
        :param bdgModel (type:numpy.ndarray) shape=(1,65) 背景配列(お約束)
        :param fgdModel (type:numpy.ndarray) shape=(1,65) 前景配列(お約束)
        :return img (type:numpy.ndarray) shape=(H,W,C) マスクを用いて切り抜いた画像情報
        """

        # 処理が2回目ならば
        if not rect is None:
            cv2.grabCut(
                img, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT
            )
        else:
            # 処理が1回目の場合
            self.init_mask[mask == 0] = 0  # 黒色は背景とする
            self.init_mask[mask == 255] = 1  # 白色は前景とする
            mask, bgdModel, fgdModel = cv2.grabCut(
                img, self.init_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK
            )

        # 画素値が0と2の画素は全て0(背景)に，画素値が1と3の画素は全て1(前景)に変更
        mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

        # マスクを用いて切り抜き
        img = img * mask[:, :, np.newaxis]

        return img


    def paint(self,img, mask):
        """
        GUIを用いてGrubCutを適用した画像を修正(新規マスクを手動生成)
        [操作方法]
            マウス左ボタン：黒色描画（誤検出部分をマスク）
            マウス右ボタン：白色描画（未検出部分をマスク)
        :param img (type:numpy.ndarray) shape=(H,W,C) GrubCutを適用した画像情報
        :param mask (type:numpy.ndarray) shape=(H,W,C) 新規マスク画像情報
        :return mask (type:numpy.ndarray) shape=(H,W) 手動で生成したマスク
        (注意) 入力のマスクは画像合成をするためにBGR形式、一方出力のマスクはGrubCutに使用するためグレースケール
        """

        # 画像合成(切り抜かれる部分とそうでない部分を明瞭化)
        showimg = cv2.addWeighted(
            src1=img, alpha=0.6, src2=self.img, beta=0.3, gamma=0
        )

        cv2.namedWindow("painting")

        def draw_circle(event, x, y, flags, param):
            # マウス左ボタンが押されている場合
            if self.mouse_left_click_flg:
                cv2.circle(showimg, (x, y), 10, (0, 0, 0), -1)
                cv2.circle(mask, (x, y), 10, (0, 0, 0), -1)
            # マウス右ボタンが押されている場合
            elif self.mouse_right_click_flg:
                cv2.circle(showimg, (x, y), 10, (255, 255, 255), -1)
                cv2.circle(mask, (x, y), 10, (255, 255, 255), -1)

            # マウス左ボタンが押されたら
            if event == cv2.EVENT_LBUTTONDOWN:
                self.mouse_left_click_flg = True
            # マウス右ボタンが押されたら
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.mouse_right_click_flg = True
            # マウスホイールが押されたら
            elif event == cv2.EVENT_MBUTTONDOWN:
                self.loop_end_flag = True
            # それ以外
            elif flags == 0:
                self.mouse_left_click_flg = False
                self.mouse_right_click_flg = False

        cv2.setMouseCallback("painting", draw_circle)

        while 1:
            cv2.imshow("painting", showimg)
            # Enterキーが押された場合、もしくはマウスホイールが押された場合に処理終了
            if cv2.waitKey(1) == self.ENTER_KEY or self.loop_end_flag:
                break

        cv2.destroyAllWindows()
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        return mask


def main(args):
    # 画像読み込み
    img = loadImg(mode=args.mode, img_url=args.img_url, img_path=args.img_path)

    gs = GrabcutSegmentation(img,args.transparent)  # インスタンス生成(クラスの__init__メソッドを実行)
    resultimg, contours = gs()  # クラスの__call__メソッドを実行

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
        "--img_path", type=str, default="./img/tokara_horse.jpg", help="ローカル上の画像パス"
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
