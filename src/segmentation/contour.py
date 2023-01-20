import os
import sys
import cv2
import argparse
import numpy as np

# utils.pyの関数
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import loadImg, displayImg


class ContourSegmantation:
    def __init__(self, img, transparent=True):
        """
        :param img (type:numpy.ndarray) shape=(H,W,C) 画像情報
        :param transparent (type:bool) 透過処理フラグ
        """
        # 二値化 上限/下限
        self.upper = (180, 255, 255)
        self.lower = (0, 0, 0)

        # 色定義
        self.colors = np.zeros(3)  # BGR

        # 画像定義
        self.bgrimg = img
        self.hsvimg = None

        # 透過処理フラグ
        self.transparent = transparent

        # パラメーター値
        self.h_para = 180
        self.s_para = 255
        self.v_para = 255
        self.flg_para = 1  # 上限設定=1, 下限設定=0

    def __call__(self):
        """
        :return resultimg (type:numpy.ndarray) shape=(H,W,C) or shape=(H,W,C,A) GrabCutを適用した画像情報
        :return contours (type:list) マスクの輪郭情報
        """
        # HSV化
        self.hsvimg = cv2.cvtColor(self.bgrimg, cv2.COLOR_BGR2HSV)

        # マスク生成（トラックバーによる手動生成）
        mask = self.trackbar()

        # マスク処理（入力画像とマスクを合成）
        resultimg = cv2.bitwise_and(self.bgrimg, mask)

        # 透過処理
        if self.transparent:
            # RGBA形式に変更
            resultimg = cv2.cvtColor(resultimg, cv2.COLOR_BGR2BGRA)
            # resultimgのアルファチャンネルを上書き
            resultimg[:, :, 3] = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # マスクの輪郭を計算
        mask = cv2.cvtColor(self.bgrimg, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return resultimg, contours

    def maskgenerator(self, img):
        """
        マスク生成を行う
        :param img (type:numpy.ndarray) shape=(H,W,C) 画像データ
        :return mask (type:numpy.ndarray) shape=(H,W,C) マスク
        """

        # 上限値、下限値を設定
        if self.flg_para:
            self.upper = (self.h_para, self.s_para, self.v_para)
        else:
            self.lower = (self.h_para, self.s_para, self.v_para)

        # 二値化
        img = cv2.inRange(img, self.lower, self.upper)

        #### memo:ゴマ塩ノイズ除去(メディアン処理)　-> マスクが曖昧になり必要がなかった

        # 輪郭抽出
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # 空のマスク生成
        mask = np.zeros_like(self.bgrimg)

        # もし輪郭が存在すれば
        if len(contours) > 0:
            # 輪郭の中で最大の面積のものを取得
            contour = max(contours, key=lambda x: cv2.contourArea(x))
            # マスクを描画
            cv2.drawContours(mask, [contour], -1, color=(255, 255, 255), thickness=-1)

        return mask

    def trackbar(self):
        """
        マスクを生成するためのトラックバー
        """

        def dummy(_):
            """ダミー関数"""
            pass

        ENTER_KEY = 13  # Enterキーコード

        # 'window'ウィンドウでパラメーター処理
        cv2.namedWindow("window", cv2.WINDOW_NORMAL)  # ウィンドウサイズ：可変
        cv2.createTrackbar("H", "window", 0, 180, dummy)
        cv2.createTrackbar("S", "window", 0, 255, dummy)
        cv2.createTrackbar("V", "window", 0, 255, dummy)
        cv2.createTrackbar("L(0),U(1)", "window", 0, 1, dummy)  # 上限(1)下限(0)

        cv2.namedWindow("image")

        def getColor(event, x, y, flags, param):
            """
            画像の色を取得
            """
            # ダブルクリックした場合
            if event == cv2.EVENT_LBUTTONDBLCLK:
                self.colors = self.hsvimg[y, x, :]  # 色取得

        # 'image'ウィンドウでマウス処理
        cv2.setMouseCallback("image", getColor)

        # 上限設定をデフォルトとしてデフォルト値設定
        upper_defo = [self.h_para, self.s_para, self.v_para]
        lower_defo = [0, 0, 0]
        cv2.setTrackbarPos("L(0),U(1)", "window", self.flg_para)
        cv2.setTrackbarPos("H", "window", upper_defo[0])
        cv2.setTrackbarPos("S", "window", upper_defo[1])
        cv2.setTrackbarPos("V", "window", upper_defo[2])

        showimg = np.ones(self.bgrimg.shape, np.uint8) * 255

        # 変更前のフラグ
        prev_flag = self.flg_para

        while 1:
            cv2.imshow("window", showimg)
            cv2.imshow("image", self.hsvimg)

            if cv2.waitKey(1) == ENTER_KEY:  # Enterキーで処理終了
                break

            # パラメーター値取得
            self.h_para = cv2.getTrackbarPos("H", "window")
            self.s_para = cv2.getTrackbarPos("S", "window")
            self.v_para = cv2.getTrackbarPos("V", "window")
            self.flg_para = cv2.getTrackbarPos("L(0),U(1)", "window")

            # フラグが変更された場合:上限または下限の値をセット
            if not prev_flag == self.flg_para:
                if self.flg_para:
                    cv2.setTrackbarPos("H", "window", upper_defo[0])
                    cv2.setTrackbarPos("S", "window", upper_defo[1])
                    cv2.setTrackbarPos("V", "window", upper_defo[2])
                else:
                    cv2.setTrackbarPos("H", "window", lower_defo[0])
                    cv2.setTrackbarPos("S", "window", lower_defo[1])
                    cv2.setTrackbarPos("V", "window", lower_defo[2])
                prev_flag = self.flg_para

            # パラメーターの値が変更された場合、二値化処理
            if self.flg_para:
                if not upper_defo == [self.h_para, self.s_para, self.v_para]:
                    showimg = self.maskgenerator(self.hsvimg)
                    upper_defo = [self.h_para, self.s_para, self.v_para]
            else:
                if not lower_defo == [self.h_para, self.s_para, self.v_para]:
                    showimg = self.maskgenerator(self.hsvimg)
                    lower_defo = [self.h_para, self.s_para, self.v_para]

            # マウスがダブルクリックされた場合：該当ピクセルの値に置換
            if self.colors.all():
                self.h_para, self.s_para, self.v_para = (
                    self.colors[0],
                    self.colors[1],
                    self.colors[2],
                )
                if self.flg_para:
                    upper_defo = [self.h_para, self.s_para, self.v_para]
                    cv2.setTrackbarPos("H", "window", upper_defo[0])
                    cv2.setTrackbarPos("S", "window", upper_defo[1])
                    cv2.setTrackbarPos("V", "window", upper_defo[2])
                else:
                    lower_defo = [self.h_para, self.s_para, self.v_para]
                    cv2.setTrackbarPos("H", "window", lower_defo[0])
                    cv2.setTrackbarPos("S", "window", lower_defo[1])
                    cv2.setTrackbarPos("V", "window", lower_defo[2])

                print("取得値：", self.colors)
                showimg = self.maskgenerator(self.hsvimg)
                self.colors = np.zeros(3)

        cv2.destroyAllWindows()
        return showimg


def main(args):
    # 画像読み込み
    bgrimg = loadImg(mode=args.mode, img_url=args.img_url, img_path=args.img_path)

    cs = ContourSegmantation(bgrimg)  # インスタンス生成(クラスの__init__メソッドを実行)
    resultimg, contours = cs()  # クラスの__call__メソッドを実行

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
