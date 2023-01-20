import cv2
import csv
import yaml
import argparse
from copy import deepcopy

# deep.pyのDeepSementationクラス
from segmentation.deeplab_v3 import DeepSegmentation

# utils.pyの関数
from utils import loadImg, displayImg, update_values

# getHorseInfo.pyの関数
from getHorseInfo import getContourVertex, getHindlimb, getJumpsuit, getNeck, getTorso, getWithersPosition

def main(args):
    if args.mode == 'net':
        inputs = args.img_url
    else:
        inputs = [args.img_path]
    
    # 結果を格納するリスト("ファイル名", "キ甲", "胴", "とも", "首","繋(長さ)","繋(角度)")
    result = []

    for name in inputs:
        print(name)

        # 画像読み込み
        if args.mode == 'net':
            img = loadImg(mode=args.mode, img_url=name)
        else:
            img = loadImg(mode=args.mode, img_path=name)
        
        if img.shape[0] < 600:
            print(img.shape[:2])
            print("画像高さ600px未満: 画像が小さすぎます")
            continue

        # インスタンス生成(クラスの__init__メソッドを実行)
        ds = DeepSegmentation(img, args.color_path, args.transparent)
        # クラスの__call__メソッドを実行
        resultimg, contours = ds()  # resultimg shape=(H,W,C,A)

        # 説明するための画像(範囲のラインや座標値テキストの描画などに使用)
        descimg = deepcopy(resultimg)

        # 特定の条件を満たす輪郭の座標を取得
        contour_vertex, bbox_position = getContourVertex(contours, descimg, args)

        if args.showinfo:
            cv2.drawContours(descimg, contours, -1, (255, 255, 0), 3)  # 輪郭描画
        

        # キ甲を探索
        wither_pos_x, wither_pos, last_toes_pos_x = \
            getWithersPosition(contour_vertex, bbox_position, resultimg, descimg, args)
        wither_length = wither_pos[1][1] - wither_pos[0][1]
        print("キ甲の長さ:", wither_length)

        # 胴を探索
        torso_pos_x = getTorso(contour_vertex, bbox_position, wither_pos_x, descimg, args)
        torso_length = torso_pos_x - wither_pos[0][0]
        print("胴の長さ:", torso_length)

        # 同期処理 (並列処理しない方が高速だった)
        neck_length = getNeck(contour_vertex, wither_pos, descimg, args)
        fetlock_length, fetlock_tilt = getJumpsuit(torso_pos_x, bbox_position, deepcopy(resultimg), descimg, args)
        hindlimb_length = getHindlimb(torso_pos_x, bbox_position, contour_vertex, last_toes_pos_x, deepcopy(resultimg), descimg, args)

        print("ともの長さ:", hindlimb_length)        
        print("首の長さ:", neck_length)
        print("繋の長さ:", fetlock_length)   
        print("繋の傾き:", fetlock_tilt)

        # 結果を保存
        result.append([name, wither_length, torso_length, hindlimb_length, neck_length, fetlock_length, fetlock_tilt])

        if args.display:
            displayImg(descimg)  # 画像を表示

    if args.csv:
        # csvファイルに書き込む
        with open("./result.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ファイル名", "キ甲", "胴", "とも", "首", "繋(長さ)", "繋(角度)"]) # ヘッダー内容
            writer.writerows(result)
        print("writed result.csv file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation")
    parser.add_argument(
        "--mode", type=str, choices=["net", "local"], default="net", help="入力画像先"
    )
    parser.add_argument(
        "--img_url",
        nargs='+',
        default="[https://blogimg.goo.ne.jp/user_image/51/48/a4f2767dcdda226304984ab5fd510435.jpg]",
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
    parser.add_argument(
        "--color_path", type=str, default="./assets/color.txt", help="色情報ファイルのパス"
    )
    parser.add_argument("--showinfo", action="store_true", help="詳細な情報を表示するか")
    parser.add_argument("--csv", action="store_false", help="結果をcsvファイルに保存するか")
    args = parser.parse_args()

    assert (
        args.transparent and args.save_format == "png"
    ), "jpg format can't transparent"

    # ymlファイルに記述された引数で更新
    with open("./src/input.yml", 'r') as handle:
        options_yaml = yaml.load(handle, Loader=yaml.SafeLoader)
    update_values(options_yaml, vars(args))

    main(args)
