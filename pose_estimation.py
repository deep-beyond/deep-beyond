import cv2
import argparse
from copy import deepcopy

# deep.pyのDeepSementationクラス
from deep import DeepSegmentation

# utils.pyの関数
from utils import loadImg, drawText, drawLine, displayImg


def getContourVertex(contours, img, args):
    """
    輪郭情報から近似輪郭の座標を得る
    :param contours (type:list) マスクの輪郭情報
    :param img (type:numpy.ndarray) shape=(H,W,C,A) 画像情報
    :return contour_vertex (type:list) 近似輪郭の座標リスト
    :return bbox_position (type:tuple) 外接矩形の位置情報
    """
    # 輪郭の座標リスト
    contour_vertex = []

    h, w = img.shape[0], img.shape[1]

    scale = 0.6  # ノイズ除去のためのスケール比
    condition_bbox_x = 0  # 条件を満たす外接矩形のx座標
    condition_bbox_y = 0  # 条件を満たす外接矩形のy座標
    condition_bbox_h = 0  # 条件を満たす外接矩形の高さ
    condition_bbox_w = 0  # 条件を満たす外接矩形の幅

    for contour in contours:
        # 輪郭線の長さ
        arclen = cv2.arcLength(contour, True)

        # 外接矩形計算
        bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(contour)

        # 外接矩形が画像サイズに対して比較的小さい場合はノイズとして無視
        if h * scale < bbox_h and w * scale < bbox_w:

            if args.showinfo:
                # 条件を満たす外接矩形を描画
                cv2.rectangle(
                    img,
                    (bbox_x, bbox_y),
                    (bbox_x + bbox_w, bbox_y + bbox_h),
                    (0, 255, 0),
                    2,
                )

            # 輪郭線の近似,描画
            approx = cv2.approxPolyDP(contour, 0.005 * arclen, True)        # 0.01 original
            cv2.drawContours(img, [approx], -1, (0, 255, 255), 2)

            # 条件を満たす外接矩形の情報を記録
            condition_bbox_x = bbox_x
            condition_bbox_y = bbox_y
            condition_bbox_h = bbox_h
            condition_bbox_w = bbox_w

            if args.showinfo:
                # 外接矩形の左上・右下の点を描画
                cv2.circle(img, (bbox_x, bbox_y), 5, (255, 0, 255), thickness=-1)
                cv2.circle(
                    img,
                    (bbox_x + bbox_w, bbox_y + bbox_h),
                    5,
                    (255, 0, 255),
                    thickness=-1,
                )
                drawText(img, str((bbox_x, bbox_y)), bbox_x, bbox_y)
                drawText(
                    img,
                    str((bbox_x + bbox_w, bbox_y + bbox_h)),
                    bbox_x + bbox_w,
                    bbox_y + bbox_h,
                )
    else:
        # 近似輪郭線の各頂点座標を得る
        for v in approx:
            x = v[0][0]
            y = v[0][1]
            pos = (x, y)
            contour_vertex.append(pos)
        bbox_position = (
            condition_bbox_x,
            condition_bbox_y,
            condition_bbox_h,
            condition_bbox_w,
        )
        return contour_vertex, bbox_position


def getIntersection(wither_pos_x, bbox_y, bbox_h, img):
    """
    輪郭とキ甲の交点をアルファ値によって判断する
    :param wither_pos_x (type:int) キ甲のX座標
    :param bbox_y (type:int) 外接矩形y座標
    :param bbox_h (type:int) 外接矩形高さ
    :param img (type:numpy.ndarray) shape=(H,W,C,A) 説明用"ではない"画像
    :return wither_pos (type:list) 輪郭とキ甲の交点
    """
    alpha = [img[y,wither_pos_x][3] for y in range(bbox_y,bbox_y+bbox_h)]

    # 画像上部における交点
    low_y = alpha.index(255) + bbox_y

    # 画像下部における交点
    alpha.reverse()
    high_y = len(alpha) - alpha.index(255) + 1 + bbox_y

    return [[wither_pos_x, low_y],[wither_pos_x, high_y]]


def getWithersPosition(contour_vertex, bbox_position, rawimg, descimg, args):
    """
    キ甲の座標情報を取得 & キ甲を描画
    :param contour_vertex (type:list) 頂点10以上の近似輪郭の座標リスト
    :param bbox_position (type:tuple) 外接矩形座標(x,y,h,w)
    :param bbox_h (type:int) 外接矩形の高さ
    :param bbox_y (type:int) 外接矩形のY座標
    :param rawimg (type:numpy.ndarray) 生の入力画像
    :param descimg (type:numpy.ndarray) 説明する画像(テキストや直線の描画などに使用)
    :return wither_pos_x (type:int) キ甲のx座標
    :return wither_pos (type:list) キ甲を形成する2点の座標
    """
    bbox_x = bbox_position[0]
    bbox_y = bbox_position[1]
    bbox_h = bbox_position[2]
    bbox_w = bbox_position[3]

    """
    1. 探索範囲を設定
    """

    # 前足を探索するため外接矩形の「上部1/3～上部3/3」「左1/4~左4/4」の範囲を見る
    onethird_h = bbox_h // 3 + bbox_y
    quarter_w = bbox_w // 4 + bbox_x

    # 足先を判定するためのライン（外接矩形の下から0.1のライン）
    lowerline = int(bbox_y + bbox_h - bbox_h * 0.1)

    # 範囲のラインを描画
    if args.showinfo:
        drawLine(descimg, (0, onethird_h), (descimg.shape[1], onethird_h), (255, 0, 255))
        drawLine(descimg, (quarter_w, 0), (quarter_w, descimg.shape[0]), (255, 0, 255))
        drawLine(descimg, (0, lowerline), (descimg.shape[1], lowerline), (125, 125, 125))

    """
    2. キ甲のx座標を探索
    """

    # 時刻t-1における辺分の情報
    prev_distance_y = 0

    # 足先のX座標の頂点
    toes_pos_xs = []
    toes_pos_ys = []

    # -1 にしないとfor文最後の実行において終点で配列外参照になる
    for i in range(len(contour_vertex) - 1):
        x1, y1 = contour_vertex[i]  # 始点
        x2, y2 = contour_vertex[i + 1]  # 終点

        # 辺分が上部1/3の位置に属しているなら処理しない
        if y1 < onethird_h and y2 < onethird_h or x1 < quarter_w:
            continue

        if args.showinfo:
            drawLine(descimg,(x1,y1),(x2,y2),(255,0,0))
            drawText(descimg,str((x1,y1)),x1,y1)

        distance_x = x2 - x1
        distance_y = y2 - y1
        tilt = round(distance_y / distance_x, 1)

        if args.showinfo:
            drawText(descimg,"Tilt:{}".format(tilt),(x1+x2)//2,(y1+y2)//2)

        # 時刻t-1に線分が急激に右肩上がりになり、時刻tで傾きが弱まった場合
        if prev_distance_y < -bbox_h * (2 / 7) and abs(tilt) < 3:  # Hard code !!　　10
            break

        # 足先の頂点を探索
        if y1 > lowerline:
            toes_pos_xs.append(x1)
            toes_pos_ys.append(y1)
            cv2.circle(descimg, (x1, y1), 5, (0, 0, 255), thickness=-1)

        # 時刻tの情報を時刻t-1の情報にする
        prev_distance_y = distance_y

    assert len(toes_pos_xs) != 0, "can't look for toes vertex"

    # キ甲のX座標：前足の頂点らの中点
    wither_pos_x = sum(toes_pos_xs) // len(toes_pos_xs)

    # キ甲のラインを描画
    if args.showinfo:
        drawLine(
            descimg, (wither_pos_x, 0), (wither_pos_x, descimg.shape[0]), color=(255, 255, 255)
        )

    # if args.display:
    #     displayImg(descimg)

    """
    3. キ甲の長さを探索(キ甲と輪郭の交点を探索)
    """

    # 輪郭とキ甲の直線の交点を探索
    wither_pos = getIntersection(wither_pos_x, bbox_y, bbox_h, rawimg)

    """
    4. キ甲と輪郭の交点を修正
    """

    # 足先に線分の頂点が位置していない場合、足先の座標に変更
    # 2点の中でY座標が大きい方の頂点がtoesYよりも小さければ変更
    toes_pos_y = sum(toes_pos_ys) // len(toes_pos_ys)
    if wither_pos[0][1] > wither_pos[1][1]:
        if wither_pos[0][1] < toes_pos_y:
            wither_pos[0][1] = toes_pos_y
    else:
        if wither_pos[1][1] < toes_pos_y:
            wither_pos[1][1] = toes_pos_y
    
    # キ甲の正確なラインを描画
    if args.display:
        drawLine(descimg,tuple(wither_pos[0]),tuple(wither_pos[1]),(0,0,255))
        drawText(descimg, str(wither_pos[0]), wither_pos[0][0] + 20, wither_pos[0][1])
        drawText(descimg, str(wither_pos[1]), wither_pos[1][0] + 20, wither_pos[1][1])

    return wither_pos_x, wither_pos


def getTorso(contour_vertex, bbox_position, wither_pos_x, descimg, args):
    """
    胴を探索
    :param contour_vertex (type:list) 頂点10以上の近似輪郭の座標リスト
    :param bbox_position (type:tuple) 外接矩形座標(x,y,h,w)
    :param wither_pos_x (type:int) キ甲のx座標
    :param descimg (type:numpy.ndarray) 説明する画像(テキストや直線の描画などに使用)
    :return torso_pos_x (type:int) 胴のx情報
    """
    bbox_y = bbox_position[1]
    bbox_h = bbox_position[2]

    """
    1. 探索範囲を設定
    """

    # 胴の終点を探索するため外接矩形の「キ甲より右側（尻側）」「上側1/3」の範囲を見る
    onethird_h = bbox_h // 3 + bbox_y

    # 範囲のラインを描画
    if args.showinfo:
        drawLine(descimg, (0, onethird_h), (descimg.shape[1], onethird_h), (255, 0, 255))
        drawLine(descimg, (wither_pos_x, 0), (wither_pos_x, descimg.shape[0]), (255, 0, 255))

    """
    2. 胴のx座標を探索
    """
    # 時刻t-1における傾きとx座標の情報
    prev_tilt = 0
    prev_x = 0

    # 胴のX座標の頂点
    torso_pos_x = 0

    # 胴の終点であるフラグ
    torso_flg = False
    
    for i in range(len(contour_vertex)):
        x1, y1 = contour_vertex[i]  # 始点

        # 配列外参照にならないようにループさせる
        if i == len(contour_vertex) - 1:
            x2, y2 = contour_vertex[0]  # 終点
        else:
            x2, y2 = contour_vertex[i + 1]  # 終点

        # 「キ甲より右側（尻側）」「上側1/3」の範囲以外ならば処理しない
        if x1 < wither_pos_x or y1 > onethird_h:
            continue

        distance_x = x2 - x1
        distance_y = y2 - y1
        tilt = round(distance_y / distance_x, 1)
        # print(tilt)

        # 傾きが正->負->正になった場合、負の箇所を胴の終点とする
        if torso_flg and tilt > 0:
            torso_pos_x = prev_x
            break

        if tilt <= 0 and prev_tilt :
            torso_flg = True

        if args.showinfo:
            cv2.circle(descimg, (x1, y1), 5, (0, 0, 255), thickness=-1)

        # 時刻tの情報を時刻t-1の情報にする
        prev_tilt = tilt
        prev_x = x1

    # 胴の終点を通る縦の直線描画
    if args.showinfo:
        drawLine(
            descimg, (torso_pos_x, 0), (torso_pos_x, descimg.shape[0]), color=(255, 255, 255)
        )

    # 胴を描画
    if args.display:
        torso_front_pos = (wither_pos_x, int(descimg.shape[0]/2))
        torso_back_pos = (torso_pos_x, int(descimg.shape[0]/2))
        drawLine(descimg, torso_front_pos, torso_back_pos, color=(0, 0, 255))
        cv2.circle(descimg, (torso_back_pos[0], torso_back_pos[1]), 5, (0, 0, 255), thickness=-1)
        drawText(descimg, "x:"+str(torso_pos_x), torso_back_pos[0], torso_back_pos[1]-30)

    return torso_pos_x


def main(args):
    # 画像読み込み
    img = loadImg(mode=args.mode, img_url=args.img_url, img_path=args.img_path)

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
    wither_pos_x, wither_pos = getWithersPosition(contour_vertex, bbox_position, resultimg, descimg, args)
    print("キ甲の長さ:",wither_pos[1][1] - wither_pos[0][1])

    # 胴を探索
    torso_pos_x = getTorso(contour_vertex, bbox_position, wither_pos_x, descimg, args)
    print("胴の長さ:", torso_pos_x - wither_pos[0][0])

    if args.display:
        displayImg(descimg)  # 画像を表示


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
        "--img_path", type=str, default="./img/tokara_horse.jpg", help="ローカル上の画像パス"
    )
    parser.add_argument("--display", action="store_false", help="表示フラグ")
    parser.add_argument("--save", action="store_true", help="保存フラグ")
    parser.add_argument("--transparent", action="store_false", help="透過フラグ")
    parser.add_argument(
        "--save_format", choices=["jpg", "png"], default="png", help="保存形式"
    )
    parser.add_argument(
        "--color_path", type=str, default="./color.txt", help="色情報ファイルのパス"
    )
    parser.add_argument("--showinfo", action="store_true", help="詳細な情報を表示するか")
    args = parser.parse_args()

    assert (
        args.transparent and args.save_format == "png"
    ), "jpg format can't transparent"

    main(args)
