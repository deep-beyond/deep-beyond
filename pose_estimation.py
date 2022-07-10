import cv2
import numpy as np
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
        # drawText(descimg, "x:"+str(torso_pos_x), torso_back_pos[0], torso_back_pos[1]-30)

    return torso_pos_x


def getHindlimb(torso_pos_x, descimg, bbox_position, img, args):
    """
    ともを探索
    :param
    """

    """
    1. 探索範囲を設定
    """
    bbox_x = bbox_position[0]
    bbox_y = bbox_position[1]
    bbox_h = bbox_position[2]
    bbox_w = bbox_position[3]

    # 画像の尻部分のみ着目
    img = img[bbox_y : bbox_y+int(bbox_h/2), torso_pos_x: bbox_x+bbox_w]
    display_img = img
    h, w = img.shape[:2]

    # コントラスト調整（二値化、エッジ強調のため)（閾値決定が重要）
    img = cv2.convertScaleAbs(img, alpha = 4.5)
    # displayImg(img)

    # グレースケール化
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ヒストグラム平坦化(エッジ強調のため) -> 強調されすぎてしまった
    # img = cv2.equalizeHist(img)
    # displayImg(img)

    # 二値化（閾値決定が重要）
    img = cv2.bilateralFilter(img,9,75,75)  # エッジ残しながらブラー
    # displayImg(img)

    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 20)
    # displayImg(img)

    # 線膨張（線をはっきりさせる） メディアンフィルタが効果的になる
    img = cv2.bitwise_not(img)
    kernel = np.ones((3,3),np.uint8)        # (3,3)
    img = cv2.dilate(img,kernel,iterations = 1)
    img = cv2.bitwise_not(img)
    # displayImg(img)

    # 画像中、最も右下にある黒色ピクセルを探索
    right_x =0
    for x in range(w-1,0,-1):
        if np.all(img[h-1][x] <=0):
            right_x = x
            break

    # 画像の枠を入れない輪郭を生成
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    contour = np.array([],dtype=np.int32)
    draw_flg = False
    for cnt in contours[0]:
        x,y = cnt[0][0],cnt[0][1]
        # print(x,y)
        if right_x - x < 5:
            draw_flg = True

        if draw_flg:
            contour = np.append(contour, cnt)
            # cv2.circle(img, (x, y), 5, (255, 0, 255), thickness=-1)
            # displayImg(img)
    contour = contour.reshape([-1,1,2])

    # 馬の輪郭を白色にする
    img= cv2.drawContours(img, contour, -1, (255,255,255), 7)
    # displayImg(img)

    # メディアンフィルタ
    img = cv2.medianBlur(img, ksize=5)  # ksize >= 7 の場合情報が欠落してしまう
    # displayImg(img)
    for _ in range(3):
        img = cv2.medianBlur(img, ksize=3)
    # displayImg(img)

    # BGR化(探索場所を明瞭に赤色にするため)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # displayImg(img)

    """
    2. 尻の先端部のx座標を探索
    """

    hip_pos_xs = [] # 候補となるx座標を記録
    prev_x = None   # 前回施行のx座標
    end_flg = False
    length = 0  # 線分の長さ(ノイズ対策)
    
    # 一番下から半分までの高さを探索
    for y in range(h-1,int(h/2),-1):    
        if end_flg:
            break
        # 右から10pxは尻の先端ではないと仮定し省略
        for x in range(w-10):
            # 画素が黒の場合
            if np.all(img[y][x] <= 0):  
                if not prev_x is None:
                    # print("prev_x - x:",prev_x-x)
                    if 0 <= x - prev_x < 5:  # 右上に伸びる連続性がある場合
                        length += 1
                    elif prev_x - x > 10:   # ノイズの場合
                        # print("noise")
                        continue
                    elif length > 30:
                        print("very long")
                        end_flg = True
                        break              
                    # 連続性が途絶え、かつ線分の長さがある程度長い場合、終了
                    elif x - prev_x > 5 and length > 15: # 連続性が途絶えた場合
                        end_flg = True
                        break
                    else:
                        length = 0

                # print(x,y) 
                # print("length:",length)
                if length > 5:
                    # img[y][x] = [0,0,255]
                    hip_pos_xs.append(x)   # 各幅の最も左の黒いピクセルをhip_pos_xとする
                # displayImg(img)
                
                prev_x = x  # 前回のx座標を記録
                break

    hip_pos_x = max(hip_pos_xs) # 候補となる複数のx座標の最大値を尻の先端部とする

    drawLine(display_img , (hip_pos_x,0), (hip_pos_x,h), color=(0, 0, 255))
    displayImg(display_img.repeat(2, axis=0).repeat(2, axis=1))
    
    # グローバル座標に変換
    hip_pos_x += torso_pos_x
    drawLine(descimg, (hip_pos_x,descimg.shape[0]), (hip_pos_x,0), color=(0, 0, 255))
    # displayImg(descimg)


def main(args):
    # 画像読み込み
    img = loadImg(mode=args.mode, img_url=args.img_url, img_path=args.img_path)
    # img = loadImg(mode=args.mode, img_url=i,  img_path=args.img_path)

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

    # ともを探索
    getHindlimb(torso_pos_x, descimg, bbox_position, deepcopy(resultimg), args)


    if args.display:
        displayImg(cv2.vconcat([resultimg, descimg]))  # 画像を表示


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation")
    parser.add_argument(
        "--mode", type=str, choices=["net", "local"], default="net", help="入力画像先"
    )
    parser.add_argument(
        "--img_url",
        type=str,
        default="https://blogimg.goo.ne.jp/user_image/51/48/a4f2767dcdda226304984ab5fd510435.jpg",
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



"""
失敗例
inputs = [
    "https://blog-imgs-51.fc2.com/u/m/a/uma1kuti/11007.jpg",
    "https://tospo-keiba.jp/images/articles/contents/shares/0002022/05/0511/%E3%82%A4%E3%83%AA%E3%82%B9%E3%83%AC%E3%83%BC%E3%83%B3.jpg",
    "https://pbs.twimg.com/media/FNcuyfnaQAAUPjj.jpg",
    "https://cdn-ak.f.st-hatena.com/images/fotolife/y/ymhope/20210309/20210309200653.jpg",
    "https://stat.ameba.jp/user_images/20220201/23/keika00/3f/19/j/o0800053315069363478.jpg?caw=800",
    "https://blogimg.goo.ne.jp/user_image/72/ff/2f9ab72d51b7851ae7a30a9b3e4e85aa.jpg",
    "https://umatokuimg.hochi.co.jp/horseimages/2020/2020102796/2020102796_1.jpg",
    "https://cdn-ak.f.st-hatena.com/images/fotolife/r/redfray/20200722/20200722191555.jpg",
]

"""

"""
成功例
inputs = [
    "https://blogimg.goo.ne.jp/user_image/5f/cb/121f584bd5a6b7ba9a285575879d1713.jpg",
    "https://blogimg.goo.ne.jp/user_image/22/0e/ff0d77f61ca14179ddffd3519cf76f2d.jpg",
    "https://blogimg.goo.ne.jp/user_image/51/48/a4f2767dcdda226304984ab5fd510435.jpg",
    "https://prtimes.jp/i/21266/9/resize/d21266-9-377533-0.jpg",
    "https://cdn.netkeiba.com/img.news/?pid=news_img&id=459925",
    "https://uma-furi.com/wp-content/uploads/2022/06/image-2.png",
    "https://blogimg.goo.ne.jp/user_image/65/c0/6efbb4a7472f3841c54fabaf87ec36d3.jpg",
    "https://blogimg.goo.ne.jp/user_image/7a/bf/a62ba86bc344b7cf730254c30dd8a774.jpg",
    "https://blogimg.goo.ne.jp/user_image/73/de/80fce1b7c34744815f4c277f2447fbe8.jpg",   

    "https://www-f.keibalab.jp/img/horse/2018103418/2018103418_05.jpg?1621591291",
    "https://www-f.keibalab.jp/img/horse/2014105979/2014105979_05.jpg?1495362433",
    "https://i.daily.jp/horse/horsecheck/2017/11/27/Images/d_10768515.jpg" ,
    "https://i.daily.jp/horse/horsecheck/2018/02/13/Images/d_10982189.jpg",
    "https://www-f.keibalab.jp/img/horse/2019106342/2019106342_34.jpg?1653187636",
    "https://kyoto-tc.jp/images/club/2020/01/side.jpg",
    "https://jra-van.jp/fun/seri/2020/imgs/select/kougaku1/1s_200713_01.jpg",
    "https://blogimg.goo.ne.jp/user_image/29/d8/fafbb42d885c8b3a3474ac08ed5510c0.jpg",
    "https://uma-furi.com/wp-content/uploads/2022/06/image.png", 
    "https://blogimg.goo.ne.jp/user_image/73/de/80fce1b7c34744815f4c277f2447fbe8.jpg",
    "https://blogimg.goo.ne.jp/user_image/6f/f5/250ad840775c1949b95d890368658fad.jpg",
]
"""

