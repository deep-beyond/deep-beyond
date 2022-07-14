import cv2
import csv
import yaml
import numpy as np
import argparse
from copy import deepcopy
from pylsd.lsd import lsd

# deep.pyのDeepSementationクラス
from deep import DeepSegmentation

# utils.pyの関数
from utils import loadImg, drawText, drawLine, displayImg, update_values

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
        # 0除算対策
        if distance_x == 0:
            distance_x += 0.00001
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

    # 前足の頂点らの最も右端の点
    last_toes_pos_x = max(toes_pos_xs)

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
        cv2.circle(descimg, tuple(wither_pos[0]), 5, (0, 0, 255), thickness=-1)
        cv2.circle(descimg, tuple(wither_pos[1]), 5, (0, 0, 255), thickness=-1)
        drawLine(descimg,tuple(wither_pos[0]),tuple(wither_pos[1]),(0,0,255))
        drawText(descimg, "x:{} y:{}".format(wither_pos[0][0],wither_pos[0][1]), wither_pos[0][0] + 20, wither_pos[0][1])
        drawText(descimg, "x:{} y:{}".format(wither_pos[1][0],wither_pos[1][1]), wither_pos[1][0] + 20, wither_pos[1][1])

    return wither_pos_x, wither_pos, last_toes_pos_x


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
        cv2.circle(descimg, torso_front_pos, 5, (0, 0, 255), thickness=-1)
        cv2.circle(descimg, (torso_back_pos[0], torso_back_pos[1]), 5, (0, 0, 255), thickness=-1)
        drawText(descimg, "x:"+str(wither_pos_x), torso_front_pos[0], torso_front_pos[1]+30)
        drawText(descimg, "x:"+str(torso_pos_x), torso_back_pos[0], torso_back_pos[1]+30)

    return torso_pos_x


def getHip(torso_pos_x, descimg, bbox_position, img):
    """
    尻の先端部の座標を探索
    :param torso_pos_x (type:int) 胴のx情報
    :param bbox_position (type:tuple) 外接矩形座標(x,y,h,w)
    :param img (type:numpy.ndarray) 生の画像
    :return hip_pos_x (type:int) 尻の先端部のx座標
    """

    """
    1. 探索範囲を設定
    """
    bbox_x = bbox_position[0]
    bbox_y = bbox_position[1]
    bbox_h = bbox_position[2]
    bbox_w = bbox_position[3]

    # 画像の尻部分のみ着目
    base_img = img[bbox_y : bbox_y+int(bbox_h/2), torso_pos_x: bbox_x+bbox_w]
    h, w = base_img.shape[:2]

    """
    2. 尻の先端のx座標を探索
    """

    # 尻の先端のx座標を記録
    hip_pos_log = []

    # 画像の大きさによって線の太さやノイズ除去の度合を変更
    if img.shape[1] < 150:
        bold = 15   # 線の太さ
        ksize = 3   # ノイズ除去の度合
    else:
        bold = 20
        ksize = 5

    for itr in [1, 2, 3]:
        # 各コントラスト値で実行
        for alpha in [1.5, 2.5, 4.5]:
            """
            前処理
            """
            # コントラスト調整
            img = cv2.convertScaleAbs(base_img, alpha = alpha)

            # グレースケール化
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 輪郭抽出(尾の線を除去するために後々必要になる)
            contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 適応的閾値の二値化
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 13, 20)

            # 直線検出して画像をデフォルメ化(検出に必要な線を洗い出す: ハフ変換では不可能)
            deform_img = np.zeros((img.shape[0], img.shape[1]),dtype=np.uint8)
            for line in lsd(img):
                x1, y1, x2, y2 = map(int,line[:4])
                deform_img = cv2.line(deform_img, (x1,y1), (x2,y2), (255), 1)        
            img = cv2.bitwise_not(deform_img)   # ネガポジ反転

            # 線膨張(線を強調)
            img = cv2.bitwise_not(img)
            kernel = np.ones((3,3),np.uint8)        
            img = cv2.dilate(img,kernel,iterations = itr)
            img = cv2.bitwise_not(img)

            # 画像内の最も右下にある黒色ピクセルを探索
            bottom_right_x_pos = 0
            for x in range(w-1,0,-1):
                if np.all(img[h-1][x] <=0):
                    bottom_right_x_pos = x
                    break

            # 馬の尾から背中にかけての輪郭のみを残す
            contour = np.array([],dtype=np.int32)
            draw_flg = False
            for cnt in contours[0]:            
                x, y = cnt[0][0], cnt[0][1]
                # 最も右下にある黒色ピクセルに近づいたら記録開始
                if bottom_right_x_pos - x < 10: 
                    draw_flg = True
                if draw_flg:
                    contour = np.append(contour, cnt)
            contour = contour.reshape([-1,1,2])

            # 馬の尾から背中にかけての輪郭を白色にする(線を除去)
            for i in range(len(contour)-1):
                x1, y1 = contour[i][0][0], contour[i][0][1]
                x2, y2 = contour[i+1][0][0], contour[i+1][0][1]
                drawLine(img , (x1, y1), (x2, y2), color=(255), bold=bold)

            # ノイズ除去
            for _ in range(4):
                img = cv2.medianBlur(img, ksize=ksize)

            """
            尻の先端のx座標を探索
            """
            cand_pos_xs = [] # 候補となるx座標を記録
            prev_x = None   # 時刻t-1におけるx座標
            end_flg = False
            length = 0  # 線分の長さ(ノイズ対策)

            # 一番下から半分までの高さを探索
            for y in range(h-1,int(h/2),-1):    
                if end_flg:
                    break
                # 右から10pxは尻の先端ではないと仮定し省略
                for x in range(w-10):
                    if np.all(img[y][x] <= 0):  # 画素が黒の場合
                        if not prev_x is None:
                            if 0 <= x-prev_x < 5:  # 右上に伸びる連続性がある場合
                                length += 1
                            elif prev_x - x > 10:   # ノイズの場合
                                continue
                            elif length > 50: # 長さがある程度長ければ終了
                                end_flg = True
                                break              
                            else:
                                length = 0  # 連続性なしと判断

                        if length > 10:
                            cand_pos_xs.append(x)
                        prev_x = x  # 時刻tのx座標を時刻t-1のx座標として記録
                        break
            
            # 候補となる複数のx座標の最大値を尻の先端部とする
            if cand_pos_xs:
                hip_pos_log.append(max(cand_pos_xs))
            else:
                print("候補なし")
        else:
            # もし記録リストが空でなければ終了、空ならば線膨張の線の太さを変更
            if not hip_pos_log == []:
                break

    # 記録の中で最も右側にある候補x座標を尻の先端のx座標とする
    hip_pos_x = max(hip_pos_log)

    # グローバル座標に変換
    hip_pos_x += torso_pos_x

    drawLine(descimg, (hip_pos_x,descimg.shape[0]), (hip_pos_x,0), color=(0, 0, 255)) # デバッグ用

    return hip_pos_x


def getHindlimb(hip_pos_x, contour_vertex, last_tpes_pos_x, descimg, args):
    """
    ともを探索
    :param hip_pos_x (type:int) 尻の先端のx座標
    :param contour_vertex (type:list) 頂点10以上の近似輪郭の座標リスト
    :param descimg (type:numpy.ndarray) 説明する画像(テキストや直線の描画などに使用)
    :return hindlimb_pos_x (type:int) ともの始点のx座標
    """

    """
    1. ともの始点を探索
    """
    # 頂点数制限(7頂点までを見る)
    limit_cnt = 0

    # ともの始点の座標
    hindlimb_pos_x = 0
    hindlimb_pos_y = float('inf')

    # -1 にしないとfor文最後の実行において終点で配列外参照になる
    for i in range(len(contour_vertex) - 1):
        x1, y1 = contour_vertex[i]  # 始点

        # 始点が外接矩形の左半分ならばスキップ
        if x1 < last_tpes_pos_x:
            continue

        # 条件を満たす7頂点を見れば処理終了
        if limit_cnt > 7:
            break

        limit_cnt += 1
        # 条件を満たす7点の中でy座標が最小の点がともの始点
        if y1 < hindlimb_pos_y:
            hindlimb_pos_y = y1
            hindlimb_pos_x = x1

    if args.display:
        drawText(descimg, "x:"+str(hindlimb_pos_x), hindlimb_pos_x, hindlimb_pos_y-30)
        drawText(descimg, "x:"+str(hip_pos_x), hip_pos_x, hindlimb_pos_y-30)
        cv2.circle(descimg, (hindlimb_pos_x, hindlimb_pos_y), 5, (0, 0, 255), thickness=-1)
        cv2.circle(descimg, (hip_pos_x, hindlimb_pos_y), 5, (0, 0, 255), thickness=-1)
        drawLine(descimg, (hindlimb_pos_x, hindlimb_pos_y), (hip_pos_x, hindlimb_pos_y), (0, 0, 255))

    return hindlimb_pos_x


def getNeck(contour_vertex, wither_pos, descimg, args):
    """
    首を探索
    :param contour_vertex (type:list) 頂点10以上の近似輪郭の座標リスト
    :param wither_pos (type:list) キ甲を形成する2点の座標
    :param descimg (type:numpy.ndarray) 説明する画像(テキストや直線の描画などに使用)
    :return neck_length (type:float) 首の長さ
    """
    # yの値が最も小さい頂点を始点とする
    contour_vertex = sorted(contour_vertex, key=lambda x: x[1])
    x1, y1 = contour_vertex[0]  # 始点
    x2, y2 = wither_pos[0]   # 終点
    neck_length = np.sqrt(abs(x2-x1)**2 + abs(y2-y1)**2)
    neck_length = round(neck_length,1)

    if args.display:
        drawLine(descimg,(x1,y1),(x2,y2),(0,0,255))
        drawText(descimg, "x:"+str(x1), x1, y1)
        cv2.circle(descimg, (x1, y1), 5, (0, 0, 255), thickness=-1)
    
    return neck_length


def getJumpsuit(torso_pos_x, bbox_position, img, descimg, args):
    """
    繋(球節:Fetlock ~ 地面ground)を探索
    :param torso_pos_x (type:int) 胴の終点のx情報
    :param bbox_position (type:tuple) 外接矩形座標(x,y,h,w)
    :param img (type:numpy.ndarray) 原画像
    :param descimg (type:numpy.ndarray) 説明する画像(テキストや直線の描画などに使用)
    :return fetlock_length (type:int) 繋の長さ
    :return fetlock_tilt (type:float) 繋の傾き
    """

    """
    1. 探索範囲を設定
    """
    bbox_y = bbox_position[1]
    bbox_h = bbox_position[2]
    
    w_border = torso_pos_x  # 胴の終点のx情報を左端
    h_border = bbox_y+int(bbox_h*2/3)   # 上端

    # 外接矩形の2/3の高さ~画像高さ,胴の終点~画像幅の範囲
    img = img[h_border: img.shape[0], w_border:img.shape[1]]    # [上端:下端, 左端:右端]

    """
    2. 後ろ脚を見るためより範囲を絞る
    """

    # 輪郭抽出
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 400:
            # 輪郭の面積が小さいものは除去
            continue
        arclen = cv2.arcLength(contour, True)   # 輪郭線の長さ
        approx = cv2.approxPolyDP(contour, 0.005 * arclen, True)  # 輪郭線の近似,描画

        cv2.drawContours(img, [approx], -1, (0,255,255), 2)

    # displayImg(img)

    # 近似輪郭の頂点をy座標でソート(大きいもの順)
    cand_pos = [[approx[i][0][0], approx[i][0][1]] for i in range(len(approx))]
    cand_pos = sorted(cand_pos, key=lambda x: x[1], reverse=True)

    # y座標が上位4つの座標において最も右にある頂点を後ろ足の頂点と頂点する
    x_max = 0
    y_max = 0
    for cand in cand_pos[:4]:
        x, y = cand
        if x_max < x:
            x_max = x
            y_max = y
    
    # cv2.circle(img, (x_max, y_max), 4, (0 , 0, 255), thickness=-1) 
    # displayImg(img)

    limit = int(bbox_h / 7) # 範囲調整用
    # 画像からはみ出ないため調整
    rightend = max(img.shape[1], x_max + 30)
    bottomend = max(img.shape[0], y_max + limit)

    # 画像を指定範囲で切り取り
    img = img[y_max - limit : bottomend, x_max - 30 : rightend]
    # displayImg(img)

    """
    3. 繋の長さと傾きを探索
    """

    # 輪郭抽出
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    arclen = cv2.arcLength(contours[0], True)   # 輪郭線の長さ
    approx = cv2.approxPolyDP(contours[0], 0.002 * arclen, True)  # 輪郭線の近似

    fetlock_x = 0   # 球節のx座標
    fetlock_y = 0   # 球節のy座標
    ground_x = 0    # 地面のx座標
    ground_y = 0    # 地面のy座標

    # 地面(画像内の最も下)のy座標を探索
    alpha = img[:,:,3]
    endflg = False
    for y in range(img.shape[0]-1,0,-1):
        if endflg:
            break
        for x in range(img.shape[1]-1,0,-1):
            # 透明でない画素であればその時点のy座標を地面のy座標として処理終了
            if alpha[y][x] == 255:
                ground_x = x
                ground_y = y
                cv2.circle(img, (ground_x, ground_y), 4, (0 , 255, 255), thickness=-1)
                endflg = True
                break

    # 球節の座標を探索
    prev_x = 0  # t-1の時のx座標
    for i in range(2,len(approx)):  # idx:2から開始 -> 画面最上部の頂点を無視
        x, y = approx[i][0][0], approx[i][0][1]
        if x - prev_x > 0:
            cv2.circle(img, (x, y), 4, (0 , 255, 0), thickness=-1)
        # print(x-prev_x)
        # 最も右に位置する(x座標最大)の頂点を球節の座標とする
        if fetlock_x < x:
            fetlock_x = x
            fetlock_y = y
        prev_x = x  # 時刻t-1 -> 時刻t
    

    # 繋の傾きを探索
    prev_y = 0  # 時刻t-1におけるy座標を記録
    log_x = []   # 条件を満たすx座標を記録
    log_y = []   # 条件を満たすy座標を記録
    flg = False
    for contour in contours:
        for cnt in contour:
            x, y = cnt[0][0], cnt[0][1]
            # y座標の傾きが負になると処理開始
            if y - prev_y < -1:
                flg = True

            # 球節よりy座標が大きい場合記録
            if flg and fetlock_y < y and x > ground_x:
                log_x.append(x)
                log_y.append(y)
                # cv2.circle(jumpsuit_img, (x, y), 4, (0 , 255, 0), thickness=-1)
            prev_y = y  # 時刻t-1 -> 時刻t

    # 傾きを求めるための球節以外の点
    endpoint_x = int(sum(log_x) / len(log_x))  
    endpoint_y = int(sum(log_y) / len(log_y))

    if args.display:
        cv2.circle(img, (fetlock_x, fetlock_y), 4, (0 , 0, 255), thickness=-1) 
        cv2.circle(img, (endpoint_x, endpoint_y), 4, (255 , 0, 0), thickness=-1)    # 角度計算に必要な点
        drawLine(img, (fetlock_x, fetlock_y), ((fetlock_x, ground_y)), (180, 105, 255))
        drawLine(img, (fetlock_x, fetlock_y), ((endpoint_x, endpoint_y)), (255, 0, 255))
        img = img.repeat(2, axis=0).repeat(2, axis=1)
        displayImg(img)

    # 繋の長さを計算
    fetlock_length = abs(fetlock_y-ground_y)

    # 繋の傾きを計算
    fetlock_tilt = (endpoint_y - fetlock_y) / (endpoint_x - fetlock_x)
    fetlock_tilt = round(fetlock_tilt,1)

    # グローバル座標に変換
    trans_x = x_max - 30 + w_border
    trans_y = y_max - limit + h_border
    fetlock_x += trans_x
    fetlock_y += trans_y
    endpoint_x += trans_x
    endpoint_y += trans_y
    ground_y += trans_y

    if args.display:
        cv2.circle(descimg, (fetlock_x, fetlock_y), 4, (0 , 0, 255), thickness=-1)  # 球節点
        cv2.circle(descimg, (endpoint_x, endpoint_y), 4, (255 , 0, 0), thickness=-1)    # 角度計算に必要な点
        drawLine(descimg, (fetlock_x, fetlock_y), ((fetlock_x, ground_y)), (180, 105, 255))
        drawLine(descimg, (fetlock_x, fetlock_y), ((endpoint_x, endpoint_y)), (255, 0, 255))

    return fetlock_length, fetlock_tilt


def main(args):
    if args.mode == 'net':
        inputs = args.img_url
    else:
        inputs = args.img_path
    
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
        wither_pos_x, wither_pos, last_toes_pos_x = getWithersPosition(contour_vertex, bbox_position, resultimg, descimg, args)
        wither_length = wither_pos[1][1] - wither_pos[0][1]
        print("キ甲の長さ:", wither_length)

        # 胴を探索
        torso_pos_x = getTorso(contour_vertex, bbox_position, wither_pos_x, descimg, args)
        torso_length = torso_pos_x - wither_pos[0][0]
        print("胴の長さ:", torso_length)

        # 尻のx座標を探索
        hip_pos_x = getHip(torso_pos_x, descimg, bbox_position, deepcopy(resultimg))

        # ともを探索
        hindlimb_pos_x = getHindlimb(hip_pos_x, contour_vertex, last_toes_pos_x, descimg, args)
        hindlimb_length = hip_pos_x - hindlimb_pos_x
        print("ともの長さ:", hindlimb_length)

        # 首を探索
        neck_length = getNeck(contour_vertex, wither_pos, descimg, args)
        print("首の長さ:",neck_length)

        # 繋を探索
        fetlock_length, fetlock_tilt = getJumpsuit(torso_pos_x, bbox_position, deepcopy(resultimg),descimg, args)
        print("繋の長さ:",fetlock_length)   
        print("繋の傾き:",fetlock_tilt)

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
    parser.add_argument("--csv", action="store_false", help="結果をcsvファイルに保存するか")
    args = parser.parse_args()

    assert (
        args.transparent and args.save_format == "png"
    ), "jpg format can't transparent"

    # ymlファイルに記述された引数で更新
    with open("./input.yml", 'r') as handle:
        options_yaml = yaml.load(handle, Loader=yaml.SafeLoader)
    update_values(options_yaml, vars(args))

    main(args)
