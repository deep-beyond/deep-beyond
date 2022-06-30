import cv2
import argparse
import numpy as np

# deep.pyのDeepSementationクラス
from deep import DeepSegmentation

# utils.pyの関数
from utils import getImg, drawText, drawLine


def searchCnt(contours, img, args):
    """
    輪郭情報から頂点10以上の近似輪郭の座標を調査
    :param contours (type:list) マスクの輪郭情報
    :return cntPos (type:list) 頂点10以上の近似輪郭の座標リスト
    """
    # 輪郭の座標リスト
    cntPos = []

    imgH, imgW = img.shape[0], img.shape[1]

    thresholdWH = 0.6
    bboxX = 0  # 条件を満たす外接矩形のX座標
    bboxY = 0  # 条件を満たす外接矩形のY座標
    bboxH = 0  # 条件を満たす外接矩形の高さ
    bboxW = 0  # 条件を満たす外接矩形の幅

    for cnt in contours:
        # 輪郭線の長さ
        arclen = cv2.arcLength(cnt, True)

        if arclen == 0.0:
            break

        # 外接矩形計算
        x, y, w, h = cv2.boundingRect(cnt)

        # 外接矩形が画像サイズに対して比較的小さい場合はノイズとして無視
        if imgH * thresholdWH < h and imgW * thresholdWH < w:
            
            if args.showinfo:
                # 条件を満たす外接矩形を描画
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 輪郭線の近似,描画
            approx = cv2.approxPolyDP(cnt, 0.01 * arclen, True)
            if args.showinfo:
                cv2.drawContours(img, [approx], -1, (0, 255, 255), 2)

            bboxX = x
            bboxY = y
            bboxH = h
            bboxW = w

    else:
        # 近似輪郭線の各座標を調査
        for app in approx:
            x = app[0][0]
            y = app[0][1]
            pos = (x, y)
            cntPos.append(pos)

        return cntPos, (bboxX,bboxY,bboxH,bboxW)

    for cnt in contours:
        x = cnt[0][0]
        y = cnt[0][1]
        pos = (x, y)
        cntPos.append(pos)

    return cntPos


def getIntersection(contours, wither_posX, delta, bboxH):
    """
    輪郭とキ甲の交点を求める
    :param contours (type:list) マスクの輪郭情報
    :param wither_posX (type:int) キ甲のX座標
    :param delta (type:int) 揺れ(キ甲のX座標と完全一致がない場合に使用)
    :return wither_pos (type:list) 輪郭とキ甲の交点
    """
    wither_pos = []  # キ甲を形成する座標

    # 輪郭とキ甲の直線の交点を探索
    for cnt in contours:
        if wither_posX + delta == cnt[0][0]:
            wither_pos.append([cnt[0][0], cnt[0][1]])

    # 2点以上あった場合
    if len(wither_pos) > 1:
        # 交点の中でY座標が最小のものと最大なものを返す
        wither_pos.sort(key=lambda x: x[1])  # Y座標のデータでソート
        maximum = wither_pos[0][1]
        minimum = wither_pos[-1][1]
        if abs(maximum - minimum) < bboxH * 0.3:  # hard code !!
            return []
        else:
            return [wither_pos[0], wither_pos[-1]]
    else:
        return wither_pos


def getWithers(cntPos, contours, bboxPositon, img, args):
    """
    キ甲の座標情報を取得 & キ甲を描画
    :param cntPos (type:list) 頂点10以上の近似輪郭の座標リスト
    :param contours (type:list) マスクの輪郭情報
    :param bboxPosition (type:tuple) バウンディングボックス座標(x,y,h,w)
    :param bboxH (type:int) 外接矩形の高さ
    :param bboxY (type:int) 外接矩形のY座標
    :param img (type:numpy.ndarray) 入力画像と同じサイズの空の画像
    """
    bboxY = bboxPositon[1]
    bboxH = bboxPositon[2]
    bboxW = bboxPositon[3]

    # 前足を探索するため画像の「上部1/3～上部3/3」を見る
    onethird_h = bboxH // 3
    onethird_w = bboxW // 3

    # 表示用
    # for cp in cntPos:
    #     cv2.circle(img, (cp[0], cp[1]), 5, (255, 255, 255), thickness=-1)
    #     cv2.imshow("display image", img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # 時刻t-1における辺分の情報
    distanceY_past = 0
    # flg = False

    # 足先のX座標の頂点
    toesX_vertex = []
    toesY_vertex = []

    lowerline = int(bboxY + bboxH - bboxH * 0.1)

    if args.showinfo:
        # 足先を判定するためのラインを表示
        drawLine(img,0,lowerline,img.shape[1],lowerline,(125,125,125))

    for i in range(len(cntPos) - 1):
        x1, y1 = cntPos[i]  # 始点
        x2, y2 = cntPos[i + 1]  # 終点

        # 辺分が上部1/3の位置に属しているなら処理しない
        if y1 < onethird_h and y2 < onethird_h or x1 < onethird_w:
            continue
        
        if args.showinfo:
            drawLine(img,x1,y1,x2,y2,(255,0,0))
            drawText(img,str((x1,y1)),x1,y1)

        distanceX = x2 - x1
        distanceY = y2 - y1
        tilt = round(distanceY / distanceX,1)
        if args.showinfo:
            drawText(img,"Tilt:{}".format(tilt),(x1+x2)//2,(y1+y2)//2)

        # 頭部付近を避けるため（頭部付近は傾きが負になる傾向）
        # if tilt > 0:
        #     flg = True

        # cv2.imshow("display image", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # 時刻t-1に線分が急激に右肩上がりになり、時刻tで傾きが弱まった場合
        if distanceY_past < -bboxH * (2 / 7) and abs(tilt) < 3: # Hard code !!　　10
            break

        # 足先の頂点を探索
        # if flg and y1 > lowerline:    # Hard code !!
        if y1 > lowerline:    # Hard code !!
            toesX_vertex.append(x1)
            toesY_vertex.append(y1)
            cv2.circle(img, (x1, y1), 5, (0, 0, 255), thickness=-1)

        # 時刻tの情報を時刻t-1の情報にする
        distanceY_past = distanceY
    
    # cv2.imshow("display image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    assert len(toesX_vertex) != 0, "can't look for toes vertex"

    # キ甲のX座標：前足の頂点らの中点
    wither_posX = sum(toesX_vertex) // len(toesX_vertex)

    # キ甲のライン
    if args.showinfo:
        drawLine(img, wither_posX, 0, wither_posX, img.shape[0], color=(255, 255, 255))

    if len(contours) > 1:
        cntSummary = [cnt for contour in contours for cnt in contour]
        contours = np.array(cntSummary)
    else:
        contours = contours[0]

    contours = [
        cnt
        for cnt in contours
        if wither_posX - args.delta <= cnt[0][0] <= wither_posX + args.delta
    ]
    deltascope = [0]
    for i in range(1, args.delta + 1):
        deltascope.append(i)
        deltascope.append(-i)

    # 輪郭とキ甲の直線の交点を探索: 誤差deltaを許可
    wither_pos = []
    for delta in deltascope:
        wither_pos = getIntersection(contours, wither_posX, delta, bboxH)
        if len(wither_pos) > 1:
            print("delta:", delta)
            break
    else:
        # 例外：失敗した場合
        drawText(img, "failure", 100, 100)
        print("failure")
        return

    toesY = sum(toesY_vertex) // len(toesY_vertex)

    # 足先に線分の頂点が位置していない場合、足先の座標に変更
    # 2点の中でY座標が大きい方の頂点がtoesYよりも小さければ変更
    if wither_pos[0][1] > wither_pos[1][1]:
        if wither_pos[0][1] < toesY:
            wither_pos[0][1] = toesY
    else:
        if wither_pos[1][1] < toesY:
            wither_pos[1][1] = toesY

    drawLine(
        img,
        wither_pos[0][0],
        wither_pos[0][1],
        wither_pos[1][0],
        wither_pos[1][1],
        color=(0, 0, 255),
    )
    if args.showinfo:
        drawText(img, str(wither_pos[0]), wither_pos[0][0]+20, wither_pos[0][1])
        drawText(img, str(wither_pos[1]), wither_pos[1][0]+20, wither_pos[1][1])
    drawText(img, "delta:{}".format(delta), 0, bboxY)

    return


def main(args):
    # 画像読み込み
    img = getImg(args)

    # インスタンス生成(クラスの__init__メソッドを実行)
    ds = DeepSegmentation(args.colorPath, img)

    # クラスの__call__メソッドを実行
    resultImg, contours = ds()

    # 特定の条件を満たす輪郭の座標を取得
    cntPos, bboxPositon = searchCnt(contours, resultImg, args)

    if args.showinfo:
        # 輪郭描画
        cv2.drawContours(resultImg, contours, -1, (255, 255, 0), 3)

    if args.display:
        ds.displayImg(resultImg)

    # キ甲を探索
    getWithers(cntPos, contours, bboxPositon, resultImg, args)

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
        default="https://www-f.keibalab.jp/img/horse/2017102170/2017102170_12.jpg?1656168128",
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
    parser.add_argument("--delta", type=int, default=6, help="誤差許容範囲")
    parser.add_argument("--showinfo", action="store_false", help="詳細な情報を表示するか")
    args = parser.parse_args()
    main(args)
