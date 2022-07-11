import cv2
import urllib


def loadImg(mode,img_url,img_path):
    """
    画像を読み込む
    :param mode (type:string) "net" or "local" ネット上にある画像orローカルファイル を読み込み
    :param img_url (type:string) ネットの画像url
    :param img_path (type:string) ローカルファイルの画像パス名
    :return img (numpy.ndarray) 画像データ
    """
    if mode == "net":
        # インターネットから画像をダウンロード
        urllib.request.urlretrieve(img_url, "./horse.jpg")
        # 画像を読み込み
        img = cv2.imread("./horse.jpg")  # (H,W,3)

    elif mode == "local":
        img = cv2.imread(img_path)

    return img


def drawText(img, text, x, y, scale=0.6, edgecolor=(0,255,0), color=(0,0,0)):
    """
    画面に文字列を描画
    :param img (type:numpy.ndarray) 画像情報
    :param text (type:string) 描画するテキスト
    :param x (type:int) テキストのx座標
    :param y (type:int) テキストのy座標
    :param scale (type:float) テキストの大きさ
    :param edgecolor (type:tuple) テキストの縁の色
    :param color (type:tuple) テキストの色
    """
    cv2.putText(
        img,
        text=text,
        org=(x, y),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=scale,
        color=edgecolor,
        thickness=4,
        lineType=cv2.LINE_4,
    )
    cv2.putText(
        img,
        text=text,
        org=(x, y),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=scale,
        color=color,
        thickness=2,
        lineType=cv2.LINE_4,
    )


def drawLine(img, start_point, end_point, color=(255,255,255), bold=2):
    """
    画像に線を描画
    :param img (type:numpy.ndarray) 画像情報
    :param start_point (type:tuple) 始点(x座標,y座標)
    :param end_point (type:tuple) 終点(x座標,y座標)
    :param color (type:tuple) 線の色
    :param bold (type:int) 線ｎ太さ
    """
    cv2.line(
        img,
        start_point,
        end_point,
        color,
        thickness=bold,
        lineType=cv2.LINE_AA,
    )


def displayImg(img):
    """
    画像を表示
    :param img (type:numpy.ndarray) 画像情報
    """
    cv2.imshow("display image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def update_values(dict_from, dict_to):
    """
    引数をymlファイルの内容に更新
    :param dict_from (type:dict) ymlファイルに記述された新規引数
    :param dict_to (type:dict) 更新される側の引数
    ref: https://github.com/salesforce/densecap/blob/5d08369ffdcb7db946ae11a8e9c8a056e47d28c2/data/utils.py#L85
    """
    for key, value in dict_from.items():
        if isinstance(value, dict):
            update_values(dict_from[key], dict_to[key])
        elif value is not None:
            dict_to[key] = dict_from[key]