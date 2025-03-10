import glob
import os
from pathlib import Path

import cv2
import numpy as np
import pyheif
from page_dewarp import page_dewarp
from PIL import Image
import pillow_heif

def remove_shadows_and_flatten(image):
    """
    モルフォロジーを使って背景の濃淡を推定し、影や照明ムラを補正して
    背景をできるだけフラット（白に近く）にする。
    戻り値は「フラット化したグレースケール画像」。
    """
    # グレースケール化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- 手法1: モルフォロジー閉じ処理で背景推定 ---
    # 大きめのカーネルで morphClose を行うと、文字より広い領域が潰されて
    # 背景の濃淡(シャドウ)が抽出されやすくなる。
    # カーネルサイズは画像の解像度に応じて変える（例: (51,51) や (101,101)など）
    kernel_size = 51
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # bg は「背景の濃淡が大きくぼやけた画像」。これを使って、
    # 元の gray から差分 or 割り算をして影を除去する。

    # 差分方式: diff = (gray - bg) の絶対値を反転させるなど
    # 割り算方式: ratio = gray / bg で正規化
    # ここでは「差分＋反転」で背景を明るく、文字を暗くする例を使う
    diff = cv2.absdiff(gray, bg)
    # 文字部分は大きな差分 -> 白くなるが、背景は差分小 -> 黒っぽくなる。
    # OCRしやすいように背景が白、文字が黒になるように反転
    shadow_removed = 255 - diff

    # 0-255 に正規化しておく（不要ならスキップ可）
    shadow_removed = cv2.normalize(shadow_removed, None, 0, 255, cv2.NORM_MINMAX)
    shadow_removed = shadow_removed.astype(np.uint8)

    return shadow_removed


def mask_red(image):
    """
    HSVベースで赤色領域をマスク（2区間）。
    戻り値は 0/255 の2値マスク（1ch）。
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 赤色の色相範囲(例: 0-10, 170-180) → 要調整
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    return mask


def better_color_correction(image):
    """
    1) 影や照明ムラを除去してグレースケールでフラット化
    2) 赤文字だけは元色を残す
    3) その他の部分は二値化（背景=白, 文字=黒）したい

    戻り値: BGR画像（赤文字は赤、それ以外は白黒）
    """
    # まず赤マスクを取得
    red_mask = mask_red(image)

    # 背景フラット化（グレースケール）
    flattened_gray = remove_shadows_and_flatten(image)

    # 二値化(Otsuなど)
    # flattened_gray は既に0が暗部、255が明部に近いので、そのままthresholdでOK
    _, bin_img = cv2.threshold(flattened_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # bin_img は 0(黒文字) or 255(白背景)。BGR に変換
    bin_bgr = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)

    # 赤領域だけは元の画像を使用する
    red_mask_3ch = cv2.merge([red_mask, red_mask, red_mask])  # shape同じ3chマスク

    # 出力画像 = 赤マスク部分は「元のカラー」、それ以外は「白黒二値」
    out = np.where(red_mask_3ch == 255, image, bin_bgr)

    return out


def read_heic_to_numpy(file_path: str):
    heif_file = pillow_heif.read_heif(file_path)[0]

    # RGB形式に変換する（念のため）
    image = heif_file.to_pillow().convert("RGB")

    # NumPyのndarrayに変換する
    rgb_array = np.array(image)

    return rgb_array


def main():
    # inputs フォルダ内の png 画像を処理して processed フォルダに出力
    input_files = sorted(Path("./inputs").glob("*.HEIC"))
    os.makedirs("processed", exist_ok=True)

    for i, input_file in enumerate(input_files):
        input_file: Path
        filename = input_file.name.replace(".HEIC", ".png")
        output_file = f"processed/{i}.png"

        image = read_heic_to_numpy(input_file)

        warped = page_dewarp(image)

        final_result = warped

        # 2. カラー調整 (白黒 + 赤)
        # final_result = better_color_correction(image)

        # 結果を保存
        cv2.imwrite(output_file, final_result)


if __name__ == "__main__":
    main()
