import glob
import os
import re
from pathlib import Path

import cv2
import numpy as np
import pyheif
import pytesseract

from page_dewarp import page_dewarp
from PIL import Image
import pillow_heif


pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'


def auto_rotate_image(image: np.ndarray) -> np.ndarray:
    """
    画像の向きをTesseract OSDを使って自動検出し、正位置に回転させる。
    入力・出力ともにOpenCVの画像形式(numpy array, BGR)。

    注意: この関数を使用するには、Tesseract OCRエンジンが
          システムにインストールされている必要があります。
    """
    # グレースケールに変換した方がOSDの精度が安定する場合があります
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    try:
        config_str = '--psm 0'

        # TesseractのOSD機能を使って回転角度や確信度などを取得
        # lang='jpn+eng' のように言語を指定すると精度が上がることがあります
        osd = pytesseract.image_to_osd(gray, output_type=pytesseract.Output.DICT, lang='osd+jpn+eng', config=config_str)

        print(osd)

        # 検出された回転角度を取得（検出失敗時は0度とする）
        rotation_angle = osd.get('rotate', 0)
        # 確信度もログ出力しておくとデバッグに便利です
        # print(f"Detected rotation: {rotation_angle} degrees, Confidence: {osd.get('orientation_conf')}")

        rotated_image = image

        # 検出された角度に応じて画像を回転させる
        if rotation_angle == 90:
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_angle == 180:
            rotated_image = cv2.rotate(image, cv2.ROTATE_180)
        elif rotation_angle == 270:
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return rotated_image

    except pytesseract.TesseractError as e:
        print(f"Tesseract OSDでエラーが発生しました: {e}")
        # エラーが発生した場合は、元の画像をそのまま返す
        return image


def remove_shadows_and_flatten(image):
    """
    モルフォロジーを使って背景の濃淡を推定し、影や照明ムラを補正して
    背景をできるだけフラット（白に近く）にする。
    戻り値は「フラット化したグレースケール画像」。
    """
    # グレースケール化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- 手法1: モルフォロジー閉じ処理で背景推定 ---
    kernel_size = 51
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # 差分を反転して影領域を明るくする
    diff = cv2.absdiff(gray, bg)
    shadow_removed = 255 - diff

    # 0-255 に正規化
    shadow_removed = cv2.normalize(shadow_removed, None, 0, 255, cv2.NORM_MINMAX)
    shadow_removed = shadow_removed.astype(np.uint8)

    return shadow_removed


def mask_red(image):
    """
    HSVベースで赤色領域をマスク（2区間）。
    戻り値は 0/255 の2値マスク（1ch）。
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 赤色の色相範囲(例: 0-10, 170-180)
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
    3) その他の部分は二値化（背景=白, 文字=黒）

    さらに追加で、
    - 純粋に白黒化した画像
    - 赤ピクセルのみ取り出した画像
    も返すようにする

    戻り値: (赤+白黒画像, 白黒画像, 赤ピクセルのみ画像) の3つ
    """
    # まず赤マスクを取得
    red_mask = mask_red(image)

    # 背景フラット化（グレースケール）
    flattened_gray = remove_shadows_and_flatten(image)

    # 二値化(Otsu)
    _, bin_img = cv2.threshold(flattened_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # bin_img は0か255の白黒 => 3chに変換しておく
    bin_bgr = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)

    # ========== (1) 赤 + 白黒 ==========
    # 赤領域だけは元のカラーを使う
    red_mask_3ch = cv2.merge([red_mask, red_mask, red_mask])  # 3chマスク
    red_and_bw = np.where(red_mask_3ch == 255, image, bin_bgr)

    # ========== (2) 純粋に白黒化した画像 ==========
    # flattened_gray の二値化結果 bin_bgr をそのまま「白黒版」として使う
    mono = bin_bgr.copy()

    # ========== (3) 赤ピクセルのみ取り出した画像 ==========
    # 背景を黒にして、赤マスク部分だけ元画像から抜き出す
    red_only = cv2.bitwise_and(image, image, mask=red_mask)

    return red_and_bw, mono, red_only


def read_heic_to_numpy(file_path: str):
    heif_file = pillow_heif.read_heif(file_path)[0]

    # RGB形式に変換する（念のため）
    image = heif_file.to_pillow().convert("RGB")

    # NumPyのndarrayに変換する
    rgb_array = np.array(image)

    return rgb_array


def main():
    name = "2-normal"

    # 入力パスを取得
    input_files = sorted(Path(f"./inputs-{name}").glob("*.heic"))

    # 出力フォルダを作成
    os.makedirs(f"processed-{name}", exist_ok=True)
    os.makedirs(f"processed-{name}/mono", exist_ok=True)
    os.makedirs(f"processed-{name}/red", exist_ok=True)

    for i, input_file in list(enumerate(input_files)):
        input_file: Path
        # 出力ファイル名のベース
        filename_base = str(i)  # または input_file.stem などでもOK

        if Path(f"processed-{name}/{filename_base}.png").exists():
            continue

        # HEIC読み込み
        image = read_heic_to_numpy(input_file)

        image = auto_rotate_image(image)

        # 1. ページの歪み補正
        try:
            # warped = page_dewarp(image)
            raise Exception
        except:
            # raise
            warped = image

        # 2. カラー調整 (赤 + 白黒 / 白黒のみ / 赤のみ)
        red_bw_result, mono_result, red_only_result = better_color_correction(warped)

        # 3. それぞれ保存
        # 赤+白黒 (従来通り processed/ に保存)
        cv2.imwrite(f"processed-{name}/{filename_base}.png", red_bw_result)

        # 白黒のみ (processed/mono/ に保存)
        cv2.imwrite(f"processed-{name}/mono/{filename_base}.png", mono_result)

        # 赤のみ (processed/red/ に保存)
        cv2.imwrite(f"processed-{name}/red/{filename_base}.png", red_only_result)


if __name__ == "__main__":
    main()
