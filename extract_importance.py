import json

import cv2
import numpy as np
import easyocr

reader = easyocr.Reader(["en"], gpu=True)


def detect_patterns(
    main_image_path,
    templates_info,
    overlap_threshold=40,
    match_threshold=0.85,
    x_th=700,
):
    """
    メイン画像から複数のテンプレートパターンを検出し、重複を考慮して結果を返す関数。

    Args:
        main_image_path (str): 検出対象のメイン画像ファイルパス。
        templates_info (list): テンプレート情報を含む辞書のリスト。
                                各辞書は {'path': 'template.png', 'color': (B, G, R), 'priority': int}
                                priorityが高いほど、重複時に優先される。
        overlap_threshold (int): 重複とみなすピクセルの距離（左上角基準）。
        match_threshold (float): テンプレートマッチングの類似度閾値 (0.0 - 1.0)。

    Returns:
        tuple: (processed_image, detected_patterns_info)
               processed_image (numpy.ndarray): 検出結果が描画された画像。
               detected_patterns_info (dict): 検出された各パターンの情報を格納した辞書。
                                              キーはテンプレートパス、値は [(x, y, w, h), ...] のリスト。
    """

    main_img_color = cv2.imread(main_image_path)
    if main_img_color is None:
        print(f"エラー: メイン画像 '{main_image_path}' を読み込めませんでした。")
        return None, {}
    main_img_gray = cv2.cvtColor(main_img_color, cv2.COLOR_BGR2GRAY)

    all_detections = []  # 全ての検出結果を格納 [(x, y, w, h, template_path, priority)]

    # テンプレートの優先順位でソート (priorityが高いものが先に処理される)
    # こうすることで、後から検出される（優先順位の低い）パターンが、
    # 既に検出された（優先順位の高い）パターンと重複した場合に除外されやすくなる
    templates_info.sort(key=lambda x: x["priority"], reverse=True)

    for template_info in templates_info:
        template_path = template_info["path"]
        template_color = template_info["color"]
        template_priority = template_info["priority"]

        template_img = cv2.imread(template_path, 0)
        if template_img is None:
            print(
                f"エラー: テンプレート画像 '{template_path}' を読み込めませんでした。"
            )
            continue

        w, h = template_img.shape[::-1]

        # テンプレートマッチング
        res = cv2.matchTemplate(main_img_gray, template_img, cv2.TM_CCOEFF_NORMED)

        # 閾値処理
        locations = np.where(res >= match_threshold)
        raw_positions = list(zip(*locations[::-1]))

        if not raw_positions:
            continue

        # 重複する矩形をまとめる
        rectangles = []
        for x, y in raw_positions:
            rectangles.append([int(x), int(y), int(w), int(h)])

        # groupThreshold=1, eps=0.2 は調整可能
        grouped_rects, _ = cv2.groupRectangles(rectangles, 1, 0.2)

        for x, y, w_rect, h_rect in grouped_rects:
            if x > x_th:
                continue

            all_detections.append(
                {
                    "x": x,
                    "y": y,
                    "w": w_rect,
                    "h": h_rect,
                    "path": template_path,
                    "color": template_color,
                    "priority": template_priority,
                }
            )

    # 重複処理: priorityが高い方を優先
    # 既に検出済みの、より優先度の高いパターンと重複する場合は、優先度の低い方を破棄
    final_detections = []

    # x,y座標でソートしておくと、ループ内の比較が少し効率的になる可能性がある
    all_detections.sort(key=lambda d: (-d["priority"], d["x"], d["y"]))

    for i, current_detection in enumerate(all_detections):
        is_overlapping_with_higher_priority = False

        for final_detection in final_detections:
            # 現在の検出が、既に確定した（より優先度が高いか同等な）検出と重複しているかチェック

            # 優先度判定:
            # - current_detectionのpriorityが低い、または同じで、
            #   final_detectionと重複している場合はスキップ
            #   (ここではfinal_detectionsには既に優先度の高いものが入っている、という前提で、
            #    current_detectionがfinal_detectionよりpriorityが低い場合は考慮不要。
            #    ただし、念のためcurrent_detectionのpriorityが高い場合は上書きするようにロジック組むことも可能だが、
            #    今回は`templates_info.sort`で事前にソートしているので不要)

            if current_detection["priority"] < final_detection["priority"]:
                # すでに確定している検出の方が優先度が高い
                # current_detectionがその高優先度検出と重複しているかチェック
                dist_x = abs(current_detection["x"] - final_detection["x"])
                dist_y = abs(current_detection["y"] - final_detection["y"])

                if dist_x < overlap_threshold and dist_y < overlap_threshold:
                    # 重複していると判断し、この検出はスキップ
                    is_overlapping_with_higher_priority = True
                    break
            elif (
                current_detection["priority"] == final_detection["priority"]
                and current_detection["path"] != final_detection["path"]
            ):
                # 同じ優先度だが異なるテンプレートの場合も、念のため重複チェック
                dist_x = abs(current_detection["x"] - final_detection["x"])
                dist_y = abs(current_detection["y"] - final_detection["y"])

                if dist_x < overlap_threshold and dist_y < overlap_threshold:
                    # 重複していると判断し、この検出はスキップ（どちらか一方を残すため）
                    # ここでは一旦スキップするが、後で同じ位置に違うテンプレートが検出される可能性もある
                    # シンプルにするため、同じ位置に同じ優先度の異なるテンプレートが来る場合は、
                    # 先に処理された方を残し、後から来た方を破棄する
                    is_overlapping_with_higher_priority = True
                    break
            # その他のケース（current_detectionのpriorityが高いか、重複なし）は、
            # final_detectionsに追加されるべき

        if not is_overlapping_with_higher_priority:
            # 同じテンプレートが重複する可能性があるので、それも排除
            # ここでは `groupRectangles` でかなり絞られているので、
            # 基本的には同じテンプレートの重複は少ないはずだが、念のため
            already_added_same_template = False
            for fd in final_detections:
                if fd["path"] == current_detection["path"]:
                    dist_x = abs(current_detection["x"] - fd["x"])
                    dist_y = abs(current_detection["y"] - fd["y"])
                    if dist_x < overlap_threshold and dist_y < overlap_threshold:
                        already_added_same_template = True
                        break
            if not already_added_same_template:
                final_detections.append(current_detection)

    # 最終的な結果描画と情報整理
    detected_patterns_info = {info["path"]: [] for info in templates_info}

    for det in final_detections:
        x, y, w, h = det["x"], det["y"], det["w"], det["h"]
        color = det["color"]
        path = det["path"]

        # 描画
        cv2.rectangle(main_img_color, (x, y), (x + w, y + h), color, 2)

        # 情報格納
        detected_patterns_info[path].append((x, y, w, h))

    return main_img_color, detected_patterns_info


def extract_importance_page(idx: int) -> list[tuple[list[str], int]]:
    main_screenshot_path = f"./inputs-teppeki/{idx}.png"  # 検出対象の画像

    # テンプレートの定義
    # 'priority': 優先度を設定。数値が高いほど優先される。
    # imp_2はimp_1を内包しているので、imp_2を優先（priorityを高く）する。
    # こうすると、imp_2が見つかった場所ではimp_2が描画され、
    # そのimp_2と重複するimp_1は描画されないようになる。
    templates_to_detect = [
        {"path": "imp_1.png", "color": (0, 0, 255), "priority": 1},  # 赤色 (BGR)
        {"path": "imp_2.png", "color": (255, 0, 0), "priority": 2},  # 青色 (BGR)
        {"path": "square.png", "color": (0, 255, 255), "priority": 0},  # 黄色 (BGR)
    ]

    # パターン検出を実行
    result_image, detected_data = detect_patterns(
        main_screenshot_path,
        templates_to_detect,
        overlap_threshold=40,  # 重複判定の距離（ピクセル）
        match_threshold=0.8,  # マッチングの類似度閾値
        x_th=700 if idx % 2 == 1 else 770,
    )

    cv2.imwrite("./detection_result.png", result_image)

    lines = [[] for _ in range(3588)]

    for template_path, locations in detected_data.items():
        for loc in locations:
            p = loc[1]

            if template_path == "square.png":
                p -= 10

            lines[p].append((template_path, loc))

    squares = []

    for line in lines:
        for tmp, loc in line:
            if tmp == "square.png":
                squares.append([loc, 0, None])
            elif tmp == "imp_1.png":
                squares[-1][1] = 1
                squares[-1][2] = loc
            else:
                squares[-1][1] = 2
                squares[-1][2] = loc

    main_img_color = cv2.imread(main_screenshot_path)

    res = []

    for i, (loc, imp, i_loc) in enumerate(squares):
        if i_loc is None:
            continue
        word_img = main_img_color[
            loc[1] - 10 : i_loc[1] + 60, loc[0] + loc[2] - 2 : i_loc[0] + 5
        ]
        # cv2.imwrite(f"./zzs/{i}.png", word_img)

        results = reader.readtext(word_img, detail=0)
        res.append((results, imp))

    return res


def process() -> list[tuple[list[str], int]]:
    final_res = []

    for i in range(439, 440):
        if i in (435, ):
            continue
        print(i)
        res = extract_importance_page(i)
        final_res.extend(res)

    return final_res


def main():
    res = process()

    print(res)

    # with open("importance.json", "w") as f:
    #     json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
