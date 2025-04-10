#!/usr/bin/env python
######################################################################
# page_dewarp.py - Proof-of-concept of page-dewarping based on a
# "cubic sheet" model. Requires OpenCV (version 3 or greater),
# PIL/Pillow, and scipy.optimize.
######################################################################
# Author:  Matt Zucker
# Date:    July 2016
# License: MIT License (see LICENSE.txt)
######################################################################

import cv2
import numpy as np
import scipy.optimize

######################################################################
# 以下、Python2版からPython3版への変更ポイント:
#  - print 文を print() 関数に変更
#  - findContours の戻り値の扱いを現行の形式に変更
#  - sys.argv, main 関数によるファイル入出力を削除
#  - 最後に画像(ndarray)を返す E2E関数 page_dewarp を定義
#  - ライセンスやコメントなどは原著のまま
######################################################################

# for some reason pylint complains about cv2 members being undefined :(
# pylint: disable=E1101

PAGE_MARGIN_X = 0       # reduced px to ignore near L/R edge
PAGE_MARGIN_Y = 0       # reduced px to ignore near T/B edge

OUTPUT_ZOOM = 1.0        # how much to zoom output relative to *original* image
OUTPUT_DPI = 300         # just affects stated DPI of PNG, not appearance
REMAP_DECIMATE = 16      # downscaling factor for remapping image

ADAPTIVE_WINSZ = 55      # window size for adaptive threshold in reduced px

TEXT_MIN_WIDTH = 15      # min reduced px width of detected text contour
TEXT_MIN_HEIGHT = 2      # min reduced px height of detected text contour
TEXT_MIN_ASPECT = 1.5    # filter out text contours below this w/h ratio
TEXT_MAX_THICKNESS = 10  # max reduced px thickness of detected text contour

EDGE_MAX_OVERLAP = 1.0   # max reduced px horiz. overlap of contours in span
EDGE_MAX_LENGTH = 100.0  # max reduced px length of edge connecting contours
EDGE_ANGLE_COST = 10.0   # cost of angles in edges (tradeoff vs. length)
EDGE_MAX_ANGLE = 7.5     # maximum change in angle allowed between contours

RVEC_IDX = slice(0, 3)   # index of rvec in params vector
TVEC_IDX = slice(3, 6)   # index of tvec in params vector
CUBIC_IDX = slice(6, 8)  # index of cubic slopes in params vector

SPAN_MIN_WIDTH = 30      # minimum reduced px width for span
SPAN_PX_PER_STEP = 20    # reduced px spacing for sampling along spans
FOCAL_LENGTH = 1.2       # normalized focal length of camera

DEBUG_LEVEL = 0          # 0=none, 1=some, 2=lots, 3=all
DEBUG_OUTPUT = 'file'    # 'file', 'screen', 'both' などを使用可能

WINDOW_NAME = 'Dewarp'   # Window name for visualization

# nice color palette for visualizing contours, etc.
CCOLORS = [
    (255, 0, 0),
    (255, 63, 0),
    (255, 127, 0),
    (255, 191, 0),
    (255, 255, 0),
    (191, 255, 0),
    (127, 255, 0),
    (63, 255, 0),
    (0, 255, 0),
    (0, 255, 63),
    (0, 255, 127),
    (0, 255, 191),
    (0, 255, 255),
    (0, 191, 255),
    (0, 127, 255),
    (0, 63, 255),
    (0, 0, 255),
    (63, 0, 255),
    (127, 0, 255),
    (191, 0, 255),
    (255, 0, 255),
    (255, 0, 191),
    (255, 0, 127),
    (255, 0, 63),
]

# default intrinsic parameter matrix
K = np.array([
    [FOCAL_LENGTH, 0, 0],
    [0, FOCAL_LENGTH, 0],
    [0, 0, 1]], dtype=np.float32)


def debug_show(prefix, step, text, display):
    """
    Debug表示用のユーティリティ。
    DEBUG_OUTPUT = 'file', 'screen', 'both' に応じて動作。
    """
    if DEBUG_OUTPUT != 'screen':
        filetext = text.replace(' ', '_')
        outfile = f"{prefix}_debug_{step}_{filetext}.png"
        cv2.imwrite(outfile, display)

    if DEBUG_OUTPUT != 'file':
        image = display.copy()
        height = image.shape[0]

        cv2.putText(image, text, (16, height - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 0), 3, cv2.LINE_AA)

        cv2.putText(image, text, (16, height - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, image)

        while cv2.waitKey(5) < 0:
            pass


def round_nearest_multiple(i, factor):
    i = int(i)
    rem = i % factor
    if not rem:
        return i
    else:
        return i + factor - rem


def pix2norm(shape, pts):
    """
    画像座標系(ピクセル) -> 正規化座標系(中心を原点、最大の辺を±1)
    """
    height, width = shape[:2]
    scl = 2.0 / (max(height, width))
    offset = np.array([width, height], dtype=pts.dtype).reshape((-1, 1, 2)) * 0.5
    return (pts - offset) * scl


def norm2pix(shape, pts, as_integer):
    """
    正規化座標系(中心を原点、最大の辺を±1) -> 画像座標系(ピクセル)
    """
    height, width = shape[:2]
    scl = max(height, width) * 0.5
    offset = np.array([0.5 * width, 0.5 * height],
                      dtype=pts.dtype).reshape((-1, 1, 2))
    rval = pts * scl + offset
    if as_integer:
        return (rval + 0.5).astype(int)
    else:
        return rval


def fltp(point):
    return tuple(point.astype(int).flatten())


def draw_correspondences(img, dstpoints, projpts):
    """
    2Dキー点(dstpoints)と投影後のキー点(projpts)を画像上に描画。
    """
    display = img.copy()
    dstpoints = norm2pix(img.shape, dstpoints, True)
    projpts = norm2pix(img.shape, projpts, True)

    for pts, color in [(projpts, (255, 0, 0)), (dstpoints, (0, 0, 255))]:
        for point in pts:
            cv2.circle(display, fltp(point), 3, color, -1, cv2.LINE_AA)

    for point_a, point_b in zip(projpts, dstpoints):
        cv2.line(display, fltp(point_a), fltp(point_b),
                 (255, 255, 255), 1, cv2.LINE_AA)

    return display


def get_default_params(corners, ycoords, xcoords):
    """
    初期パラメータ(カメラ姿勢、ページ寸法、カーブ係数等)を推定する。
    """
    # page width and height
    page_width = np.linalg.norm(corners[1] - corners[0])
    page_height = np.linalg.norm(corners[-1] - corners[0])
    rough_dims = (page_width, page_height)

    # our initial guess for the cubic has no slope
    cubic_slopes = [0.0, 0.0]

    # object points of flat page in 3D coordinates
    corners_object3d = np.array([
        [0, 0, 0],
        [page_width, 0, 0],
        [page_width, page_height, 0],
        [0, page_height, 0]])

    # estimate rotation and translation from four 2D-to-3D point
    # correspondences
    _, rvec, tvec = cv2.solvePnP(corners_object3d,
                                 corners, K, np.zeros(5))

    span_counts = [len(xc) for xc in xcoords]

    params = np.hstack((np.array(rvec).flatten(),
                        np.array(tvec).flatten(),
                        np.array(cubic_slopes).flatten(),
                        ycoords.flatten()) +
                       tuple(xcoords))

    return rough_dims, span_counts, params


def project_xy(xy_coords, pvec):
    """
    ページ上の (x, y) 座標から3次元にマッピングし、画像平面へ投影。
    """
    alpha, beta = tuple(pvec[CUBIC_IDX])

    # f(0) = 0, f'(0) = alpha, f(1) = 0, f'(1) = beta
    # 上記条件から導出される三次多項式係数
    poly = np.array([
        alpha + beta,
        -2 * alpha + -1 * beta,
        alpha,
        0
    ])

    xy_coords = xy_coords.reshape((-1, 2))
    z_coords = np.polyval(poly, xy_coords[:, 0])

    objpoints = np.hstack((xy_coords, z_coords.reshape((-1, 1))))

    image_points, _ = cv2.projectPoints(objpoints,
                                        pvec[RVEC_IDX],
                                        pvec[TVEC_IDX],
                                        K, np.zeros(5))

    return image_points


def project_keypoints(pvec, keypoint_index):
    """
    パラメータ pvec を用いてキー点の画像座標を投影。
    """
    xy_coords = pvec[keypoint_index]
    xy_coords[0, :] = 0  # 先頭は corners[0]扱い(=0?) -> 原コード準拠

    return project_xy(xy_coords, pvec)


def resize_to_screen(src, maxw=1280, maxh=700, copy=False):
    """
    表示用に大きすぎる場合に画面サイズに合わせて縮小。
    """
    height, width = src.shape[:2]

    scl_x = float(width) / maxw
    scl_y = float(height) / maxh

    scl = int(np.ceil(max(scl_x, scl_y)))

    if scl > 1.0:
        inv_scl = 1.0 / scl
        img = cv2.resize(src, (0, 0), None, inv_scl, inv_scl, cv2.INTER_AREA)
    elif copy:
        img = src.copy()
    else:
        img = src

    return img


def box(width, height):
    return np.ones((height, width), dtype=np.uint8)


def get_page_extents(small):
    """
    画像の周囲を一定マージンで切り取ったページ領域のマスクと、その矩形輪郭を取得。
    """
    height, width = small.shape[:2]

    xmin = PAGE_MARGIN_X
    ymin = PAGE_MARGIN_Y
    xmax = width - PAGE_MARGIN_X
    ymax = height - PAGE_MARGIN_Y

    page = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(page, (xmin, ymin), (xmax, ymax), (255, 255, 255), -1)

    outline = np.array([
        [xmin, ymin],
        [xmin, ymax],
        [xmax, ymax],
        [xmax, ymin]
    ])

    return page, outline


def get_mask(small, pagemask, masktype):
    """
    テキストまたは線を拾うための2値マスク生成。
    """
    sgray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

    if masktype == 'text':
        mask = cv2.adaptiveThreshold(sgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV,
                                     ADAPTIVE_WINSZ,
                                     25)

        if DEBUG_LEVEL >= 3:
            debug_show('debug', 0.1, 'thresholded', mask)

        mask = cv2.dilate(mask, box(9, 1))

        if DEBUG_LEVEL >= 3:
            debug_show('debug', 0.2, 'dilated', mask)

        mask = cv2.erode(mask, box(1, 3))

        if DEBUG_LEVEL >= 3:
            debug_show('debug', 0.3, 'eroded', mask)
    else:
        mask = cv2.adaptiveThreshold(sgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV,
                                     ADAPTIVE_WINSZ,
                                     7)
        if DEBUG_LEVEL >= 3:
            debug_show('debug', 0.4, 'thresholded', mask)

        mask = cv2.erode(mask, box(3, 1), iterations=3)

        if DEBUG_LEVEL >= 3:
            debug_show('debug', 0.5, 'eroded', mask)

        mask = cv2.dilate(mask, box(8, 2))

        if DEBUG_LEVEL >= 3:
            debug_show('debug', 0.6, 'dilated', mask)

    return np.minimum(mask, pagemask)


def interval_measure_overlap(int_a, int_b):
    return min(int_a[1], int_b[1]) - max(int_a[0], int_b[0])


def angle_dist(angle_b, angle_a):
    diff = angle_b - angle_a
    while diff > np.pi:
        diff -= 2 * np.pi
    while diff < -np.pi:
        diff += 2 * np.pi
    return np.abs(diff)


def blob_mean_and_tangent(contour):
    """
    与えられた輪郭のモーメントから中心と主成分(接線方向)を求める。
    """
    moments = cv2.moments(contour)
    area = moments['m00']

    if area < 1e-7:
        area = 1e-7

    mean_x = moments['m10'] / area
    mean_y = moments['m01'] / area

    moments_matrix = np.array([
        [moments['mu20'], moments['mu11']],
        [moments['mu11'], moments['mu02']]
    ]) / area

    # PCA
    _, svd_u, _ = cv2.SVDecomp(moments_matrix)
    center = np.array([mean_x, mean_y])
    tangent = svd_u[:, 0].flatten().copy()

    return center, tangent


class ContourInfo:
    """
    テキストや線の輪郭オブジェクトを保持し、位置や向きなどを計算・格納するクラス。
    """
    def __init__(self, contour, rect, mask):
        self.contour = contour
        self.rect = rect
        self.mask = mask

        self.center, self.tangent = blob_mean_and_tangent(contour)
        self.angle = np.arctan2(self.tangent[1], self.tangent[0])

        clx = [self.proj_x(pt) for pt in contour]
        lxmin = min(clx)
        lxmax = max(clx)

        self.local_xrng = (lxmin, lxmax)
        self.point0 = self.center + self.tangent * lxmin
        self.point1 = self.center + self.tangent * lxmax

        self.pred = None
        self.succ = None

    def proj_x(self, point):
        return np.dot(self.tangent, point.flatten() - self.center)

    def local_overlap(self, other):
        xmin = self.proj_x(other.point0)
        xmax = self.proj_x(other.point1)
        return interval_measure_overlap(self.local_xrng, (xmin, xmax))


def generate_candidate_edge(cinfo_a, cinfo_b):
    """
    cinfo_a -> cinfo_b の連結が妥当かどうか判定し、スコアと一緒に返す。
    """
    # 左右の順序を保証 (x座標が小さい方が left とみなす)
    if cinfo_a.point0[0] > cinfo_b.point1[0]:
        cinfo_a, cinfo_b = cinfo_b, cinfo_a

    x_overlap_a = cinfo_a.local_overlap(cinfo_b)
    x_overlap_b = cinfo_b.local_overlap(cinfo_a)

    overall_tangent = cinfo_b.center - cinfo_a.center
    overall_angle = np.arctan2(overall_tangent[1], overall_tangent[0])

    delta_angle = max(angle_dist(cinfo_a.angle, overall_angle),
                      angle_dist(cinfo_b.angle, overall_angle)) * 180 / np.pi

    x_overlap = max(x_overlap_a, x_overlap_b)
    dist = np.linalg.norm(cinfo_b.point0 - cinfo_a.point1)

    if (dist > EDGE_MAX_LENGTH or
            x_overlap > EDGE_MAX_OVERLAP or
            delta_angle > EDGE_MAX_ANGLE):
        return None
    else:
        score = dist + delta_angle * EDGE_ANGLE_COST
        return (score, cinfo_a, cinfo_b)


def make_tight_mask(contour, xmin, ymin, width, height):
    """
    輪郭のバウンディングボックスに合わせた小領域マスクを生成。
    """
    tight_mask = np.zeros((height, width), dtype=np.uint8)
    tight_contour = contour - np.array((xmin, ymin)).reshape((-1, 1, 2))
    cv2.drawContours(tight_mask, [tight_contour], 0, (1, 1, 1), -1)
    return tight_mask


def get_contours(small, pagemask, masktype):
    """
    指定masktype('text' or 'line')に応じた輪郭を抽出し、ContourInfoとして返す。
    """
    mask = get_mask(small, pagemask, masktype)

    # Python3以降のcv2.findContoursの戻り値は (contours, hierarchy)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)

    contours_out = []
    for contour in contours:
        rect = cv2.boundingRect(contour)
        xmin, ymin, width, height = rect

        if (width < TEXT_MIN_WIDTH or
                height < TEXT_MIN_HEIGHT or
                width < TEXT_MIN_ASPECT * height):
            continue

        tight_mask = make_tight_mask(contour, xmin, ymin, width, height)
        if tight_mask.sum(axis=0).max() > TEXT_MAX_THICKNESS:
            continue

        contours_out.append(ContourInfo(contour, rect, tight_mask))

    if DEBUG_LEVEL >= 2:
        visualize_contours(small, contours_out)

    return contours_out


def assemble_spans(small, pagemask, cinfo_list):
    """
    連結可能な輪郭を繋げてスパン(列)としてまとまりを作る。
    """
    cinfo_list = sorted(cinfo_list, key=lambda c: c.rect[1])

    # generate all candidate edges
    candidate_edges = []
    for i, cinfo_i in enumerate(cinfo_list):
        for j in range(i):
            edge = generate_candidate_edge(cinfo_i, cinfo_list[j])
            if edge is not None:
                candidate_edges.append(edge)

    # sort candidate edges by score (lower is better)
    candidate_edges.sort()

    # 実際に連結してスパンを作る
    for _, cinfo_a, cinfo_b in candidate_edges:
        if cinfo_a.succ is None and cinfo_b.pred is None:
            cinfo_a.succ = cinfo_b
            cinfo_b.pred = cinfo_a

    spans = []
    while cinfo_list:
        cinfo = cinfo_list[0]
        while cinfo.pred:
            cinfo = cinfo.pred
        cur_span = []
        width = 0.0
        while cinfo:
            cinfo_list.remove(cinfo)
            cur_span.append(cinfo)
            width += cinfo.local_xrng[1] - cinfo.local_xrng[0]
            cinfo = cinfo.succ

        if width > SPAN_MIN_WIDTH:
            spans.append(cur_span)

    if DEBUG_LEVEL >= 2:
        visualize_spans(small, pagemask, spans)

    return spans


def sample_spans(shape, spans):
    """
    各スパン(列)上を一定ステップでサンプリングし、正規化座標系で返す。
    """
    span_points = []
    for span in spans:
        contour_points = []
        for cinfo in span:
            yvals = np.arange(cinfo.mask.shape[0]).reshape((-1, 1))
            totals = (yvals * cinfo.mask).sum(axis=0)
            means = totals / cinfo.mask.sum(axis=0)

            xmin, ymin = cinfo.rect[:2]
            step = SPAN_PX_PER_STEP
            start = ((len(means) - 1) % step) / 2
            contour_points += [(x + xmin, means[x] + ymin)
                               for x in range(int(start), len(means), step)]

        contour_points = np.array(contour_points, dtype=np.float32).reshape((-1, 1, 2))
        contour_points = pix2norm(shape, contour_points)
        span_points.append(contour_points)
    return span_points


def keypoints_from_samples(small, pagemask, page_outline, span_points):
    """
    サンプリング点群からページの4隅と、各スパンの x, y 座標を求める。
    """
    # 全スパンの接線方向の重み付き平均ベクトルを求める
    all_evecs = np.array([[0.0, 0.0]])
    all_weights = 0

    for points in span_points:
        _, evec = cv2.PCACompute(points.reshape((-1, 2)), None, maxComponents=1)
        weight = np.linalg.norm(points[-1] - points[0])
        all_evecs += evec * weight
        all_weights += weight

    evec = all_evecs / all_weights
    x_dir = evec.flatten()
    if x_dir[0] < 0:
        x_dir = -x_dir
    y_dir = np.array([-x_dir[1], x_dir[0]])

    pagecoords = cv2.convexHull(page_outline)
    pagecoords = pix2norm(pagemask.shape, pagecoords.reshape((-1, 1, 2)))
    pagecoords = pagecoords.reshape((-1, 2))

    px_coords = np.dot(pagecoords, x_dir)
    py_coords = np.dot(pagecoords, y_dir)
    px0 = px_coords.min()
    px1 = px_coords.max()
    py0 = py_coords.min()
    py1 = py_coords.max()

    p00 = px0 * x_dir + py0 * y_dir
    p10 = px1 * x_dir + py0 * y_dir
    p11 = px1 * x_dir + py1 * y_dir
    p01 = px0 * x_dir + py1 * y_dir
    corners = np.vstack((p00, p10, p11, p01)).reshape((-1, 1, 2))

    ycoords = []
    xcoords = []
    for points in span_points:
        pts = points.reshape((-1, 2))
        px_coords = np.dot(pts, x_dir)
        py_coords = np.dot(pts, y_dir)
        ycoords.append(py_coords.mean() - py0)
        xcoords.append(px_coords - px0)

    if DEBUG_LEVEL >= 2:
        visualize_span_points(small, span_points, corners)

    return corners, np.array(ycoords), xcoords


def visualize_contours(small, cinfo_list):
    """
    テキスト輪郭を色分けして描画 (デバッグ用)。
    """
    regions = np.zeros_like(small)
    for j, cinfo in enumerate(cinfo_list):
        cv2.drawContours(regions, [cinfo.contour], 0,
                         CCOLORS[j % len(CCOLORS)], -1)

    mask = (regions.max(axis=2) != 0)
    display = small.copy()
    display[mask] = (display[mask] / 2) + (regions[mask] / 2)

    for j, cinfo in enumerate(cinfo_list):
        cv2.circle(display, fltp(cinfo.center), 3, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(display, fltp(cinfo.point0), fltp(cinfo.point1),
                 (255, 255, 255), 1, cv2.LINE_AA)

    debug_show('debug', 1, 'contours', display)


def visualize_spans(small, pagemask, spans):
    """
    スパン(列)ごとに色分けして描画 (デバッグ用)。
    """
    regions = np.zeros_like(small)
    for i, span in enumerate(spans):
        contours = [cinfo.contour for cinfo in span]
        cv2.drawContours(regions, contours, -1,
                         CCOLORS[i * 3 % len(CCOLORS)], -1)

    mask = (regions.max(axis=2) != 0)
    display = small.copy()
    display[mask] = (display[mask] / 2) + (regions[mask] / 2)
    display[pagemask == 0] /= 4

    debug_show('debug', 2, 'spans', display)


def visualize_span_points(small, span_points, corners):
    """
    スパンのサンプル点やページ4隅を描画 (デバッグ用)。
    """
    display = small.copy()
    for i, points in enumerate(span_points):
        points = norm2pix(small.shape, points, False)
        mean, small_evec = cv2.PCACompute(points.reshape((-1, 2)), None, maxComponents=1)
        dps = np.dot(points.reshape((-1, 2)), small_evec.reshape((2, 1)))
        dpm = np.dot(mean.flatten(), small_evec.flatten())
        point0 = mean + small_evec * (dps.min() - dpm)
        point1 = mean + small_evec * (dps.max() - dpm)

        for point in points:
            cv2.circle(display, fltp(point), 3,
                       CCOLORS[i % len(CCOLORS)], -1, cv2.LINE_AA)

        cv2.line(display, fltp(point0), fltp(point1),
                 (255, 255, 255), 1, cv2.LINE_AA)

    cv2.polylines(display, [norm2pix(small.shape, corners, True)],
                  True, (255, 255, 255))

    debug_show('debug', 3, 'span_points', display)


def make_keypoint_index(span_counts):
    """
    スパンごとにキー点のindexを割り当てる。
    """
    nspans = len(span_counts)
    npts = sum(span_counts)
    keypoint_index = np.zeros((npts + 1, 2), dtype=int)
    start = 1

    for i, count in enumerate(span_counts):
        end = start + count
        keypoint_index[start:start + end, 1] = 8 + i
        start = end

    keypoint_index[1:, 0] = np.arange(npts) + 8 + nspans
    return keypoint_index


def optimize_params(small, dstpoints, span_counts, params):
    """
    Powell法でキー点の二乗誤差を最小化するように最適化する。
    """
    keypoint_index = make_keypoint_index(span_counts)

    def objective(pvec):
        ppts = project_keypoints(pvec, keypoint_index)
        return np.sum((dstpoints - ppts) ** 2)

    print("  initial objective is", objective(params))

    if DEBUG_LEVEL >= 1:
        projpts = project_keypoints(params, keypoint_index)
        display = draw_correspondences(small, dstpoints, projpts)
        debug_show('debug', 4, 'keypoints_before', display)

    print("  optimizing", len(params), "parameters...")
    res = scipy.optimize.minimize(objective, params, method='Powell')
    print("  final objective is", res.fun)
    params = res.x

    if DEBUG_LEVEL >= 1:
        projpts = project_keypoints(params, keypoint_index)
        display = draw_correspondences(small, dstpoints, projpts)
        debug_show('debug', 5, 'keypoints_after', display)

    return params


def get_page_dims(corners, rough_dims, params):
    """
    ページ右下コーナーを使ってページの実寸(幅x高さ)を調整する。
    """
    dst_br = corners[2].flatten()
    dims = np.array(rough_dims)

    def objective(d):
        proj_br = project_xy(d, params)
        return np.sum((dst_br - proj_br.flatten()) ** 2)

    res = scipy.optimize.minimize(objective, dims, method='Powell')
    dims = res.x

    print("  got page dims", dims[0], "x", dims[1])
    return dims


def remap_image(img, small, page_dims, params):
    height = 0.5 * page_dims[1] * OUTPUT_ZOOM * img.shape[0]
    height = round_nearest_multiple(height, REMAP_DECIMATE)
    width = round_nearest_multiple(height * page_dims[0] / page_dims[1], REMAP_DECIMATE)

    print("  output will be {}x{}".format(int(width), int(height)))

    height_small = int(height / REMAP_DECIMATE)
    width_small = int(width / REMAP_DECIMATE)

    page_x_range = np.linspace(0, page_dims[0], width_small)
    page_y_range = np.linspace(0, page_dims[1], height_small)
    page_x_coords, page_y_coords = np.meshgrid(page_x_range, page_y_range)
    page_xy_coords = np.hstack((page_x_coords.flatten().reshape((-1, 1)),
                                page_y_coords.flatten().reshape((-1, 1))))
    page_xy_coords = page_xy_coords.astype(np.float32)

    image_points = project_xy(page_xy_coords, params)
    image_points = norm2pix(img.shape, image_points, False)

    # リサイズして float32 にそろえる
    map_x = image_points[:, 0, 0].reshape(page_x_coords.shape)
    map_y = image_points[:, 0, 1].reshape(page_y_coords.shape)
    map_x = cv2.resize(map_x, (int(width), int(height)), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    map_y = cv2.resize(map_y, (int(width), int(height)), interpolation=cv2.INTER_CUBIC).astype(np.float32)

    # ---- 白黒2値化用 ----
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    remapped = cv2.remap(img_gray, map_x, map_y,
                         interpolation=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REPLICATE)
    thresh = cv2.adaptiveThreshold(remapped, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY,
                                   ADAPTIVE_WINSZ, 25)

    # ---- カラーで同じ補正をかける(赤色領域の抽出用) ----
    remapped_color = cv2.remap(img, map_x, map_y,
                               interpolation=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE).astype(np.uint8)

    # HSVで赤色を抽出
    hsv = cv2.cvtColor(remapped_color, cv2.COLOR_RGB2HSV)

    # 「赤色」と判定する範囲(例)
    # ここは要件次第で微調整してください
    lower_red1 = np.array([0,  60,  60], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([170, 60,  60], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)  # 0 or 255

    # ---- 最終出力(白黒+赤)を作る ----
    H, W = thresh.shape
    final_image = np.zeros((H, W, 3), dtype=np.uint8)  # BGRでもRGBでもOK

    # (1) 赤マスクがONのところを赤色に
    final_image[red_mask > 0] = (0, 0, 255)  # OpenCVはBGRなので赤は(0,0,255)

    # (2) 赤じゃなく、かつ thresh=白 のところを白に
    #     thresh が 255 -> 背景(白)
    white_mask = (red_mask == 0) & (thresh == 255)
    final_image[white_mask] = (255, 255, 255)

    # (3) それ以外は黒のまま(文字は黒)
    #     つまり初期値(0,0,0)が残る

    if DEBUG_LEVEL >= 1:
        h_small = small.shape[0]
        w_small = int(round(h_small * float(W) / H))
        debug_disp = cv2.resize(final_image, (w_small, h_small), interpolation=cv2.INTER_AREA)
        debug_show('debug', 6, 'output', debug_disp)

    return final_image


######################################################################
# E2E インターフェイス
#   画像(np.ndarray)を受け取り、補正後の画像(np.ndarray)を返す。
######################################################################
def page_dewarp(img: np.ndarray) -> np.ndarray:
    """
    入力: img (カラー画像 or グレースケール画像, OpenCV形式のndarray)
    出力: 補正後の2値画像(ndarray)
    """
    # カラーであることを仮定し、必要に応じて調整
    if len(img.shape) == 2:
        # グレースケールの場合はカラー化
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        # RGBA等の場合、RGBに変換
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    small = resize_to_screen(img)
    print("Loaded image with size {}x{}, resized to {}x{}".format(
        img.shape[1], img.shape[0], small.shape[1], small.shape[0]))

    if DEBUG_LEVEL >= 3:
        debug_show('debug', 0.0, 'original', small)

    # ページ範囲抽出
    pagemask, page_outline = get_page_extents(small)

    # テキスト用輪郭抽出
    cinfo_list = get_contours(small, pagemask, 'text')
    spans = assemble_spans(small, pagemask, cinfo_list)

    # テキストスパンが十分に少ない場合、線として再度輪郭抽出
    if len(spans) < 3:
        print("  detecting lines because only", len(spans), "text spans")
        cinfo_list_line = get_contours(small, pagemask, 'line')
        spans2 = assemble_spans(small, pagemask, cinfo_list_line)
        if len(spans2) > len(spans):
            spans = spans2

    if len(spans) < 1:
        print("No valid spans detected. Returning original image.")
        # スパンがなければ何もせずリターン
        return cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

    span_points = sample_spans(small.shape, spans)
    print("  got", len(spans), "spans",
          "with", sum([len(pts) for pts in span_points]), "points.")

    corners, ycoords, xcoords = keypoints_from_samples(small, pagemask,
                                                       page_outline, span_points)

    rough_dims, span_counts, params = get_default_params(corners,
                                                         ycoords, xcoords)

    # dstpoints: corners[0] + span_points(全て) を結合したもの
    dstpoints = np.vstack((corners[0].reshape((1, 1, 2)),) +
                          tuple(span_points))

    params = optimize_params(small, dstpoints, span_counts, params)
    page_dims = get_page_dims(corners, rough_dims, params)
    out_image = remap_image(img, small, page_dims, params)

    print("Dewarp finished.")
    return out_image
