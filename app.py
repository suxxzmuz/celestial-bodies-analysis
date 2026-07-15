import os
import cv2
import numpy as np
import base64
import sqlite3
import math
from datetime import datetime
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# [공통 설정] 이미지 분석 타겟 크기
TARGET_SIZE = 600
TARGET_RADIUS = 250
TARGET_CENTER = (TARGET_SIZE // 2, TARGET_SIZE // 2)

# ==========================================
# 0. 데이터베이스(SQLite) 설정
# ==========================================
def init_db():
    conn = sqlite3.connect('astrovision.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS community (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            category TEXT NOT NULL,
            description TEXT NOT NULL,
            image_base64 TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def get_db_connection():
    conn = sqlite3.connect('astrovision.db')
    conn.row_factory = sqlite3.Row 
    return conn

# ==========================================
# 1. 웹페이지 화면 이동 (라우팅)
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/main')
def main_page():
    return render_template('main.html')

@app.route('/cat-sun-analysis')
def cat_sun_page():
    return render_template('cat-sun-analysis.html')

@app.route('/sun-single-image')
def single_page():
    return render_template('sun-single-image.html')

@app.route('/sun-continuous-image')
def series_page():
    return render_template('sun-continuous-image.html')

# ==========================================
# 커뮤니티 게시판 DB 통신 API
# ==========================================
@app.route('/api/posts', methods=['GET'])
def get_posts():
    conn = get_db_connection()
    posts = conn.execute('SELECT * FROM community ORDER BY id DESC').fetchall()
    conn.close()
    post_list = [dict(post) for post in posts]
    return jsonify({'status': 'success', 'data': post_list})

@app.route('/api/posts', methods=['POST'])
def add_post():
    data = request.json
    title = data.get('title')
    category = data.get('category')
    desc = data.get('description')
    image_base64 = data.get('image') 

    conn = get_db_connection()
    conn.execute('''
        INSERT INTO community (title, category, description, image_base64)
        VALUES (?, ?, ?, ?)
    ''', (title, category, desc, image_base64))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success'})

# ==========================================
# 2. 단일 영상 분석 엔진 
# ==========================================
@app.route('/analyze_single', methods=['POST'])
def analyze_single():
    if 'image' not in request.files:
        return jsonify({'status': 'fail', 'message': '이미지 파일이 없습니다.'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'status': 'fail', 'message': '선택된 파일이 없습니다.'})

    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({'status': 'fail', 'message': '이미지를 불러올 수 없습니다.'})

        h_orig, w_orig = image.shape[:2]
        scale = 0.2
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

        gray      = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        circles = cv2.HoughCircles(
            gray_blur, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=100,
            param1=100, param2=30,
            minRadius=100, maxRadius=500
        )
        if circles is None:
            return jsonify({'status': 'fail', 'message': '이미지에서 태양을 검출하지 못했습니다.'})

        circles = np.round(circles[0, :]).astype("int")
        x, y, r = circles[0]
        sun_area = np.pi * (r ** 2)

        mask   = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        masked = cv2.bitwise_and(gray, gray, mask=mask)  

        sun_pixels = masked[mask == 255]
        mean_brightness = float(np.mean(sun_pixels))
        std_brightness  = float(np.std(sun_pixels))

        thresh_param  = int(request.form.get('threshValue',  1))
        min_spot_size = int(request.form.get('minSpotSize',  5))
        max_spot_size = int(request.form.get('maxSpotSize',  2)) 

        spots = cv2.adaptiveThreshold(
            masked, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31, thresh_param
        )
        spots = cv2.bitwise_and(spots, spots, mask=mask)

        contours, _ = cv2.findContours(spots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        spots_info = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_spot_size or area > max_spot_size:
                continue
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            ratio = (area / sun_area) * 100
            spots_info.append({
                'cx': cx, 'cy': cy,
                'area': round(area, 2),
                'ratio': round(ratio, 5)
            })

        spots_info.sort(key=lambda s: s["cx"])

        results    = image.copy()
        spot_data  = []
        spot_count = 0

        for i, spot in enumerate(spots_info, start=1):
            spot_count += 1
            cx, cy = spot['cx'], spot['cy']
            area, ratio = spot['area'], spot['ratio']

            spot_data.append({
                'id':    i,
                'cx':    cx,
                'cy':    cy,
                'area':  area,
                'ratio': ratio
            })

            cv2.circle(results, (cx, cy), 8, (255, 0, 0), 2)
            cv2.putText(results, str(i), (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.circle(results, (x, y), r, (0, 255, 0), 3)
        cv2.circle(results, (x, y), 3, (255, 0, 0), -1)

        _, buffer   = cv2.imencode('.jpg', results)
        img_base64  = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'status': 'success',
            'sun_info': {
                'center_x': int(x),
                'center_y': int(y),
                'radius':   int(r),
                'sun_area': round(sun_area, 2)
            },
            'brightness': {
                'mean': round(mean_brightness, 4),
                'std':  round(std_brightness,  4)
            },
            'spot_count':  spot_count,
            'data':        spot_data,
            'image_base64': img_base64
        })

    except Exception as e:
        return jsonify({'status': 'fail', 'message': f'분석 중 에러 발생: {str(e)}'})

# ==========================================
# 3. 연속 영상 분석 엔진 (main_series.py 완벽 동일 포팅)
# ==========================================

def rotate_point(x, y, center_x, center_y, angle_deg):
    """main_series.py와 동일한 좌표 회전 함수"""
    theta = math.radians(angle_deg)
    rel_x = x - center_x
    rel_y = y - center_y
    rot_x = rel_x * math.cos(theta) + rel_y * math.sin(theta)
    rot_y = -rel_x * math.sin(theta) + rel_y * math.cos(theta)
    return rot_x, rot_y


def normalize_sun_image(image):
    """
    main_series.py와 동일한 태양 원판 정규화:
    1) 0.2 배율 리사이즈
    2) HoughCircles 로 태양 검출
    3) TARGET_RADIUS/r 비율로 스케일 후 crop-paste 방식으로 TARGET_CENTER 정렬
    반환: (normalized_img | None, sun_x, sun_y, sun_r)
    """
    # ── Step 1: 0.2 배율 축소 (main_series.py 동일) ──
    image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=100,
        param1=100, param2=30,
        minRadius=100, maxRadius=500
    )
    if circles is None:
        return None, None, None, None

    x, y, r = [int(v) for v in np.round(circles[0]).astype(int)[0]]

    # ── Step 2: TARGET_RADIUS/r 스케일 resize ──
    scale = TARGET_RADIUS / r
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    new_x = int(x * scale)
    new_y = int(y * scale)

    # ── Step 3: crop-paste 로 중심 정렬 (main_series.py 동일) ──
    target_x, target_y = TARGET_CENTER
    normalized = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
    start_x = target_x - new_x
    start_y = target_y - new_y

    src_x1 = max(0, -start_x);  src_y1 = max(0, -start_y)
    src_x2 = min(resized.shape[1], TARGET_SIZE - start_x)
    src_y2 = min(resized.shape[0], TARGET_SIZE - start_y)

    dst_x1 = max(0, start_x);   dst_y1 = max(0, start_y)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    normalized[dst_y1:dst_y2, dst_x1:dst_x2] = resized[src_y1:src_y2, src_x1:src_x2]
    return normalized, x, y, r


def detect_spots_main(normalized):
    """
    main_series.py의 detect_spots() 와 완전히 동일한 로직:
    - GaussianBlur(5,5)
    - 원형 마스크 적용
    - adaptiveThreshold(ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 31, 8)  ← 핵심 노이즈 제거
    - 면적 필터: 2 px² 미만 또는 500 px² 초과 제거
    """
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 원형 마스크 (태양 원판 영역만)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, TARGET_CENTER, TARGET_RADIUS, 255, -1)
    masked = cv2.bitwise_and(gray, gray, mask=mask)

    # ── adaptiveThreshold: 노이즈 제거의 핵심 (main_series.py 동일 파라미터) ──
    spots_mask = cv2.adaptiveThreshold(
        masked, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 8
    )
    spots_mask = cv2.bitwise_and(spots_mask, spots_mask, mask=mask)

    contours, _ = cv2.findContours(spots_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    spot_list = []
    for contour in contours:
        area = cv2.contourArea(contour)
        # ── 면적 필터: 2~500 px² (main_series.py 동일) ──
        if area < 2 or area > 500:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        spot_list.append({"cx": cx, "cy": cy, "area": round(area, 2)})

    # x 좌표 기준 정렬 (main_series.py 동일)
    spot_list.sort(key=lambda s: s["cx"])
    return spot_list


def template_match_spots(first_data, second_data):
    """
    main_series.py의 템플릿 매칭 로직 완전 동일 포팅.
    - 첫 번째 사진의 각 흑점을 템플릿으로 잘라
    - 두 번째 사진의 후보 흑점 패치와 cv2.TM_CCOEFF_NORMED 비교
    - used_candidates 로 중복 배정 방지
    - score 기준 최고 후보 선택 (threshold 없음, main_series.py 동일)
    """
    size = 15  # 템플릿 반쪽 크기 (main_series.py 동일)

    first_gray  = cv2.cvtColor(first_data["normalized"], cv2.COLOR_BGR2GRAY)
    second_gray = cv2.cvtColor(second_data["normalized"], cv2.COLOR_BGR2GRAY)

    used_candidates = set()
    matches = []

    for spot_index, first_spot in enumerate(first_data["spots"], start=1):
        first_cx = first_spot["cx"]
        first_cy = first_spot["cy"]

        tx1 = max(0, first_cx - size);  ty1 = max(0, first_cy - size)
        tx2 = min(first_gray.shape[1], first_cx + size)
        ty2 = min(first_gray.shape[0], first_cy + size)
        template = first_gray[ty1:ty2, tx1:tx2]

        best_score = -1
        best_candidate_index = None
        best_candidate = None

        for candidate_index, candidate in enumerate(second_data["spots"]):
            if candidate_index in used_candidates:
                continue

            candidate_cx = candidate["cx"]
            candidate_cy = candidate["cy"]

            cx1 = max(0, candidate_cx - size);  cy1 = max(0, candidate_cy - size)
            cx2 = min(second_gray.shape[1], candidate_cx + size)
            cy2 = min(second_gray.shape[0], candidate_cy + size)
            candidate_patch = second_gray[cy1:cy2, cx1:cx2]

            # 가장자리로 인해 크기가 다르면 비교 불가 (main_series.py 동일)
            if candidate_patch.shape != template.shape:
                continue

            score_map = cv2.matchTemplate(candidate_patch, template, cv2.TM_CCOEFF_NORMED)
            score = float(score_map[0][0])

            if score > best_score:
                best_score = score
                best_candidate_index = candidate_index
                best_candidate = candidate

        if best_candidate is not None:
            used_candidates.add(best_candidate_index)
            matched_cx = best_candidate["cx"]
            matched_cy = best_candidate["cy"]
            matches.append({
                "id":      spot_index,
                "start_x": first_cx,  "start_y": first_cy,
                "end_x":   matched_cx, "end_y":   matched_cy,
                "score":   round(best_score, 3),
                "area":    first_spot["area"]
            })

    return matches


@app.route('/analyze_series', methods=['POST'])
def analyze_series():
    files = request.files.getlist('images')
    if not files or len(files) < 2:
        return jsonify({'status': 'fail', 'message': '2장 이상의 이미지를 업로드해야 합니다.'})

    try:
        files = sorted(files, key=lambda f: f.filename)

        # ── Step 1: 전체 사진 정규화 + 흑점 검출 ──
        series_data = []
        for idx, file_obj in enumerate(files):
            file_bytes = np.frombuffer(file_obj.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                continue

            normalized, sun_x, sun_y, sun_r = normalize_sun_image(image)
            if normalized is None:
                # 태양 검출 실패 시 건너뜀
                continue

            spots = detect_spots_main(normalized)

            series_data.append({
                "index":      idx + 1,
                "filename":   file_obj.filename,
                "normalized": normalized,
                "spots":      spots,
                "sun_x":      sun_x,
                "sun_y":      sun_y,
                "sun_r":      sun_r,
            })

        if len(series_data) < 2:
            return jsonify({'status': 'fail', 'message': '태양을 검출할 수 있는 이미지가 2장 미만입니다.'})

        # ── Step 2: 첫 번째 ↔ 두 번째 사진 템플릿 매칭 (main_series.py 동일) ──
        first_data  = series_data[0]
        second_data = series_data[1]

        matches = template_match_spots(first_data, second_data)

        if not matches:
            return jsonify({'status': 'fail', 'message': '흑점을 추적하지 못했습니다. 이미지를 확인해 주세요.'})

        # ── Step 3: 대표 흑점 선정 — 태양 중심에 가장 가까운 흑점 (main_series.py 동일) ──
        center_x, center_y = TARGET_CENTER
        for m in matches:
            m["center_distance"] = math.hypot(m["start_x"] - center_x, m["start_y"] - center_y)

        rep_match = min(matches, key=lambda m: m["center_distance"])

        # ── Step 4: 추정 적도 및 자전축 (main_series.py 동일) ──
        best_dx = rep_match["end_x"] - rep_match["start_x"]
        best_dy = rep_match["end_y"] - rep_match["start_y"]
        equator_angle       = math.degrees(math.atan2(best_dy, best_dx))
        rotation_axis_angle = equator_angle + 90.0

        # ── Step 5: 각 흑점 심화 수식 (main_series.py 동일) ──
        output_data   = []
        result_visual = second_data["normalized"].copy()

        for m in matches:
            # 회전 후 좌표
            rot_start_x, rot_start_y = rotate_point(
                m["start_x"], m["start_y"], center_x, center_y, equator_angle)
            rot_end_x, rot_end_y = rotate_point(
                m["end_x"],   m["end_y"],   center_x, center_y, equator_angle)

            # 적도 방향 이동량
            equator_move = rot_end_x - rot_start_x

            # 추정 위도: 회전 좌표계의 평균 y 사용 (main_series.py 동일)
            mean_rot_y     = (rot_start_y + rot_end_y) / 2.0
            latitude_ratio = max(-1.0, min(1.0, mean_rot_y / TARGET_RADIUS))
            latitude       = math.degrees(math.asin(latitude_ratio))

            # 보정 이동각: 위도선 반지름으로 나눔 (main_series.py 동일)
            latitude_radius = TARGET_RADIUS * math.cos(math.radians(latitude))
            if abs(latitude_radius) < 1:
                rotation_angle = 0.0
            else:
                rotation_angle = math.degrees(equator_move / latitude_radius)

            dx = m["end_x"] - m["start_x"]
            dy = m["end_y"] - m["start_y"]
            move_angle = math.degrees(math.atan2(dy, dx))

            output_data.append({
                'id':               m['id'],
                'start_x':         m['start_x'],  'start_y':         m['start_y'],
                'end_x':           m['end_x'],    'end_y':           m['end_y'],
                'dx':              dx,             'dy':              dy,
                'move_angle':      round(move_angle, 2),
                'distance':        round(m['center_distance'], 2),
                'score':           m['score'],
                'rot_start_x':     round(rot_start_x, 2),
                'rot_start_y':     round(rot_start_y, 2),
                'rot_end_x':       round(rot_end_x, 2),
                'rot_end_y':       round(rot_end_y, 2),
                'equator_move':    round(equator_move, 2),
                'latitude':        round(latitude, 2),
                'rotation_angle':  round(rotation_angle, 3),
            })

            # 결과 이미지 시각화
            cv2.arrowedLine(result_visual,
                            (m['start_x'], m['start_y']),
                            (m['end_x'],   m['end_y']),
                            (0, 0, 255), 2, tipLength=0.3)
            cv2.circle(result_visual, (m['start_x'], m['start_y']), 3, (255, 0, 0), -1)
            is_rep = (m['id'] == rep_match['id'])
            label_color = (0, 215, 255) if is_rep else (0, 255, 0)
            cv2.putText(result_visual,
                        f"{'[REP]' if is_rep else ''}ID:{m['id']}",
                        (m['end_x'] + 5, m['end_y'] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, label_color, 1)

        # 태양 원 + 추정 적도선 그리기
        cv2.circle(result_visual, TARGET_CENTER, TARGET_RADIUS, (255, 255, 255), 1)
        eq_len = 300
        eq_dx = int(eq_len * math.cos(math.radians(equator_angle)))
        eq_dy = int(eq_len * math.sin(math.radians(equator_angle)))
        cv2.line(result_visual,
                 (TARGET_CENTER[0] - eq_dx, TARGET_CENTER[1] - eq_dy),
                 (TARGET_CENTER[0] + eq_dx, TARGET_CENTER[1] + eq_dy),
                 (0, 255, 255), 1)
        # 추정 자전축선 그리기
        ax_len = 280
        ax_dx = int(ax_len * math.cos(math.radians(rotation_axis_angle)))
        ax_dy = int(ax_len * math.sin(math.radians(rotation_axis_angle)))
        cv2.line(result_visual,
                 (TARGET_CENTER[0] - ax_dx, TARGET_CENTER[1] - ax_dy),
                 (TARGET_CENTER[0] + ax_dx, TARGET_CENTER[1] + ax_dy),
                 (255, 100, 0), 1)

        _, buffer = cv2.imencode('.jpg', result_visual)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # ── Step 6: 메타데이터 (정상 분석 장수 / 사진 이름 / 흑점 수 / 개별 좌표) ──
        image_summaries = []
        for data in series_data:
            image_summaries.append({
                "index":      data["index"],
                "filename":   data["filename"],
                "spot_count": len(data["spots"]),
                "spots":      [{"cx": s["cx"], "cy": s["cy"], "area": s["area"]}
                               for s in data["spots"]]
            })

        return jsonify({
            'status': 'success',
            'global_info': {
                'analyzed_count':        len(series_data),
                'representative_spot_id': rep_match['id'],
                'representative_center_distance': round(rep_match['center_distance'], 2),
                'equator_angle':          round(equator_angle, 2),
                'rotation_axis_angle':    round(rotation_axis_angle, 2),
            },
            'image_summaries': image_summaries,
            'data':            output_data,
            'image_base64':    img_base64,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'fail', 'message': f'연속 분석 중 에러 발생: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)