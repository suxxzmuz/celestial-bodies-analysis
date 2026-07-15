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

        thresh_param  = int(request.form.get('threshValue',  12))
        min_spot_size = int(request.form.get('minSpotSize',  1))
        max_spot_size = int(request.form.get('maxSpotSize',  500)) 

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
# 3. 연속 영상 분석 엔진
# ==========================================

def rotate_point(x, y, center_x, center_y, angle_deg):
    """
    지정된 각도만큼 좌표를 회전 (main_series.py 동일 수식)
    """
    theta = math.radians(angle_deg)
    rel_x = x - center_x
    rel_y = y - center_y
    rot_x = rel_x * math.cos(theta) + rel_y * math.sin(theta)
    rot_y = -rel_x * math.sin(theta) + rel_y * math.cos(theta)
    return rot_x, rot_y


def normalize_sun_image(image):
    """
    [처리 순서 1~3] main_series.py와 동일한 정규화 방식
    - 흑백 변환 → GaussianBlur(5,5) → HoughCircles로 태양 검출
    - 태양 중심을 (300,300), 반지름 250px, 600×600으로 통일
    - resize + 픽셀 복사 방식 (warpAffine 사용 안 함)
    반환: (normalized_image, sun_x, sun_y, sun_r) 또는 None
    """
    # [1] 크기 축소 (원본 고해상도 대응)
    image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2)

    # [2] 흑백 변환 + Gaussian Blur (5, 5)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # [3] Hough Circle로 태양 중심(x, y)과 반지름 r 검출
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=100,
        param1=100, param2=30,
        minRadius=100, maxRadius=500
    )
    if circles is None:
        return None, None, None, None

    circles = np.round(circles[0]).astype(int)
    x, y, r = circles[0]

    # [4] 정규화: 태양 중심→(300,300), 반지름→250px, 크기→600×600
    #     main_series.py 와 동일한 resize + 픽셀 복사 방식
    target_x, target_y = TARGET_CENTER
    scale = TARGET_RADIUS / r
    resized = cv2.resize(image, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_LINEAR)
    new_x = int(x * scale)
    new_y = int(y * scale)

    normalized = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
    start_x = target_x - new_x
    start_y = target_y - new_y

    src_x1 = max(0, -start_x)
    src_y1 = max(0, -start_y)
    src_x2 = min(resized.shape[1], TARGET_SIZE - start_x)
    src_y2 = min(resized.shape[0], TARGET_SIZE - start_y)

    dst_x1 = max(0, start_x)
    dst_y1 = max(0, start_y)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    normalized[dst_y1:dst_y2, dst_x1:dst_x2] = \
        resized[src_y1:src_y2, src_x1:src_x2]

    return normalized, x, y, r


def detect_spots_series(normalized, thresh_c, min_spot_size, max_spot_size):
    """
    [처리 순서 4~7] main_series.py의 detect_spots() 완전 동일 구현
    ✅ 흑백 변환 → GaussianBlur(5,5) → 원형 마스크 → adaptiveThreshold
    ✅ 노이즈 제거: area < min_spot_size or area > max_spot_size
    - thresh_c: adaptiveThreshold의 C 파라미터 (사용자 설정, 기본 8)
    - min_spot_size: 최소 흑점 면적 (사용자 설정, 기본 2)
    - max_spot_size: 최대 흑점 면적 (사용자 설정, 기본 500)
    """
    # [4] 정규화 이미지 흑백 변환
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)

    # [5] Gaussian Blur (5, 5) 적용 — main_series.py 동일
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # [6] 원형 마스크: 태양 원판 내부만 분석 (바깥 배경 제외)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, TARGET_CENTER, TARGET_RADIUS, 255, -1)
    masked = cv2.bitwise_and(gray, gray, mask=mask)

    # [7] Adaptive Threshold + THRESH_BINARY_INV → 어두운 흑점을 흰색으로
    #     blockSize=31, C=thresh_c (사용자 설정값)
    spots_mask = cv2.adaptiveThreshold(
        masked, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, thresh_c
    )
    # 마스크 다시 AND: 원판 바깥 노이즈 완전 차단
    spots_mask = cv2.bitwise_and(spots_mask, spots_mask, mask=mask)

    # [8] 윤곽선 검출 (main_series.py: RETR_LIST)
    contours, _ = cv2.findContours(
        spots_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    spot_list = []
    for contour in contours:
        area = cv2.contourArea(contour)

        # ✅ [노이즈 제거] min_spot_size 미만 or max_spot_size 초과 제거
        #    main_series.py: area < 2 or area > 500 → 사용자 값으로 대체
        if area < min_spot_size or area > max_spot_size:
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


def template_match_spots(first_normalized, first_spots,
                         second_normalized, second_spots,
                         score_threshold):
    """
    [처리 순서 - 템플릿 매칭] main_series.py 동일 구현
    - 첫 번째 사진의 각 흑점 → 두 번째 사진 후보와 패치 비교
    - 중복 배정 방지 (used_candidates set)
    - score_threshold 이상인 매칭만 유효로 인정
    반환: matches 리스트
    """
    first_gray = cv2.cvtColor(first_normalized, cv2.COLOR_BGR2GRAY)
    second_gray = cv2.cvtColor(second_normalized, cv2.COLOR_BGR2GRAY)

    matches = []
    used_candidates = set()
    patch_half = 15  # 템플릿 반쪽 크기 (main_series.py: size=15)

    for spot_index, first_spot in enumerate(first_spots, start=1):
        first_cx = first_spot["cx"]
        first_cy = first_spot["cy"]

        # 첫 번째 사진에서 템플릿 패치 자르기
        tx1 = max(0, first_cx - patch_half)
        ty1 = max(0, first_cy - patch_half)
        tx2 = min(first_gray.shape[1], first_cx + patch_half)
        ty2 = min(first_gray.shape[0], first_cy + patch_half)
        template = first_gray[ty1:ty2, tx1:tx2]

        best_score = -1.0
        best_candidate_index = None
        best_candidate = None

        # 두 번째 사진의 흑점 후보와 패치 비교
        for cand_idx, candidate in enumerate(second_spots):
            if cand_idx in used_candidates:
                continue

            cand_cx = candidate["cx"]
            cand_cy = candidate["cy"]

            cx1 = max(0, cand_cx - patch_half)
            cy1 = max(0, cand_cy - patch_half)
            cx2 = min(second_gray.shape[1], cand_cx + patch_half)
            cy2 = min(second_gray.shape[0], cand_cy + patch_half)
            candidate_patch = second_gray[cy1:cy2, cx1:cx2]

            # 크기가 다르면 비교 불가 (가장자리 잘림)
            if candidate_patch.shape != template.shape:
                continue

            score_map = cv2.matchTemplate(
                candidate_patch, template, cv2.TM_CCOEFF_NORMED
            )
            score = float(score_map[0][0])

            if score > best_score:
                best_score = score
                best_candidate_index = cand_idx
                best_candidate = candidate

        # score_threshold 이상인 경우만 매칭 확정
        if best_candidate is not None and best_score >= score_threshold:
            used_candidates.add(best_candidate_index)
            matches.append({
                "id": spot_index,
                "start_x": first_cx,
                "start_y": first_cy,
                "end_x": best_candidate["cx"],
                "end_y": best_candidate["cy"],
                "area": first_spot["area"],
                "score": round(best_score, 3)
            })

    return matches


@app.route('/analyze_series', methods=['POST'])
def analyze_series():
    files = request.files.getlist('images')
    if not files or len(files) < 2:
        return jsonify({
            'status': 'fail',
            'message': '2장 이상의 이미지를 업로드해야 합니다.'
        })

    try:
        # ✅ 사용자 설정 파라미터 수신
        # thresh_c: adaptiveThreshold C값 (낮을수록 민감, 기본 8)
        thresh_c      = int(float(request.form.get('threshC', 8)))
        # min_spot_size: 최소 흑점 면적 px² (기본 2, main_series.py 기본값)
        min_spot_size = float(request.form.get('minSpotSize', 2.0))
        # max_spot_size: 최대 흑점 면적 px² (기본 500, main_series.py 기본값)
        max_spot_size = float(request.form.get('maxSpotSize', 500.0))
        # score_threshold: 템플릿 매칭 최소 유사도 (기본 0.3)
        score_threshold = float(request.form.get('scoreThreshold', 0.3))

        files = sorted(files, key=lambda f: f.filename)

        # ==========================================
        # [처리 순서 1] 각 사진 순서대로 처리
        # ==========================================
        series_data = []
        for idx, file_obj in enumerate(files, start=1):
            file_bytes = np.frombuffer(file_obj.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                continue

            # [처리 순서 2~3] 태양 검출 + 정규화 (600×600, 중심(300,300), r=250)
            normalized, sun_x, sun_y, sun_r = normalize_sun_image(image)
            if normalized is None:
                continue  # 태양 미검출 → 스킵

            # [처리 순서 4~7] 흑점 검출 + 노이즈 제거
            spots = detect_spots_series(
                normalized, thresh_c, min_spot_size, max_spot_size
            )

            series_data.append({
                "index":      idx,
                "filename":   file_obj.filename,
                "normalized": normalized,
                "spots":      spots
            })

        if len(series_data) < 2:
            return jsonify({
                'status': 'fail',
                'message': f'태양이 검출된 이미지가 {len(series_data)}장입니다. 최소 2장이 필요합니다.'
            })

        # ==========================================
        # 템플릿 매칭: 첫 번째 → 두 번째 사진
        # ==========================================
        first_data  = series_data[0]
        second_data = series_data[1]

        matches = template_match_spots(
            first_data["normalized"],  first_data["spots"],
            second_data["normalized"], second_data["spots"],
            score_threshold
        )

        if not matches:
            return jsonify({
                'status': 'fail',
                'message': '흑점을 추적하지 못했습니다. 추적 민감도(C값)나 최소 흑점 크기를 조절해보세요.'
            })

        # ==========================================
        # 대표 흑점 선정
        # ==========================================
        center_x, center_y = TARGET_CENTER
        for m in matches:
            m["center_distance"] = round(math.hypot(
                m["start_x"] - center_x,
                m["start_y"] - center_y
            ), 2)

        rep_match = min(matches, key=lambda m: m["center_distance"])

        # ==========================================
        # 추정 적도 방향 & 자전축 계산
        # ==========================================
        best_dx = rep_match["end_x"] - rep_match["start_x"]
        best_dy = rep_match["end_y"] - rep_match["start_y"]
        equator_angle       = math.degrees(math.atan2(best_dy, best_dx))
        rotation_axis_angle = equator_angle + 90.0

        # ==========================================
        # 각 매칭 흑점 심화 계산
        # ==========================================
        output_data = []
        for m in matches:
            # 회전 후 시작 좌표
            rot_start_x, rot_start_y = rotate_point(
                m["start_x"], m["start_y"],
                center_x, center_y, equator_angle
            )
            # 회전 후 종료 좌표
            rot_end_x, rot_end_y = rotate_point(
                m["end_x"], m["end_y"],
                center_x, center_y, equator_angle
            )

            # 적도 방향 이동량 (px)
            equator_move = rot_end_x - rot_start_x

            mean_rot_y = (rot_start_y + rot_end_y) / 2.0
            latitude_ratio = max(-1.0, min(1.0, mean_rot_y / TARGET_RADIUS))
            latitude = math.degrees(math.asin(latitude_ratio))

            # 위도선 반지름 → 보정 이동각
            latitude_radius = TARGET_RADIUS * math.cos(math.radians(latitude))
            if abs(latitude_radius) < 1:
                rotation_angle = 0.0
            else:
                rotation_angle = math.degrees(equator_move / latitude_radius)

            # 원래 이동 방향각 (dx, dy 기반)
            dx = m["end_x"] - m["start_x"]
            dy = m["end_y"] - m["start_y"]
            move_angle = round(math.degrees(math.atan2(dy, dx)), 2)

            output_data.append({
                # 원좌표
                'id':        m['id'],
                'start_x':   m['start_x'],
                'start_y':   m['start_y'],
                'end_x':     m['end_x'],
                'end_y':     m['end_y'],
                'dx':        dx,
                'dy':        dy,
                # 매칭 정보
                'distance':        m['center_distance'],
                'score':           m['score'],
                # 회전 후 좌표
                'rot_start_x':     round(rot_start_x, 2),
                'rot_start_y':     round(rot_start_y, 2),
                'rot_end_x':       round(rot_end_x, 2),
                'rot_end_y':       round(rot_end_y, 2),
                # 심화 연산값
                'equator_move':    round(equator_move, 2),
                'latitude':        round(latitude, 2),
                'rotation_angle':  round(rotation_angle, 3),
                'move_angle':      move_angle,
            })

        # ==========================================
        # 시각화 이미지 생성 
        # ==========================================
        result_visual = second_data["normalized"].copy()

        # 흑점 이동 화살표
        for m in matches:
            cv2.arrowedLine(
                result_visual,
                (m["start_x"], m["start_y"]),
                (m["end_x"],   m["end_y"]),
                (0, 0, 255), 2, tipLength=0.3
            )
            cv2.circle(result_visual, (m["start_x"], m["start_y"]), 4, (255, 100, 0), -1)
            cv2.putText(
                result_visual,
                f"ID:{m['id']}",
                (m["end_x"] + 5, m["end_y"] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
            )

        # 태양 테두리
        cv2.circle(result_visual, TARGET_CENTER, TARGET_RADIUS, (255, 255, 255), 1)

        # 추정 적도선 (하늘색)
        eq_len = TARGET_RADIUS
        eq_dx = int(eq_len * math.cos(math.radians(equator_angle)))
        eq_dy = int(eq_len * math.sin(math.radians(equator_angle)))
        cv2.line(
            result_visual,
            (center_x - eq_dx, center_y - eq_dy),
            (center_x + eq_dx, center_y + eq_dy),
            (0, 217, 255), 2
        )

        # 추정 자전축 (주황색)
        ax_len = TARGET_RADIUS
        ax_angle = equator_angle + 90.0
        ax_dx = int(ax_len * math.cos(math.radians(ax_angle)))
        ax_dy = int(ax_len * math.sin(math.radians(ax_angle)))
        cv2.line(
            result_visual,
            (center_x - ax_dx, center_y - ax_dy),
            (center_x + ax_dx, center_y + ax_dy),
            (0, 100, 255), 2
        )

        # 대표 흑점 강조 (노란 원)
        cv2.circle(
            result_visual,
            (rep_match["start_x"], rep_match["start_y"]),
            8, (0, 255, 255), 2
        )

        _, buffer = cv2.imencode('.jpg', result_visual)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # ==========================================
        # 메타데이터 (사진 이름, 흑점 수, 개별 좌표)
        # ==========================================
        image_summaries = []
        for data in series_data:
            image_summaries.append({
                "index":      data["index"],
                "filename":   data["filename"],
                "spot_count": len(data["spots"]),
                "spots":      data["spots"]  
            })

        return jsonify({
            'status': 'success',
            'global_info': {
                'analyzed_count':              len(series_data),
                'representative_spot_id':      rep_match['id'],
                'representative_center_distance': rep_match['center_distance'],
                'equator_angle':               round(equator_angle, 2),
                'rotation_axis_angle':         round(rotation_axis_angle, 2),
            },
            'image_summaries': image_summaries,
            'data':            output_data,
            'image_base64':    img_base64
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'fail',
            'message': f'연속 분석 중 에러 발생: {str(e)}'
        })


if __name__ == '__main__':
    app.run(debug=True, port=5000)