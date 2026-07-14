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

        thresh_param  = int(request.form.get('threshValue',  8))
        min_spot_size = int(request.form.get('minSpotSize',  3))
        max_spot_size = int(request.form.get('maxSpotSize',  500)) 

        spots = cv2.adaptiveThreshold(
            masked, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31, thresh_param
        )
        spots = cv2.bitwise_and(spots, spots, mask=mask)

        # =========================================================
        # [수정 1] 노이즈 제거: 단일 영상 분석 시 모폴로지 연산 추가
        # =========================================================
        kernel = np.ones((3, 3), np.uint8)
        spots = cv2.morphologyEx(spots, cv2.MORPH_OPEN, kernel, iterations=1)
        # =========================================================

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
    theta = math.radians(angle_deg)
    rel_x = x - center_x
    rel_y = y - center_y
    rot_x = rel_x * math.cos(theta) + rel_y * math.sin(theta)
    rot_y = -rel_x * math.sin(theta) + rel_y * math.cos(theta)
    return rot_x, rot_y

@app.route('/analyze_series', methods=['POST'])
def analyze_series():
    files = request.files.getlist('images')
    if len(files) < 2:
        return jsonify({'status': 'fail', 'message': '최소 2장 이상의 이미지가 필요합니다.'})

    try:
        # 파일명 기준 정렬
        files = sorted(files, key=lambda f: f.filename)
        series_data = []

        # 1. 모든 이미지 정규화 및 흑점 검출
        for index, file in enumerate(files, start=1):
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                continue

            # 화면 크기 조절 및 흑백/가우시안 처리
            image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            # 태양 원판 검출
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=100, param2=30, minRadius=100, maxRadius=500)
            if circles is None:
                continue
            
            circles = np.round(circles[0]).astype(int)
            x, y, r = circles[0]

            # 태양 원판 중앙 정렬 및 크기 정규화
            target_x, target_y = TARGET_CENTER
            scale = TARGET_RADIUS / r
            resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
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

            normalized[dst_y1:dst_y2, dst_x1:dst_x2] = resized[src_y1:src_y2, src_x1:src_x2]

            # 흑점 검출
            norm_gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
            norm_gray = cv2.GaussianBlur(norm_gray, (5, 5), 0)
            mask = np.zeros(norm_gray.shape, dtype=np.uint8)
            cv2.circle(mask, TARGET_CENTER, TARGET_RADIUS, 255, -1)
            masked = cv2.bitwise_and(norm_gray, norm_gray, mask=mask)
            
            spots_mask = cv2.adaptiveThreshold(masked, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8)
            spots_mask = cv2.bitwise_and(spots_mask, spots_mask, mask=mask)

            # =========================================================
            # [수정 2] 노이즈 제거: 연속 영상 분석 시 모폴로지 연산 추가
            # =========================================================
            kernel_series = np.ones((3, 3), np.uint8)
            spots_mask = cv2.morphologyEx(spots_mask, cv2.MORPH_OPEN, kernel_series, iterations=1)
            # =========================================================

            contours, _ = cv2.findContours(spots_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            spot_list = []
            for contour in contours:
                area = cv2.contourArea(contour)
                # =========================================================
                # [수정 3] 연속 영상 분석 시 노이즈 면적 필터링 기준 강화 (2 -> 10으로 변경)
                # =========================================================
                if area < 10 or area > 500:
                    continue
                # =========================================================
                moments = cv2.moments(contour)
                if moments["m00"] == 0:
                    continue
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                spot_list.append({"x": cx, "y": cy, "area": float(area)})
            
            spot_list.sort(key=lambda spot: spot["x"])
            
            series_data.append({
                "index": index,
                "filename": file.filename,
                "normalized": normalized,
                "gray": norm_gray,
                "spots": spot_list
            })

        if len(series_data) < 2:
            return jsonify({'status': 'fail', 'message': '정상 분석 가능한(태양 검출 완료) 사진이 2장 미만입니다.'})

        # 2. 첫 번째 → 두 번째 사진 템플릿 매칭
        first_data = series_data[0]
        second_data = series_data[1]
        first_gray = first_data["gray"]
        second_gray = second_data["gray"]
        match_result_image = second_data["normalized"].copy()

        matches = []
        used_candidates = set()
        size = 15

        for spot_index, first_spot in enumerate(first_data["spots"], start=1):
            first_cx, first_cy = first_spot["x"], first_spot["y"]
            
            tx1 = max(0, first_cx - size)
            ty1 = max(0, first_cy - size)
            tx2 = min(first_gray.shape[1], first_cx + size)
            ty2 = min(first_gray.shape[0], first_cy + size)
            
            template = first_gray[ty1:ty2, tx1:tx2]
            
            best_score = -1
            best_candidate_index = None
            best_candidate = None

            for candidate_index, candidate in enumerate(second_data["spots"]):
                if candidate_index in used_candidates:
                    continue

                candidate_cx, candidate_cy = candidate["x"], candidate["y"]
                cx1 = max(0, candidate_cx - size)
                cy1 = max(0, candidate_cy - size)
                cx2 = min(second_gray.shape[1], candidate_cx + size)
                cy2 = min(second_gray.shape[0], candidate_cy + size)

                candidate_patch = second_gray[cy1:cy2, cx1:cx2]
                if candidate_patch.shape != template.shape:
                    continue

                score_map = cv2.matchTemplate(candidate_patch, template, cv2.TM_CCOEFF_NORMED)
                score = float(score_map[0][0])

                if score > best_score:
                    best_score = score
                    best_candidate_index = candidate_index
                    best_candidate = candidate

            # 유사도 0.7 이상인 매칭만 자동 확정 처리
            if best_candidate is not None and best_score >= 0.70:
                used_candidates.add(best_candidate_index)
                matches.append({
                    "id": spot_index,
                    "start_x": first_cx, "start_y": first_cy,
                    "x": best_candidate["x"], "y": best_candidate["y"],
                    "score": best_score
                })

        if not matches:
            return jsonify({'status': 'fail', 'message': '유효한 흑점 매칭 결과가 없습니다.'})

        # 3. 물리적 계산 (대표 흑점, 적도 방향, 자전축, 위도 등)
        center_x, center_y = TARGET_CENTER
        for match in matches:
            match["center_distance"] = math.hypot(match["start_x"] - center_x, match["start_y"] - center_y)

        # 태양 중심에 가장 가까운 대표 흑점 선정
        best_match = min(matches, key=lambda match: match["center_distance"])
        
        best_dx = best_match["x"] - best_match["start_x"]
        best_dy = best_match["y"] - best_match["start_y"]

        equator_angle = math.degrees(math.atan2(best_dy, best_dx))
        rotation_axis_angle = equator_angle + 90

        output_match_data = []
        for match in matches:
            # 회전 후 좌표
            start_rot_x, start_rot_y = rotate_point(match["start_x"], match["start_y"], center_x, center_y, equator_angle)
            end_rot_x, end_rot_y = rotate_point(match["x"], match["y"], center_x, center_y, equator_angle)

            equator_move = end_rot_x - start_rot_x
            mean_rot_y = (start_rot_y + end_rot_y) / 2
            
            # 태양 위도 근사 및 보정 이동각 계산
            latitude_ratio = max(-1.0, min(1.0, mean_rot_y / TARGET_RADIUS))
            latitude = math.degrees(math.asin(latitude_ratio))
            
            latitude_radius = TARGET_RADIUS * math.cos(math.radians(latitude))
            if abs(latitude_radius) < 1:
                rotation_angle = 0.0
            else:
                rotation_angle = math.degrees(equator_move / latitude_radius)

            output_match_data.append({
                "id": match["id"],
                "start_x": match["start_x"], "start_y": match["start_y"],
                "end_x": match["x"], "end_y": match["y"],
                "start_rot_x": round(start_rot_x, 2),
                "start_rot_y": round(start_rot_y, 2),
                "end_rot_x": round(end_rot_x, 2),
                "end_rot_y": round(end_rot_y, 2),
                "equator_move": round(equator_move, 2),
                "latitude": round(latitude, 2),
                "rotation_angle": round(rotation_angle, 3),
                "center_distance": round(match["center_distance"], 2),
                "score": round(match["score"], 3)
            })

            # 시각화 박스 및 텍스트 추가
            cv2.arrowedLine(match_result_image, (match["start_x"], match["start_y"]), (match["x"], match["y"]), (0, 0, 255), 2, tipLength=0.3)
            cv2.rectangle(match_result_image, (match["x"] - size, match["y"] - size), (match["x"] + size, match["y"] + size), (0, 255, 255), 2)
            cv2.putText(match_result_image, f"ID:{match['id']}", (match["x"] - size, match["y"] - size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 4. 사진별 메타데이터 작성
        image_summaries = []
        for data in series_data:
            image_summaries.append({
                "index": data["index"],
                "filename": data["filename"],
                "spot_count": len(data["spots"]),
                "spots": data["spots"]
            })

        cv2.circle(match_result_image, TARGET_CENTER, TARGET_RADIUS, (255, 255, 255), 1)
        _, buffer = cv2.imencode('.jpg', match_result_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'status': 'success',
            'global_info': {
                'analyzed_count': len(series_data),               
                'representative_spot_id': best_match["id"],       
                'equator_angle': round(equator_angle, 2),         
                'rotation_axis_angle': round(rotation_axis_angle, 2) 
            },
            'image_summaries': image_summaries,                   
            'match_data': output_match_data,                      
            'image_base64': img_base64
        })

    except Exception as e:
        return jsonify({'status': 'fail', 'message': f'에러 발생: {str(e)}'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)