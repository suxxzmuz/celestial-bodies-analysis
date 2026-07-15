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
# 3. 연속 영상 분석 엔진 (main_series.py 완벽 통합 + 노이즈 제거)
# ==========================================

def rotate_point(x, y, center_x, center_y, angle_deg):
    """지정된 각도만큼 좌표를 회전시켜 적도 기준 평면으로 정렬"""
    theta = math.radians(angle_deg)
    rel_x = x - center_x
    rel_y = y - center_y
    rot_x = rel_x * math.cos(theta) + rel_y * math.sin(theta)
    rot_y = -rel_x * math.sin(theta) + rel_y * math.cos(theta)
    return rot_x, rot_y

def process_and_extract_spots(file_obj, thresh_value):
    """이미지 정규화 및 [노이즈 제거]가 포함된 흑점 추출"""
    file_bytes = np.frombuffer(file_obj.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None: return None, None

    # [1] 태양 영역 정규화 (크기 및 중심 맞추기)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, 
                               param1=100, param2=30, minRadius=50, maxRadius=min(image.shape[:2])//2)
    
    normalized_img = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
    if circles is not None:
        # Numpy 에러 방지를 위한 명시적 int 변환
        x, y, r = int(circles[0][0][0]), int(circles[0][0][1]), int(circles[0][0][2])
        scale = TARGET_RADIUS / float(r)
        M = cv2.getRotationMatrix2D((float(x), float(y)), 0, scale)
        M[0, 2] += TARGET_CENTER[0] - x
        M[1, 2] += TARGET_CENTER[1] - y
        normalized_img = cv2.warpAffine(image, M, (TARGET_SIZE, TARGET_SIZE))
    else:
        normalized_img = cv2.resize(image, (TARGET_SIZE, TARGET_SIZE))

    # [2] 흑점 검출 및 강력한 노이즈 제거 
    norm_gray = cv2.cvtColor(normalized_img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(norm_gray.shape, dtype=np.uint8)
    cv2.circle(mask, TARGET_CENTER, TARGET_RADIUS, 255, -1)
    
    # 흑점 이진화
    _, thresh = cv2.threshold(norm_gray, thresh_value, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
    
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    spots = []
    spot_id = 1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= 5.0: # 5px 미만의 남은 노이즈 2차 차단
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                spots.append({
                    "id": spot_id,
                    "cx": cx, "cy": cy,
                    "area": round(area, 2)
                })
                spot_id += 1
                
    return normalized_img, spots


@app.route('/analyze_series', methods=['POST'])
def analyze_series():
    files = request.files.getlist('images')
    if not files or len(files) < 2:
        return jsonify({'status': 'fail', 'message': '2장 이상의 이미지를 업로드해야 합니다.'})

    try:
        thresh_val = int(request.form.get('threshValue', 80))
        files = sorted(files, key=lambda x: x.filename)
        
        series_data = []
        for idx, file_obj in enumerate(files):
            img, spots = process_and_extract_spots(file_obj, thresh_val)
            if img is not None:
                series_data.append({
                    "index": idx + 1,
                    "filename": file_obj.filename,
                    "img": img,
                    "spots": spots
                })
        
        if len(series_data) < 2:
            return jsonify({'status': 'fail', 'message': '분석 가능한 이미지가 2장 미만입니다.'})

        first_data = series_data[0]
        last_data = series_data[-1]
        img1, spots1 = first_data['img'], first_data['spots']
        img2, spots2 = last_data['img'], last_data['spots']

        # 1. 흑점 매칭
        matches = []
        match_id = 1
        for s1 in spots1:
            best_match = None
            min_dist = 200 
            for s2 in spots2:
                dist = math.hypot(s1['cx'] - s2['cx'], s1['cy'] - s2['cy'])
                if dist < min_dist:
                    min_dist = dist
                    best_match = s2
            
            if best_match:
                matches.append({
                    'id': match_id,
                    'start_x': s1['cx'], 'start_y': s1['cy'],
                    'end_x': best_match['cx'], 'end_y': best_match['cy'],
                    'area': s1['area'], 
                    'dx': best_match['cx'] - s1['cx'],
                    'dy': best_match['cy'] - s1['cy'],
                    'distance': round(min_dist, 2)
                })
                match_id += 1

        if not matches:
            return jsonify({'status': 'fail', 'message': '흑점을 추적하지 못했습니다. 민감도를 조절해보세요.'})

        # 2. 대표 흑점 선정
        rep_match = max(matches, key=lambda x: x['area'])
        
        # 3. 추정 적도 및 추정 자전축 계산
        equator_angle = math.degrees(math.atan2(rep_match['dy'], rep_match['dx']))
        rotation_axis_angle = equator_angle + 90.0

        output_data = []
        result_visual = img2.copy()
        
        # 4. 각 매칭 흑점 심화 수식 계산
        for m in matches:
            # 회전 후 좌표
            rot_start_x, rot_start_y = rotate_point(m['start_x'], m['start_y'], TARGET_CENTER[0], TARGET_CENTER[1], equator_angle)
            rot_end_x, rot_end_y = rotate_point(m['end_x'], m['end_y'], TARGET_CENTER[0], TARGET_CENTER[1], equator_angle)
            
            # 적도 방향 이동량
            equator_move = rot_end_x - rot_start_x
            
            # 추정 위도
            latitude_ratio = (TARGET_CENTER[1] - rot_start_y) / TARGET_RADIUS
            latitude_ratio = max(-1.0, min(1.0, latitude_ratio)) 
            latitude = math.degrees(math.asin(latitude_ratio))
            
            # 보정 이동각
            latitude_radius = TARGET_RADIUS * math.cos(math.radians(latitude))
            if abs(latitude_radius) < 1:
                rotation_angle = 0.0
            else:
                rotation_angle = math.degrees(equator_move / latitude_radius)
            
            output_data.append({
                'id': m['id'],
                'start_x': m['start_x'], 'start_y': m['start_y'],
                'end_x': m['end_x'], 'end_y': m['end_y'],
                'dx': m['dx'], 'dy': m['dy'],
                'distance': m['distance'],
                'rot_start_x': round(rot_start_x, 2), 
                'rot_start_y': round(rot_start_y, 2),
                'rot_end_x': round(rot_end_x, 2), 
                'rot_end_y': round(rot_end_y, 2),
                'equator_move': round(equator_move, 2),
                'latitude': round(latitude, 2),
                'rotation_angle': round(rotation_angle, 3)
            })

            cv2.arrowedLine(result_visual, (m['start_x'], m['start_y']), (m['end_x'], m['end_y']), (0, 0, 255), 2, tipLength=0.3)
            cv2.circle(result_visual, (m['start_x'], m['start_y']), 3, (255, 0, 0), -1)
            cv2.putText(result_visual, f"ID:{m['id']}", (m['end_x'] + 5, m['end_y'] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.circle(result_visual, TARGET_CENTER, TARGET_RADIUS, (255, 255, 255), 1)
        eq_len = 300
        eq_dx = int(eq_len * math.cos(math.radians(equator_angle)))
        eq_dy = int(eq_len * math.sin(math.radians(equator_angle)))
        cv2.line(result_visual, (TARGET_CENTER[0]-eq_dx, TARGET_CENTER[1]-eq_dy), 
                 (TARGET_CENTER[0]+eq_dx, TARGET_CENTER[1]+eq_dy), (0, 255, 255), 1)

        _, buffer = cv2.imencode('.jpg', result_visual)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # 5. 메타데이터 (정상 분석 장수, 사진 이름, 흑점 수, 개별 좌표 등)
        image_summaries = []
        for data in series_data:
            image_summaries.append({
                "index": data["index"],
                "filename": data["filename"],
                "spot_count": len(data["spots"]),
                "spots": data["spots"]
            })

        return jsonify({
            'status': 'success',
            'global_info': {
                'analyzed_count': len(series_data),
                'representative_spot_id': rep_match['id'],
                'equator_angle': round(equator_angle, 2),
                'rotation_axis_angle': round(rotation_axis_angle, 2)
            },
            'image_summaries': image_summaries, 
            'data': output_data, 
            'image_base64': img_base64
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'fail', 'message': f'연속 분석 중 에러 발생: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)