import os
import cv2
import numpy as np
import base64
import sqlite3
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
    # 데이터베이스 파일(astrovision.db) 연결 및 테이블 생성
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

# 서버가 켜질 때 DB 테이블이 없으면 자동으로 만듭니다.
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

@app.route('/log-in')
def login_page():
    return render_template('log-in.html')

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
#커뮤니티 게시판 DB 통신 API
# ==========================================
# (1) DB에 저장된 모든 분석 글 가져오기
@app.route('/api/posts', methods=['GET'])
def get_posts():
    conn = get_db_connection()
    # 최신 글이 먼저 나오도록 역순(DESC)으로 가져옵니다.
    posts = conn.execute('SELECT * FROM community ORDER BY id DESC').fetchall()
    conn.close()
    
    # 파이썬 DB 데이터를 웹이 이해할 수 있는 JSON(리스트) 형태로 변환
    post_list = [dict(post) for post in posts]
    return jsonify({'status': 'success', 'data': post_list})

# (2) 웹에서 업로드한 새 분석 글을 DB에 저장하기
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

        image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, 
                                   param1=100, param2=30, minRadius=100, maxRadius=500)
        
        if circles is None:
            return jsonify({'status': 'fail', 'message': '이미지에서 태양을 검출하지 못했습니다.'})

        circles = np.round(circles[0, :]).astype("int")
        x, y, r = circles[0]
        sun_area = np.pi * (r ** 2)

        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        masked = cv2.bitwise_and(gray_blur, gray_blur, mask=mask)
        
        sun_only = cv2.bitwise_not(masked)
        sun_only = cv2.bitwise_and(sun_only, sun_only, mask=mask)
        spots = cv2.adaptiveThreshold(sun_only, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        spots = cv2.bitwise_and(spots, spots, mask=mask)

        contours, _ = cv2.findContours(spots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results = image.copy()
        
        spot_data = []
        spot_count = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 5:
                continue
                
            spot_count += 1
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            ratio = (area / sun_area) * 100
            spot_data.append({
                'id': spot_count,
                'cx': cx, 'cy': cy,
                'area': round(area, 2),
                'ratio': round(ratio, 5)
            })

            cv2.circle(results, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(results, str(spot_count), (cx + 5, cy - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        cv2.circle(results, (x, y), r, (0, 255, 0), 2)
        _, buffer = cv2.imencode('.jpg', results)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'status': 'success',
            'spot_count': spot_count,
            'data': spot_data,
            'image_base64': img_base64
        })

    except Exception as e:
        return jsonify({'status': 'fail', 'message': f'분석 중 에러 발생: {str(e)}'})

# ==========================================
# 3. 연속 영상 분석 엔진
# ==========================================
def detect_spots_for_series(normalized_img):
    gray = cv2.cvtColor(normalized_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, TARGET_CENTER, TARGET_RADIUS, 255, -1)
    masked = cv2.bitwise_and(gray, gray, mask=mask)
    
    sun_only = cv2.bitwise_not(masked)
    sun_only = cv2.bitwise_and(sun_only, sun_only, mask=mask)
    spots_mask = cv2.adaptiveThreshold(sun_only, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
    spots_mask = cv2.bitwise_and(spots_mask, spots_mask, mask=mask)
    
    contours, _ = cv2.findContours(spots_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    spots_list = []
    
    for contour in contours:
        if cv2.contourArea(contour) >= 5:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                spots_list.append((cx, cy))
    return spots_list

@app.route('/analyze_series', methods=['POST'])
def analyze_series():
    files = request.files.getlist('images')
    if len(files) < 2:
        return jsonify({'status': 'fail', 'message': '최소 2장 이상의 이미지가 필요합니다.'})

    try:
        files = sorted(files, key=lambda f: f.filename)
        img_bytes1 = np.frombuffer(files[0].read(), np.uint8)
        img_bytes2 = np.frombuffer(files[1].read(), np.uint8)
        
        img1 = cv2.imdecode(img_bytes1, cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(img_bytes2, cv2.IMREAD_COLOR)
        
        if img1 is None or img2 is None:
            return jsonify({'status': 'fail', 'message': '이미지를 읽을 수 없습니다.'})

        img1_norm = cv2.resize(img1, (TARGET_SIZE, TARGET_SIZE))
        img2_norm = cv2.resize(img2, (TARGET_SIZE, TARGET_SIZE))

        spots1 = detect_spots_for_series(img1_norm)
        gray2 = cv2.cvtColor(img2_norm, cv2.COLOR_BGR2GRAY)
        result_visual = img2_norm.copy()
        
        output_data = []
        match_id = 1
        
        for pt1 in spots1:
            x1, y1 = pt1
            w, h = 30, 30
            if x1 - w//2 < 0 or x1 + w//2 >= TARGET_SIZE or y1 - h//2 < 0 or y1 + h//2 >= TARGET_SIZE:
                continue
                
            template = gray2[y1 - h//2 : y1 + h//2, x1 - w//2 : x1 + w//2]
            
            search_r = 80
            sx1 = max(0, x1 - search_r)
            sy1 = max(0, y1 - search_r)
            sx2 = min(TARGET_SIZE, x1 + search_r)
            sy2 = min(TARGET_SIZE, y1 + search_r)
            
            search_area = gray2[sy1:sy2, sx1:sx2]
            if search_area.shape[0] < template.shape[0] or search_area.shape[1] < template.shape[1]:
                continue

            res = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            if max_val >= 0.75:
                best_x = sx1 + max_loc[0] + w//2
                best_y = sy1 + max_loc[1] + h//2
                
                dx = best_x - x1
                dy = best_y - y1
                distance = (dx**2 + dy**2) ** 0.5
                
                output_data.append({
                    'id': match_id,
                    'start_x': x1, 'start_y': y1,
                    'end_x': best_x, 'end_y': best_y,
                    'dx': int(dx), 'dy': int(dy),
                    'distance': round(distance, 2)
                })

                cv2.arrowedLine(result_visual, (x1, y1), (best_x, best_y), (0, 0, 255), 2, tipLength=0.3)
                cv2.circle(result_visual, (x1, y1), 3, (255, 0, 0), -1)
                cv2.putText(result_visual, f"ID:{match_id}", (best_x + 5, best_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                match_id += 1

        cv2.circle(result_visual, TARGET_CENTER, TARGET_RADIUS, (255, 255, 255), 1)
        _, buffer = cv2.imencode('.jpg', result_visual)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'status': 'success',
            'data': output_data,
            'image_base64': img_base64
        })

    except Exception as e:
        return jsonify({'status': 'fail', 'message': f'에러 발생: {str(e)}'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)