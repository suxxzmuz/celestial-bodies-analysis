import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ==========================================
# 1. 화면 라우팅 (페이지 이동 설정)
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sun-single-image')
def single_page():
    return render_template('sun-single-image.html')

@app.route('/sun-continuous-image')
def series_page():
    return render_template('sun-continuous-image.html')


# ==========================================
# 2. 단일 영상 분석 백엔드 로직 (기존 main_single.py 내용)
# ==========================================
@app.route('/analyze_single', methods=['POST'])
def analyze_single():
    if 'image' not in request.files:
        return jsonify({'status': 'fail', 'message': '이미지 파일이 없습니다.'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'status': 'fail', 'message': '선택된 파일이 없습니다.'})

    # 브라우저가 업로드한 파일 읽기
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if image is None:
        return jsonify({'status': 'fail', 'message': '이미지를 불러올 수 없습니다.'})

    # 1) 이미지 크기 표준화 (웹 호환용)
    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2) 태양 검출 (Hough Circles)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=100, param2=30, minRadius=50, maxRadius=400)
    
    if circles is None:
        return jsonify({'status': 'fail', 'message': '태양(원형 영역)을 검출하지 못했습니다.'})

    circles = np.round(circles[0, :]).astype("int")
    x, y, r = circles[0]

    # 3) 관심 영역(ROI) 마스킹 및 흑점 이진화
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)
    masked = cv2.bitwise_and(gray, gray, mask=mask)
    
    # 적응형 이진화로 흑점 추출
    spots = cv2.adaptiveThreshold(masked, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    spots = cv2.bitwise_and(spots, spots, mask=mask)

    # 4) 컨투어 검출 및 데이터 연산
    contours, _ = cv2.findContours(spots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    results_img = image.copy()
    cv2.circle(results_img, (x, y), r, (0, 255, 0), 2) # 태양 외곽선 그리기

    spot_data = []
    spot_count = 0
    sun_area = np.pi * (r ** 2)

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < 5: # 너무 작은 노이즈 제거
            continue
        
        spot_count += 1
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x, y

        ratio = (area / sun_area) * 100
        spot_data.append({
            'number': spot_count,
            'cx': cx,
            'cy': cy,
            'area': round(area, 2),
            'ratio': round(ratio, 5)
        })

        # 결과 이미지에 번호와 외곽선 마킹
        cv2.drawContours(results_img, [contour], -1, (0, 0, 255), 2)
        cv2.putText(results_img, str(spot_count), (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 5) 이미지를 웹 브라우저로 전송하기 위해 Base64로 인코딩
    _, buffer = cv2.imencode('.jpg', results_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'status': 'success',
        'image_base64': img_base64,
        'sun_coord': f"({x}, {y})",
        'sun_radius': f"{r} px",
        'spot_count': spot_count,
        'data': spot_data
    })


# ==========================================
# 3. 연속 영상 분석 백엔드 로직 (기존 main_series.py 내용)
# ==========================================
@app.route('/analyze_series', methods=['POST'])
def analyze_series():
    files = request.files.getlist('images')
    if len(files) < 2:
        return jsonify({'status': 'fail', 'message': '최소 2장 이상의 이미지가 필요합니다.'})

    images = []
    for file in files:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is not None:
            # 해상도를 동일하게 통일 (600x600 규격화)
            img_resized = cv2.resize(img, (600, 600))
            images.append(img_resized)

    if len(images) < 2:
        return jsonify({'status': 'fail', 'message': '이미지를 정상적으로 파싱하지 못했습니다.'})

    # 첫 번째 이미지와 두 번째 이미지 비교 추적
    img1, img2 = images[0], images[1]
    
    # 흑점 검출 함수 정의 (기존 알고리즘 이식)
    def get_spots_positions(img_src):
        gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (300, 300), 250, 255, -1) # 중앙 기준 마스킹
        masked = cv2.bitwise_and(gray, gray, mask=mask)
        spots_mask = cv2.adaptiveThreshold(masked, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        spots_mask = cv2.bitwise_and(spots_mask, spots_mask, mask=mask)
        contours, _ = cv2.findContours(spots_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        positions = []
        for c in contours:
            if cv2.contourArea(c) > 5:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    positions.append((cx, cy))
        return positions

    pos1 = get_spots_positions(img1)
    
    # 결과 출력용 베이스 이미지 (2번 사진 위에 화살표 표기)
    result_img = img2.copy()
    track_data = []
    track_id = 1

    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 템플릿 매칭 기반 추적 알고리즘 구현
    for pt1 in pos1:
        sx, sy = pt1
        # 경계선 예외 처리
        if sx - 20 < 0 or sx + 20 > 600 or sy - 20 < 0 or sy + 20 > 600:
            continue
            
        # 1번 이미지에서 흑점 템플릿 잘라내기
        template = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)[sy-20:sy+20, sx-20:sx+20]
        
        # 2번 이미지의 주변 탐색 영역 정의
        search_area = gray2[max(0, sy-60):min(600, sy+60), max(0, sx-60):min(600, sx+60)]
        if search_area.shape[0] < template.shape[0] or search_area.shape[1] < template.shape[1]:
            continue

        res = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > 0.6: # 유사도 임계값
            # 탐색 영역 기준 실제 2번 이미지 좌표 계산
            ex = max(0, sx - 60) + max_loc[0] + 20
            ey = max(0, sy - 60) + max_loc[1] + 20

            dx = ex - sx
            dy = ey - sy
            distance = round((dx**2 + dy**2)**0.5, 2)

            track_data.append({
                'id': track_id,
                'start_x': sx, 'start_y': sy,
                'end_x': ex, 'end_y': ey,
                'dx': dx, 'dy': dy,
                'distance': distance
            })

            # 이동 궤적 시각화 (화살표 및 텍스트 마킹)
            cv2.arrowedLine(result_img, (sx, sy), (ex, ey), (0, 255, 0), 2, tip_length=0.3)
            cv2.circle(result_img, (sx, sy), 3, (0, 0, 255), -1)
            cv2.putText(result_img, f"ID {track_id}", (ex + 5, ey - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            track_id += 1

    # 웹 전송용 인코딩
    _, buffer = cv2.imencode('.jpg', result_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'status': 'success',
        'image_base64': img_base64,
        'data': track_data
    })


if __name__ == '__main__':
    # 로컬 테스트용 (Render 서버 구동 시에는 gunicorn이 실행하므로 무시됨)
    app.run(debug=True)