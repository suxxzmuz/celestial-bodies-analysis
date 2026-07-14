import math
import cv2
import numpy as np
import os
from tkinter import Tk, filedialog

#이미지 여러 장 선택
root = Tk()
root.withdraw()
file_paths = filedialog.askopenfilenames(title="분석할 연속된 이미지를 선택해주세요.", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
if not file_paths:
    print("파일을 선택하지 않았습니다.")
    exit()

#파일명 기준으로 정렬
file_paths = sorted(file_paths)
print("=" * 42)
print("            ASTROVISION v1.0")
print("       SOLAR SERIES IMAGE ANALYSIS")
print("=" * 42)
print(f"\n선택된 파일 수: {len(file_paths)}장")

#각 이미지 분석
TARGET_SIZE = 600
TARGET_RADIUS = 250
TARGET_CENTER = (TARGET_SIZE // 2, TARGET_SIZE // 2)
def detect_spots(normalized):
   gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
   gray = cv2.GaussianBlur(gray, (5, 5), 0)
   mask = np.zeros(gray.shape, dtype=np.uint8)
   cv2.circle(mask, TARGET_CENTER, TARGET_RADIUS, 255, -1)
   masked = cv2.bitwise_and(gray, gray, mask=mask)
   spots_mask = cv2.adaptiveThreshold(masked, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8)
   spots_mask = cv2.bitwise_and(spots_mask, spots_mask, mask=mask)
   contours, _ = cv2.findContours(spots_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

   spot_list = []

   for contour in contours:
     area = cv2.contourArea(contour)
     if area < 2 or area > 500:
         continue
     moments = cv2.moments(contour)
     if moments["m00"] == 0:
         continue
     cx = int(moments["m10"] / moments["m00"])
     cy = int(moments["m01"] / moments["m00"])
     spot_list.append({"x": cx, "y": cy, "area": area})
     
   spot_list.sort(key=lambda spot: spot["x"])
   return spot_list, spots_mask

series_data = []
for index, file_path in enumerate(file_paths, start=1):
    #한글 경로 지원
    image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        print(f"\n[{index}] 이미지 불러오기 실패: {file_path}")
        continue

    #화면 크기 조절
    image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2)
    #흑백 변환+가우시안
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    #태양 원판 검출
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=100, param2=30, minRadius=100, maxRadius=500)
    filename = os.path.basename(file_path)
    if circles is None:
        print(f"\n[{index}] {filename}")
        print("태양을 찾지 못했습니다.")
        continue
    circles = np.round(circles[0]).astype(int)
    x, y, r = circles[0]
    #태양 원판 정규화
    target_x, target_y = TARGET_CENTER
    scale = TARGET_RADIUS/r
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

    normalized[dst_y1:dst_y2, dst_x1:dst_x2] = \
        resized[src_y1:src_y2, src_x1:src_x2]
    spot_list, spot_mask = detect_spots(normalized)
    series_data.append({"index":index, "file_path":file_path, "filename":filename, "image":image, "gray":gray, "sun_x":x, "sun_y":y, "sun_r":r, "normalized":normalized, "spots":spot_list, "spot_mask":spot_mask})
    
for data in series_data:
   result_image = data["normalized"].copy()
   for spot_index, spot in enumerate(data["spots"], start=1):
      cx = spot["x"]
      cy = spot["y"]
      #위치 표시
      cv2.circle(result_image, (cx, cy), 5, (0, 255, 0,), -1)
      #번호 표시
      cv2.putText(result_image, str(spot_index), (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

      if data["index"] == 1:
         size = 30
         x1 = max(0, cx - size)
         y1 = max(0, cy - size)

         x2 = min(result_image.shape[1], cx + size)
         y2 = min(result_image.shape[0], cy + size)

         template = data["normalized"][y1:y2, x1:x2]
         cv2.imwrite(f"output/template_{spot_index}.png", template)
      data["result_image"] = result_image
      cv2.imshow(f"Series Result {data['index']}", result_image)

#사진 템플릿 매칭
# ============================
# 첫 번째 → 두 번째 사진 후보 기반 템플릿 매칭
# ============================

if len(series_data) >= 2:
    first_data = series_data[0]
    second_data = series_data[1]

    first_gray = cv2.cvtColor(
        first_data["normalized"],
        cv2.COLOR_BGR2GRAY
    )

    second_gray = cv2.cvtColor(
        second_data["normalized"],
        cv2.COLOR_BGR2GRAY
    )

    match_result_image = second_data["normalized"].copy()

    matches = []

    # 같은 후보가 여러 흑점에 중복 배정되는 것 방지
    used_candidates = set()

    # 템플릿 반쪽 크기
    size = 15

    # 첫 번째 사진의 각 흑점 반복
    for spot_index, first_spot in enumerate(first_data["spots"], start=1):
     first_cx = first_spot["x"]
     first_cy = first_spot["y"]

     # 첫 번째 사진에서 템플릿 자르기
     tx1 = max(0, first_cx - size)
     ty1 = max(0, first_cy - size)
     tx2 = min(first_gray.shape[1], first_cx + size)
     ty2 = min(first_gray.shape[0], first_cy + size)

     template = first_gray[ty1:ty2, tx1:tx2]

     best_score = -1
     best_candidate_index = None
     best_candidate = None

     # 두 번째 사진에서 검출된 흑점 후보만 비교
     for candidate_index, candidate in enumerate(second_data["spots"]):
         if candidate_index in used_candidates:
             continue

         candidate_cx = candidate["x"]
         candidate_cy = candidate["y"]

         cx1 = max(0, candidate_cx - size)
         cy1 = max(0, candidate_cy - size)
         cx2 = min(second_gray.shape[1], candidate_cx + size)
         cy2 = min(second_gray.shape[0], candidate_cy + size)

         candidate_patch = second_gray[cy1:cy2, cx1:cx2]

         # 가장자리 때문에 두 조각의 크기가 다르면 비교하지 않음
         if candidate_patch.shape != template.shape:
             continue

         score_map = cv2.matchTemplate(candidate_patch, template, cv2.TM_CCOEFF_NORMED)

         score = float(score_map[0][0])

         if score > best_score:
             best_score = score
             best_candidate_index = candidate_index
             best_candidate = candidate

        # 적절한 후보를 찾은 경우
     if best_candidate is not None:
         used_candidates.add(best_candidate_index)

         matched_cx = best_candidate["x"]
         matched_cy = best_candidate["y"]

         matches.append({"id": spot_index, "start_x": first_cx, "start_y": first_cy, "x": matched_cx, "y": matched_cy, "score": best_score})

         # 후보 위치에 사각형 표시
         cv2.rectangle(match_result_image, (matched_cx - size, matched_cy - size), (matched_cx + size, matched_cy + size), (0, 255, 255), 2)

         cv2.putText(match_result_image, f"{spot_index}: {best_score:.2f}", (matched_cx - size, matched_cy - size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


    print("\n" + "=" * 54)
    print("1번 사진 → 2번 사진 매칭 후보")
    print("=" * 54)

    for match in matches:
     print(f"흑점 {match['id']} → "f"후보 좌표=({match['x']}, {match['y']}), "f"유사도={match['score']:.3f}")

    cv2.imshow("Template Matching Candidates", match_result_image)
    # 창이 실제로 화면에 표시되도록 잠깐 갱신
    cv2.waitKey(1)
    # 사용자 확인
    confirmed_matches = []

    print("\n" + "=" * 54)
    print("매칭 결과를 확인해 주세요.")
    print("후보 사진을 본 뒤 y 또는 n을 입력하세요.")
    print("=" * 54)

    for match in matches:
     print(f"\n흑점 {match['id']}"f" | 후보 좌표=({match['x']}, {match['y']})"f" | 유사도={match['score']:.3f}")

     while True:
      answer = input("같은 흑점이 맞나요? (y/n): ").strip().lower()

      if answer == "y":
         confirmed_matches.append(match)
         print("→ 매칭 확정")
         break

      elif answer == "n":
         print("→ 매칭 보류")
         break

      else:
         print("y 또는 n만 입력해 주세요.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #확정된 흑점 이동량 계산
    print("\n" + "=" * 54)
    print("확정된 흑점 이동 결과")
    print("=" * 54)

    #적도 방향 및 자전축 추정
    direction_angles = []
    for match in confirmed_matches:
       dx = match["x"] - match["start_x"]
       dy = match["y"] - match["start_y"]

       if math.hypot(dx, dy) < 1:
          continue
       
       angle = math.degrees(math.atan2(dy, dx))

       direction_angles.append(angle)

    #평균 방향 계산
    rotation_axis_angle = sum(direction_angles) / len(direction_angles)

    #자전축 방향
    equator_angle = rotation_axis_angle + 90

    print("\n" + "=" * 54)
    print("자전축 추정")
    print("=" * 54)

    print(f"적도 방향 : {equator_angle:.2f}°")
    print(f"자전축 방향 : {rotation_axis_angle:.2f}°")
    
    for match in confirmed_matches:
       dx = match["x"] - match["start_x"]
       dy = match["y"] - match["start_y"]

       distance = (dx ** 2 + dy ** 2) ** 0.5

       match["dx"] = dx
       match["dy"] = dy
       match["distance"] = distance

       print(f"\n흑점 {match['id']}" f"\n 1번 사진: ({match['start_x']}, {match['start_y']})" f"\n 2번 사진: ({match['x']}, {match['y']})" f"\n X 이동량: {dx:+d} px" f"\n Y 이동량: {dy:+d} px" f"\n 직선 이동 거리: {distance:.2f} px")
#결과 요약
print("\n" + "-"*54)
print(f"정상 분석된 사진: {len(series_data)}장")
print("-"*54)
if len(series_data) < 2:
 print("연속 분석에는 최소 2장의 사진이 필요합니다.")
 exit()
print("연속 분석 준비 완료")
print("\n" + "=" * 54)
print("사진별 흑점 검출 결과")
print("=" * 54)
for data in series_data:
   print(f"\n[{data['index']}] {data['filename']}")
   print(f"검출 흑점: {len(data['spots'])}개")
   for spot_index, spot in enumerate(data["spots"], start=1):
      print(f"      {spot_index}.   " f"좌표=({spot['x']}, {spot['y']}" f"면적={spot['area']:.2f} px^2")