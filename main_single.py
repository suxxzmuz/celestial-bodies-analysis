import csv
import os
import cv2
import numpy as np
from tkinter import Tk, filedialog

#파일 선택 창 열기
root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="분석할 태양 사진을 선택하세요", filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
if file_path == "":
    print("파일을 선택하지 않았습니다.")
    exit()
image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
if image is None:
    print("이미지를 불러오지 못했습니다.")
    exit()

#화면에 맞게 크기 조정
image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2)

#흑백 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#가우시안 블러
gray = cv2.GaussianBlur(gray, (5, 5), 0)

#태양 검출
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=100, param2=30, minRadius=100, maxRadius=500)
if circles is None:
    print("태양을 검출하지 못했습니다.")
    exit()
circles = np.round(circles[0]).astype(int)
x, y, r = circles[0]

#사진 상 태양 원판의 면적
sun_area = np.pi * r**2

#태양 마스크 생성
mask = np.zeros(gray.shape, dtype=np.uint8)
cv2.circle(mask, (x, y), r, 255, -1)
masked = cv2.bitwise_and(gray, gray, mask=mask)

#태양 원판 내부 픽셀만 추출
sun_pixels = masked[mask == 255]

#평균과 표준편차 계산
mean = np.mean(sun_pixels)
std = np.std(sun_pixels)

#흑점 후보 검출
spots = cv2.adaptiveThreshold(masked, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8)
spots = cv2.bitwise_and(spots, spots, mask=mask)

#흑점 개별 인식
contours, _ = cv2.findContours(spots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
results = image.copy()
cv2.circle(results, (x, y), r, (0, 255, 0), 3)
cv2.circle(results, (x, y), 3, (255, 0, 0), -1)
spot_count = 0
spots_info = []
spot_data = []
for contour in contours:
    area = cv2.contourArea(contour)
    #너무 작거나 큰 흑점은 제외
    if area < 2 or area > 500:
        continue
    M = cv2.moments(contour)
    if M["m00"] == 0:
        continue
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    #해당 흑점이 태양 원판에서 차지하는 비율
    area_ratio = area / sun_area * 100
    spots_info.append({"cx": cx, "cy": cy, "area": area, "ratio": area_ratio})

#x좌표 왼-오른 순으로 정렬
spots_info.sort(key=lambda s: s["cx"])

#번호 매기기
spot_data = []
for i, spot in enumerate(spots_info, start=1):
    cx = spot["cx"]
    cy = spot["cy"]
    area = spot["area"]
    ratio = spot["ratio"]
    cv2.circle(results, (cx, cy), 8, (255, 0, 0), 2)
    cv2.putText(results, str(i), (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    spot_data.append((i, cx, cy, area, ratio))
spot_count = len(spot_data)

#분석 결과 표 출력
original_name = os.path.splitext(os.path.basename(file_path))[0]
print("\n" + "=" * 42)
print("            ASTROVISION v1.0")
print("       SOLAR SINGLE IMAGE ANALYSIS")
print("=" * 42)
print(f"\n분석 파일: {original_name}")
print(f"태양 중심 좌표: ({x}, {y})")
print(f"반지름: {r} px")
print(f"검출 흑점: {spot_count}개")
print("\n" + "-" * 60)
print("번호 | 중심 좌표 (x, y) | 면적 (px^2) | 태양 대비(%)")
print("-" * 60)
for data in spot_data:
    number, cx, cy, area, ratio = data
    print(f"{number:<4} | ({cx:<6}, {cy:<6}) | {area:>11.2f} | {ratio:>.5f}")
print("-" * 60)

#결과 화면 출력
display_results = cv2.resize(results, (0, 0), fx=0.8, fy=0.8)
display_masked = cv2.resize(masked, (0, 0), fx=0.8, fy=0.8)
display_spots = cv2.resize(spots, (0, 0), fx=0.8, fy=0.8)

#결과 저장
os.makedirs("output", exist_ok=True)

#원본 파일 이름 가져오기
filename = os.path.splitext(os.path.basename(file_path))[0]

#저장 경로 만들기
save_path = f"output/{filename}_result.jpg"

#결과 이미지 저장
cv2.imwrite(save_path, results)
print(f"\n결과 이미지 저장 완료")
print(f"{save_path}")

#cvs 저장
csv_path = f"output/{filename}_result.csv"
with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    #제목
    writer.writerow(["번호", "중심X", "중심Y", "면적 (px^2)", "태양 대비(%)"])
    #데이터
    for data in spot_data:
        writer.writerow(data)
print(f"\nCSV 저장 완료")
print(f"{csv_path}")
cv2.imshow("ASTROVISION_Result", display_results)
cv2.imshow("Masked Sun Disk", display_masked)
cv2.imshow("Sunspot Candidates", display_spots)
cv2.waitKey(0)
cv2.destroyAllWindows()