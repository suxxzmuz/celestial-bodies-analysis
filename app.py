from fastapi import FastAPI
import numpy as np
from astropy.io import fits
import ctypes
import os
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore

app = FastAPI()

# --- 1. Firebase (Firestore) 초기화 ---
# 다운받은 JSON 키 파일이 app.py와 같은 폴더에 있어야 합니다.
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


# --- 2. C 라이브러리 설정 ---
dll_path = os.path.abspath("image_core.dll")
c_lib = ctypes.CDLL(dll_path)

c_lib.remove_noise.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int,
    ctypes.c_double
]


# --- 3. 더미 FITS 생성 ---
def create_dummy_fits():
    file_name = "dummy.fits"
    if not os.path.exists(file_name):
        data = np.random.rand(100, 100) * 255
        hdu = fits.PrimaryHDU(data)
        hdu.writeto(file_name)

create_dummy_fits()


# --- 4. 영상 분석 및 Firebase 저장 API ---
@app.get("/analyze")
def analyze_fits():
    file_name = "dummy.fits"
    
    with fits.open(file_name) as hdul:
        data = hdul[0].data.astype(np.float64)
        
    height, width = data.shape
    total_pixels = height * width
    original_mean = float(np.mean(data))
    
    flat_data = data.flatten()
    threshold = 120.0  # 임계값
    c_lib.remove_noise(flat_data, total_pixels, threshold)
    
    processed_data = flat_data.reshape(height, width)
    processed_mean = float(np.mean(processed_data))
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 분석 결과를 딕셔너리 형태로 묶기
    analysis_data = {
        "file_name": file_name,
        "width": width,
        "height": height,
        "original_mean": round(original_mean, 2),
        "processed_mean": round(processed_mean, 2),
        "threshold": threshold,
        "timestamp": current_time
    }
    
    # Firebase Firestore의 'analysis_history' 컬렉션에 새 문서(Document) 추가
    db.collection("analysis_history").add(analysis_data)
    
    return {
        "status": "success",
        "message": "분석 완료 및 Firebase 저장 성공",
        "data": analysis_data
    }


# --- 5. Firebase 과거 분석 이력 조회 API ---
@app.get("/history")
def get_analysis_history():
    """Firebase에 저장된 분석 기록을 시간 역순으로 가져옵니다."""
    # 'analysis_history' 컬렉션에서 timestamp 기준 내림차순(최신순) 정렬하여 가져오기
    docs = db.collection("analysis_history").order_by(
        "timestamp", direction=firestore.Query.DESCENDING
    ).stream()
    
    history_list = []
    for doc in docs:
        doc_data = doc.to_dict()
        doc_data["id"] = doc.id # Firebase가 자동 생성한 고유 문서 ID 추가
        history_list.append(doc_data)
    
    return {
        "status": "success",
        "total_count": len(history_list),
        "history": history_list
    }