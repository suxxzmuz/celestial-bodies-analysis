print("====== ASTROVISION ======")
print("1. 단일 이미지 분석")
print("2. 연속 이미지 분석")
choice = input("분석 모드를 선택하세요 : ")
if choice == "1":
    import main_single
elif choice == "2":
    import main_series
else:
    print("잘못된 입력입니다.")