# -*- coding: utf-8 -*-

# 패키지 가져오기
import pandas as pd
import ai_wonder as wonder

# 사용자 입력 함수
def user_input(prompt, default):
    response = input(f"{prompt} (default: {default}): ")
    return response if response else default

# 드라이버 함수
if __name__ == "__main__":
    print(f"낙찰 예측 v2 '투찰금액(원)' 예측기")
    print("AI Wonder 제공\n")
    
    # 사용자 입력
    공고종목 = user_input("공고종목", "'석면해체'")
    지역 = user_input("지역", "'충남'")
    발주기관 = user_input("발주기관", "'서울특별시교육청 영락고등학교'")
    기초금액원 = int(user_input("기초금액(원)", 393787117))
    추정가격원 = float(user_input("추정가격(원)", 359052000.316))
    낙찰하한율 = float(user_input("낙찰하한율(%)", 87.141))
    예정가격원 = int(user_input("예정가격(원)", 394275348))
    예가기초 = float(user_input("예가_기초(%)", 100.103))
    낙찰하한가원 = int(user_input("낙찰하한가(원)", 346064922))
    투찰기초 = float(user_input("투찰_기초(%)", 88.056))
    A값원 = float(user_input("A값(원)", 26799063.241))
    순공사원가원 = float(user_input("순공사원가(원)", 319086294.332))
    복수예비가1 = int(user_input("복수예비가1", 395354661))
    복수예비가2 = int(user_input("복수예비가2", 392033214))
    복수예비가3 = int(user_input("복수예비가3", 392687670))
    복수예비가4 = int(user_input("복수예비가4", 397017960))
    투찰율 = float(user_input("투찰율(%)", 87.179))
    기초대비 = float(user_input("기초대비(%)", 88.093))
    업체사정율 = float(user_input("업체사정율(%)", 100.143))

    # 입력값으로 데이터포인트 만들기
    point = pd.DataFrame([{
        '공고종목': 공고종목,
        '지역': 지역,
        '발주기관': 발주기관,
        '기초금액(원)': 기초금액원,
        '추정가격(원)': 추정가격원,
        '낙찰하한율(%)': 낙찰하한율,
        '예정가격(원)': 예정가격원,
        '예가_기초(%)': 예가기초,
        '낙찰하한가(원)': 낙찰하한가원,
        '투찰_기초(%)': 투찰기초,
        'A값(원)': A값원,
        '순공사원가(원)': 순공사원가원,
        '복수예비가1': 복수예비가1,
        '복수예비가2': 복수예비가2,
        '복수예비가3': 복수예비가3,
        '복수예비가4': 복수예비가4,
        '투찰율(%)': 투찰율,
        '기초대비(%)': 기초대비,
        '업체사정율(%)': 업체사정율,
    }])

    # 예측
    model = wonder.load_model('낙찰 예측 v2_model.pkl')
    prediction = str(model.predict(point)[0])
    print(f"\n'투찰금액(원)'의 예측값은 {int(float(prediction))}입니다.")
###
