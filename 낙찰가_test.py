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
    print(f"낙찰가 '투찰금액(원)' 예측기")
    print("AI Wonder 제공\n")
    
    # 사용자 입력
    공고종목 = user_input("공고종목", "'석면해체'")
    지역 = user_input("지역", "'전국,대전'")
    발주기관 = user_input("발주기관", "'특수전사령부'")
    순공사원가원 = user_input("순공사원가(원)", "'429738643'")
    기초금액원 = int(user_input("기초금액(원)", 210456000))
    추정가격원 = int(user_input("추정가격(원)", 191323636))
    낙찰하한율 = float(user_input("낙찰하한율(%)", 87.745))
    예정가격원 = int(user_input("예정가격(원)", 208787125))
    예가기초 = float(user_input("예가/기초(%)", 99.207))
    낙찰하한가원 = int(user_input("낙찰하한가(원)", 184739812))
    투찰기초 = float(user_input("투찰/기초(%)", 87.78))
    A값원 = float(user_input("A값(원)", 12562614.0))
    복수예비가1 = int(user_input("복수예비가_1", 209159600))
    복수예비가2 = int(user_input("복수예비가_2", 207130800))
    복수예비가3 = int(user_input("복수예비가_3", 206505800))
    복수예비가4 = int(user_input("복수예비가_4", 212352300))
    투찰율 = float(user_input("투찰율(%)", 87.747))
    기초대비 = float(user_input("기초대비(%)", 87.782))
    업체사정율 = float(user_input("업체사정율(%)", 99.209))

    # 입력값으로 데이터포인트 만들기
    point = pd.DataFrame([{
        '공고종목': 공고종목,
        '지역': 지역,
        '발주기관': 발주기관,
        '순공사원가(원)': 순공사원가원,
        '기초금액(원)': 기초금액원,
        '추정가격(원)': 추정가격원,
        '낙찰하한율(%)': 낙찰하한율,
        '예정가격(원)': 예정가격원,
        '예가/기초(%)': 예가기초,
        '낙찰하한가(원)': 낙찰하한가원,
        '투찰/기초(%)': 투찰기초,
        'A값(원)': A값원,
        '복수예비가_1': 복수예비가1,
        '복수예비가_2': 복수예비가2,
        '복수예비가_3': 복수예비가3,
        '복수예비가_4': 복수예비가4,
        '투찰율(%)': 투찰율,
        '기초대비(%)': 기초대비,
        '업체사정율(%)': 업체사정율,
    }])

    # 예측
    model = wonder.load_model('낙찰가_model.pkl')
    prediction = str(model.predict(point)[0])
    print(f"\n'투찰금액(원)'의 예측값은 {int(float(prediction))}입니다.")
###
