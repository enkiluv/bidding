# -*- coding: utf-8 -*-

# 패키지 가져오기
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import ai_wonder as wonder

# 로더 함수
@st.cache_resource
def load_context(dataname):
    state = wonder.load_state(f"{dataname}_state.pkl")
    model = wonder.input_piped_model(state)
    return state, model

# 드라이버 함수
if __name__ == "__main__":
    # 스트림릿 인터페이스
    st.subheader(f"낙찰가 '투찰금액(원)' 예측기")
    st.markdown(":blue[**AI Wonder**] 제공")

    # 라디오버튼들을 가로로 배치하도록 설정
    st.write('<style> div.row-widget.stRadio > div { flex-direction: row; } </style>',
        unsafe_allow_html=True)

    # 사용자 입력
    공고종목 = st.radio("공고종목", ['비계구조[대]', '석면해체', '석면해체,비계구조[대]'], index=2)
    지역 = st.selectbox("지역", ['강원,충북,충남', '대전,충남', '전국', '전국,경남', '전국,대전', '전국,부산', '충남', '충북,충남,세종'], index=1)
    발주기관 = st.text_input("발주기관", value="국방부계룡대근무지원단")
    순공사원가원 = st.text_input("순공사원가(원)", value="38415523")
    기초금액원 = st.number_input("기초금액(원)", value=210456000)
    추정가격원 = st.number_input("추정가격(원)", value=191323636)
    낙찰하한율 = st.number_input("낙찰하한율(%)", value=87.745)
    예정가격원 = st.number_input("예정가격(원)", value=208787125)
    예가기초 = st.number_input("예가/기초(%)", value=99.207)
    낙찰하한가원 = st.number_input("낙찰하한가(원)", value=184739812)
    투찰기초 = st.number_input("투찰/기초(%)", value=87.78)
    A값원 = st.number_input("A값(원)", value=12562614.0)
    복수예비가1 = st.number_input("복수예비가_1", value=209159600)
    복수예비가2 = st.number_input("복수예비가_2", value=207130800)
    복수예비가3 = st.number_input("복수예비가_3", value=206505800)
    복수예비가4 = st.number_input("복수예비가_4", value=212352300)
    투찰율 = st.number_input("투찰율(%)", value=87.747)
    기초대비 = st.number_input("기초대비(%)", value=87.782)
    업체사정율 = st.number_input("업체사정율(%)", value=99.209)

    st.markdown("")

    # 입력값으로 데이터 만들기
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

    # 컨텍스트 로드
    state, model = load_context('낙찰가')

    # 예측 및 설명
    if st.button('예측'):
        st.markdown("")

        with st.spinner("추론 중..."):
            prediction = str(model.predict(point)[0])
            st.success(f"**{state.target}**의 예측값은 **{int(float(prediction))}** 입니다.")
            st.markdown("")

        with st.spinner("설명 생성 중..."):
            st.info("피처 중요도")
            importances = pd.DataFrame(wonder.local_explanations(state, point), columns=["피처", "값", "중요도"])
            st.dataframe(importances.round(3))

