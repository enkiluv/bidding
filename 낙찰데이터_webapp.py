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
    st.subheader(f"낙찰데이터 '낙찰금액(원)' 예측기")
    st.markdown(":blue[**AI Wonder**] 제공")

    # 라디오버튼들을 가로로 배치하도록 설정
    st.write('<style> div.row-widget.stRadio > div { flex-direction: row; } </style>',
        unsafe_allow_html=True)

    # 사용자 입력
    공고종목 = st.radio("공고종목", ['석면해체', '석면해체,비계구조[대]'], index=0)
    지역 = st.radio("지역", ['충남'], index=0)
    발주기관 = st.selectbox("발주기관", ['국방출판지원단', '충청남도천안교육지원청', '충청남도서산교육지원청', '충청남도논산계룡교육지원청', '충청남도 서산시', '국방부계룡대근무지원단', '공주대학교', '국군재정관리단', '충청남도교육청 태안여자고등 \n학교', '충청남도예산교육지원청', '육군훈련소', '충청남도교육청 보령정심학교'], index=6)
    기초금액원 = st.number_input("기초금액(원)", value=156816000)
    추정가격원 = st.number_input("추정가격(원)", value=142560000.0)
    낙찰하한율 = st.number_input("낙찰하한율(%)", value=87.745)
    예정가격원 = st.number_input("예정가격(원)", value=158725300)
    예가기초 = st.number_input("예가_기초(%)", value=101.218)
    낙찰하한가원 = st.number_input("낙찰하한가(원)", value=139273515)
    투찰기초 = st.number_input("투찰_기초(%)", value=88.813)
    A값원 = st.number_input("A값(원)", value=17529985.0)
    순공사원가원 = st.number_input("순공사원가(원)", value=0.0)
    투찰율 = st.number_input("투찰율(%)", value=87.751)
    기초대비 = st.number_input("기초대비(%)", value=88.82)
    업체사정율 = st.number_input("업체사정율(%)", value=101.225)

    st.markdown("")

    # 입력값으로 데이터 만들기
    point = pd.DataFrame([{
        '공고종목': 공고종목,
        '지역': 지역,
        '발주기관': 발주기관,
        '기초금액(원)': int(기초금액원),
        '추정가격(원)': int(추정가격원),
        '낙찰하한율(%)': 낙찰하한율,
        '예정가격(원)': int(예정가격원),
        '예가_기초(%)': 예가기초,
        '낙찰하한가(원)': int(낙찰하한가원),
        '투찰_기초(%)': 투찰기초,
        'A값(원)': A값원,
        '순공사원가(원)': int(순공사원가원),
        '투찰율(%)': 투찰율,
        '기초대비(%)': 기초대비,
        '업체사정율(%)': 업체사정율,
    }])

    # 컨텍스트 로드
    state, model = load_context('낙찰데이터')

    # 예측 및 설명
    if st.button('예측'):
        st.markdown("")

        with st.spinner("추론 중..."):
            prediction = str(model.predict(point)[0])
            st.success(f"**{state.target}**의 예측값은 **{format(int(float(prediction)), ',')}** 입니다.")
            st.markdown("")

        with st.spinner("설명 생성 중..."):
            st.info("피처 중요도")
            importances = pd.DataFrame(wonder.local_explanations(state, point), columns=["피처", "값", "중요도"])
            st.dataframe(importances.round(3))

