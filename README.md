# ✏️ AI를 활용한 사용자 행동 기반 집중도 탐지 및 실제 학습 시간 자동 기록 앱 'Studify'

## ✨ 프로젝트 정보
### Studify(스터디파이)
>사용자의 공부 자세를 실시간으로 분석하는 AI 모델을 통해 <br />
>공부 여부, 집중 상태, 수면 여부를 자동으로 추정하고, <br />
>이를 바탕으로 **실제 공부 시간 기록** 및 **피드백을 제공**하는 학습 보조 애플리케이션입니다.

### Introduction
- 2025 ICT 한이음 드림업 공모전 출품작
- 덕성여자대학교 컴퓨터공학전공 졸업 프로젝트
<br />

https://github.com/user-attachments/assets/bd7d650b-9c57-4ec6-ac7f-7972262d6e04

<br />

## 📃 프로젝트 소개
### 개발 배경 및 필요성
🔥 대한민국의 높은 교육열
> 대한민국은 높은 수준의 교육열을 보이며, 학생들의 주간 평균 공부 시간은 OECD 평균보다 15시간 이상 많은 것으로 나타납니다. <br />
> 이러한 공부 환경을 고려하여, 사용자의 공부 여부와 집중도를 자동으로 분석하는 보조 도구를 통해 학생들의 자기관리 능력과 학습 효율성 상승에 도움을 주고자 합니다. <br />

💡 AI 기술의 교육적 활용 가능성 제시
> 코로나19 팬데믹을 계기로 온라인 강의가 활성화되면서, 학생의 웹캠 영상을 기반으로 집중도를 분석하는 AI 기술에 대한 연구가 활발히 이루어졌습니다. <br />
> 그러나 대면 수업이 다시 중심이 된 현재, 해당 기술은 실제 교육 현장에서 활용이 제한되는 한계가 있습니다. <br />
> 따라서 기존의 얼굴 인식 중심 분석에서 나아가 학생의 자세 및 손 모양을 기반으로 공부 상태를 판단할 수 있는 모델을 구현함으로써, AI 기술의 교육적 활용 가능성과 그 방향성을 제시하고자 합니다.
<br />

### 기획 의도
- 사용자의 공부 자세를 실시간으로 분석하고 공부 여부, 집중 상태, 수면 여부를 판단하는 작업을 통해 학습 과정에서의 **자기 성찰 기회** 제공
- 공부 데이터를 활용한 공부 시간, 집중 시간, 수면 시간 비율 통계를 통해 사용자의 자기주도 학습 지원 및 **올바른 공부 습관 체화** 도모

### 서비스명
- study + amplify의 합성어로, **사용자 학습 효율 증진**을 의미
<br />

## :bulb: Key Features
### 손 모양 기반 실제 공부 여부 추정
 > 사용자가 촬영한 공부 영상에서 손 모양의 좌표 추출 <br />
 > 추출한 좌표를 바탕으로 펜을 쥔 손/그렇지 않은 손 분류 모델을 통해 실제 필기 및 공부 여부 추정
### 공부 자세 기반 집중도 분석 및 수면 여부 탐지
 > 사용자가 촬영한 공부 영상에서 사용자의 몸(얼굴+상체)의 좌표 추출 <br />
 > 추출한 좌표를 바탕으로 공부 자세 분류 모델을 통해 바른 자세/ 기울어진 자세(앞, 옆, 뒤)/ 엎드린 자세/ 뒤로 넘어간 자세 분류 <br />
 > 분류한 공부 자세를 토대로 공부 집중 여부 및 수면 여부 추정 <br />
### 실제 공부 시간 기록
 > 손 모양 분석을 이용한 공부 여부 추정과 자세 추정 결과를 종합적으로 조합하여 녹화 시간 대비 실제 공부 시간 산출 <br />
### 통계 및 AI 피드백
 > 실제 공부 시간, 전체 학습 중 집중한 시간, 수면 시간의 비율을 통계 그래프로 시각화하여 제공. 공부 데이터 바탕 개인 맞춤형 AI 피드백 제공
<br />

## 모델 소개
### 펜 쥔손 분류 모델
>설명 <br />
> ..

### 자세 분류 모델
> 공부 중, 집중력 저하, 수면 상태의 6가지 자세를 학습 <br />
> 이미지에서 MediaPipe를 통해 상체 좌표를 추출하여 모델 입력값으로 활용

### 학습 자세
|라벨명|설명|예시 이미지|
|---|---|---|
|good_pose|척추 중립 자세로, 좌우 어깨 높이가 균형을 유지하는 자세|.|
|nfocus_lean_back|좌골이 의자의 앞 부근에 위치해 상체가 후방으로 기운 자세|.|
|nfocus_lean_forward|목을 숙여 책상이나 팔에 상체를 지지하는 자세|.|
|nfocus_lean_side|한쪽 팔이나 손으로 몸을 지지하여 목과 어깨가 한쪽으로 휜 비대칭 자세|.|
|sleep_head_back|상체가 후방으로 기운 상태에서 목의 관절각이 180도를 넘어 머리가 의자 뒤로 넘어간 자세|.|
|sleep_head_down|머리를 숙여 얼굴이 화면에 보이지 않아 공부 여부를 판별할 수 없는 자세|.|
### Feature 구성
|Feature|개수|설명|
|------|---|---|
|Pose 절대 좌표|36|12개 랜드마크 * 3축|
|face 절대 좌표|15|5개 랜드마크 * 3축|
|합계|51|17개 랜드마크 * 3축, 최종 입력 feature vector|
### 모델 구조
|제목|설명|
|------|---|
|입력층|상체 및 얼굴 주요 특징점 좌표 기준으로 구성된 51차원 피처 입력|
|은닉층1|128개 뉴런, ReLU 활성화 함수 적용, Batch Normalization 및 Dropout(0.3)|
|은닉층2|64개 뉴런, ReLU 활성화 함수 적용, Batch Normalization 및 Dropout(0.3)|
|은닉층3|32개 뉴런, ReLU 활성화 함수 적용, Batch Normalization 및 Dropout(0.3)|
|출력층|1개 뉴런(softmax 함수)로 다중 클래스 분류 수행|
### 통합 상태


 ## :computer: Tech Stack
- Client: <img src="https://img.shields.io/badge/Android-3DDC84?style=flat-square&logo=Android&logoColor=white"> <img src="https://img.shields.io/badge/Kotlin-7F52FF?style=flat-square&logo=Kotlin&logoColor=white"> <img src="https://img.shields.io/badge/mediapipe-0097A7.svg?style=flat-square&logo=mediapipe&logoColor=white">
- Server: <img src="https://img.shields.io/badge/SpringBoot-6DB33F?style=flat-square&logo=springboot&logoColor=white"> <img src="https://img.shields.io/badge/java-007396?style=flat-square&logo=OpenJDK&logoColor=white">
- AI: <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"> <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=TensorFlow&logoColor=white"> <img src="https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=Keras&logoColor=white">
- DB: <img src="https://img.shields.io/badge/MySQL-4479A1.svg?style=flat-square&logo=mysql&logoColor=white"> <img src="https://img.shields.io/badge/Redis-FF4438?style=flat-square&logo=redis&logoColor=white">
- Deploy: <img src="https://img.shields.io/badge/Amazon%20EC2-FF9900?style=flat-square&logo=Amazon%20EC2&logoColor=white"> <img src="https://img.shields.io/badge/docker-%230db7ed.svg?style=flat-square&logo=docker&logoColor=white"> <img src="https://img.shields.io/badge/Amazon RDS-527FFF?style=flat-square&logo=amazonrds&logoColor=white">
- Storage: <img src="https://img.shields.io/badge/Amazon%20S3-569A31?style=flat-square&logo=Amazon%20S3&logoColor=white">
<br />

## :hammer: System Architecture
<img src="https://github.com/user-attachments/assets/021a4418-26e7-4f69-961b-d67a4a218125" width=760 />
<br />
<br />
