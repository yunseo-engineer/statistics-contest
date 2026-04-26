# 광역버스 기반 수도권-서울 업무지구 통근 출발지 불균형 진단
 
> 머신러닝 기반 통근 부담 분석을 통해 지역별 교통 정책 우선순위를 도출하는 데이터 분석 프로젝트
 
## Project Overview
 
본 연구는 **광역버스 통행 데이터**를 기반으로 서울 주요 업무지구(강남구 역삼1동)로 향하는 수도권 18개 지역의 통근 부담 불균형을 정량적으로 분석합니다.
 
**핵심 문제**: OECD 통계상 한국의 평균 통근시간이 58분으로 최장인 가운데, 지역별 통근 부담의 편차를 체계적으로 측정하고 정책 우선순위를 도출할 필요성이 제기됨.
 
**주요 성과**:
- 6개 파라미터 기반 **통근부담지수(Commute Burden Index)** 설계
- XAI SHAP 분석을 통한 파라미터별 중요도 파악
- 정책 우선도 1위 지역(수지구) 대상 유전알고리즘 기반 급행 노선 설계
- **통근부담지수 57.1% 감소** (0.524 → 0.225)
---
 
## Tech Stack
 
| 분류 | 기술 |
|------|------|
| **Data Processing** | Python, Pandas, NumPy |
| **Geospatial Analysis** | Haversine Formula, Tmap API |
| **Machine Learning** | Scikit-learn (K-Means, PCA, MinMax Scaling) |
| **Interpretability** | SHAP (SHapley Additive exPlanations) |
| **Optimization** | Genetic Algorithm |
| **Visualization** | Matplotlib, Seaborn |
| **Data Sources** | 서울통계통합플랫폼, SGIS, 교통카드 빅데이터, Tmap API |
 
---
 
## Analysis Pipeline
 
### 1️⃣ **데이터 적재 및 임베딩 최적화**
 
교통카드 빅데이터, SGIS 기업생태분석지도, Tmap API 정규경로 데이터를 ETL 파이프라인으로 통합. **시공간 정규화(spatiotemporal normalization)**를 통해 06시-09시 첨두시 시간대별 재차인원 분포의 스케일 왜곡 제거.
 
- 서울시 상주-주간인구 (2020), 기업생태분석지도 (2023)
- 교통카드 빅데이터 시스템 (2025년 4월)
- Tmap 대중교통 API를 통한 실제 소요시간 및 경로 데이터
- 도착지: 강남구 역삼1동 | 출발지: 유입 인구 상위 18개 지역의 시청/구청
### 2️⃣ **다중 특성 엔지니어링 및 가중화 설계**
 
**혼잡도(Congestion Index)**를 승객 수용성 포화도 기반으로 재정의: 초승 혼잡도에 40% 가중, 환승 혼잡도에 60% 가중 (심리 부담감 대칭성 반영).
 
**Haversine 좌표계**로 지구 표면 거리를 계산하여 직선거리-실제경로 효율성 비율(T/D ratio) 도출.
 
**정류장 통과 속도 복잡도(Cpx)** = Σ(정류장 수 / 운행시간)로 정차간격의 정성적 부담을 정량화.
 
```python
# 혼잡도 계산
C = (0.4 * congestion_first) + (0.6 * congestion_transfer)
 
# 정류장 복잡도
Cpx = (stop_count_first / running_time_first) + (stop_count_transfer / running_time_transfer)
 
# 거리 대비 시간 효율성
T/D = straight_distance (Haversine) / travel_time
```
 
### 3️⃣ **가중합 통근부담지수(Weighted CBI) 산출**
 
**Min-Max 정규화**로 6개 파라미터(혼잡도, 노선복잡도, 환승횟수, 도보시간, 거리대비시간, 통근비용)의 단위 표준화. 
 
**AHP(Analytic Hierarchy Process)** 기반 다기준 의사결정으로 파라미터별 가중치를 산정. 교통공학 문헌상 환승 스트레스 회귀계수 비중을 반영한 가중합 가중치 도출.
 
```python
# Min-Max 정규화
X_normalized = (X - X_min) / (X_max - X_min)
 
# 가중합 지수
CBI = (0.24×C) + (0.24×Cpx) + (0.24×Tr) + (0.15×W) + (0.08×T/D) + (0.05×M)
```
 
### 4️⃣ **차원축소 및 비지도 클러스터링**
 
**PCA(Principal Component Analysis)**로 6차원 특성 공간을 분산 설명력 85% 이상 유지하며 2-3차원 저차원 공간으로 축약.
 
**k-Means 클러스터링(k=3)**으로 지역 동질성 분류. **Elbow Method**와 **실루엣 계수(Silhouette Coefficient)**로 최적 군집 수 검증. **WCSS(Within-Cluster Sum of Squares)** 감소율 변곡점 분석.
 
```python
# PCA 적용 (분산 설명력 85% 유지)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_normalized)
 
# K-Means 클러스터링
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_pca)
 
# 실루엣 계수 검증
silhouette_score(X_pca, clusters)
```
 
### 5️⃣ **SHAP 기반 모델 해석성 분석**
 
**SHAP(SHapley Additive exPlanations)** 가치 분해를 통한 개별 지역별 예측치 기여도 분석.
 
클러스터별 피처 중요도 순위 도출로 **"왜 이 지역이 높은 부담을 보이는가"**에 대한 인과적 해석 제공.
 
**Force Plot, Summary Plot**으로 전역-국소 설명성(global-local explainability) 확보.
 
```python
# SHAP Explainer
explainer = shap.KMeansExplainer(kmeans, X_pca)
shap_values = explainer.shap_values(X_pca)
 
# 피처 중요도 시각화
shap.summary_plot(shap_values, X_normalized, feature_names=feature_names)
```
 
### 6️⃣ **정책 우선도 평가 및 유전알고리즘 최적화**
 
**정책 우선도 = 통근부담지수 × 유입인구**로 행정 영향도-시급성 이중 축 평가.
 
클러스터 0 (수지구) 대상으로 **유전알고리즘(Genetic Algorithm)** 기반 조합 최적화: 정류장별 활성도를 적응도(fitness) 함수로 정의하여 상위 8개 정류장 선정.
 
**Pareto 최적성** 원리로 통행시간-혼잡도 트레이드오프 해결.
 
```python
# 정책 우선도 계산
policy_priority = burden_index * inflow_population
 
# 유전알고리즘 적응도 함수
def fitness(solution):
    total_activity = sum(stop_activity[i] for i in solution)
    travel_time_reduction = estimate_time_savings(solution)
    return total_activity * travel_time_reduction
 
# 결과: 통근부담지수 57.1% 감소 (0.524 → 0.225)
```
 
---
 
## Key Results
 
### 📊 클러스터별 특성 및 개선 방안
 
| 클러스터 | 지역 | 핵심요인 | 대책 |
|---------|------|---------|------|
| **0** | 수지구, 기흥구, 수정구 등 (고부담) | 노선복잡도↑, 혼잡도↑ | 급행 노선 신설, 노선 단순화 |
| **1** | 분당구, 광주, 의정부 (중간부담) | 혼잡도↑, 도보시간↑ | 배차 간격 단축, 셔틀/마을버스 구축 |
| **2** | 하남, 남양주, 과천 (저부담) | 거리대비시간↑, 혼잡도↑ | 버스전용차선 확대, 직통·급행버스 신설 |
 
### 🎯 수지구 급행 노선 설계 결과
 
**통근부담지수 개선**: 0.524 → 0.225 (**57.1% 감소**)
 
- 활성도 기반 상위 8개 정류장 선정
- 통행시간 단축 및 환승 횟수 감소
- 실질적인 통근 스트레스 완화 실증
---
