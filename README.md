## 簡單介紹
#### Model → final_analysis.py
#### dataset → sacropneia_F_Student_20260315.csv
#### Output labeled of sarcopenia ( 0 未確診 ; 1 確診 ) → Sarcopenia_Labeled_Data_2026.csv

## 第一部分：完整實驗與分析流程 
定義問題→排除雜訊→多維度建模→臨床建議。流程分為以下六大步驟：
### Step 1: 資料前處理與標記 (Data Preprocessing & Labeling)
載入資料：讀取包含 591 位女性（其中 433 位為 65 歲以上）的生理與測量數據 CSV 檔。 (File → sacropneia_F_Student_20260315.csv)
生成標準答案 (Ground Truth)：原始資料並無「肌少症」標籤。我們依照亞洲/歐洲醫學標準(第二部分會提到)，透過程式自動將「握力不足且肌肉量不足」的受試者標記為 1（患病），其餘為 0。
```
# 建立目標變數：肌少症 (女性：握力 < 18 且 ASMI <= 5.7)
df['sarcopenia'] = ((df['muscle_power'] < 18) & (df['muscle_mass'] <= 5.7)).astype(int)
df_clean = df.dropna(subset=features + ['muscle_power', 'muscle_mass', 'sarcopenia'])
```

### Step 2: 變數篩選與防範數據洩漏 (Feature Selection & Prevent Data Leakage)
關鍵修正：為了開發真正具備「預測」價值的模型，我們嚴格將測量力量的 new_test、muscle_power、muscle_mass 等會導致 模型作弊（AUC=1.0）的目標變數剔除。
確立預測因子：僅保留年齡、身高、體重、小腿圍、功能問卷等 5 項「外觀與基礎生理指標」作為預測特徵。

### Step 3: 統計檢定與基準建立 (Statistical Modeling)
使用 OLS 線性迴歸預測「肌肉力量」與「肌肉質量」，並透過 P-value 找出統計學上的顯著因果關係（例如：小腿圍對肌肉質量有極顯著的正相關）。
使用 邏輯迴歸 (Logistic Regression) 作為預測肌少症的 Baseline（基準線），觀察傳統統計學的分類效能（AUC 約 0.86~0.89）。

### Step 4: 機器學習演算法開發 (Machine Learning)
迴歸任務 (Regression)：導入 Ridge (嶺迴歸) 與 Random Forest (隨機森林)。發現隨機森林在預測力量/質量時，RMSE 顯著較低且 R² 高達 0.94，證明生理數據間存在非線性關係。
分類任務 (Classification)：導入 XGBoost 與 Random Forest 來預測肌少症。

### Step 5: 模型評估與視覺化 (Evaluation & Visualization)
產出並對比 ROC 曲線。證實機器學習模型（特別是 XGBoost）在預測肌少症上，AUC 高達 0.97 以上，展現極強的疾病捕捉能力。

### Step 6: 發展臨床簡約模型 (Parsimonious Model)
綜合 XGBoost 的「特徵重要性 (Feature Importance)」與統計學的 P-value，剔除貢獻度極低且不顯著的「體重」與「問卷」。
最終輸出完整的標記資料表（CSV），並建議未來開發僅需少數變數的快速篩檢工具。



## 第二部分：如何就現有資訊判斷出肌少症？
判斷肌少症其實分為兩個層次：黃金標準（實驗室怎麼看）與預測標準（未來社區診所怎麼看）。

### 1. 實驗室的黃金標準 (Ground Truth 標記法)
這是我們程式碼中用來產生 sarcopenia 答案的絕對標準（參考 EWGSOP2 準則）：
必須同時滿足以下兩個條件，才會被判定為肌少症：
肌肉力量不足：女性握力 (Muscle Power)<𝟏𝟖" kg"。
肌肉質量不足：四肢骨骼肌質量指數 (Muscle Mass / ASMI)≤𝟓.𝟕" " 〖"kg\/m" 〗^𝟐。
備註：只要缺其中一項（例如握力很低但肌肉量正常），在此專案中都不算確診肌少症。

### 2. 精簡化預測標準 (演算法篩檢法)
簡而言之，透過不看握力與肌肉量的資訊，觀察模型還對那些資訊感興趣來當作判斷依據。
根據機器學習成果（XGBoost），即使不測量握力與肌肉量，只要看以下現有資訊，就能以高達 97% 的準確率判斷（預測）她有沒有肌少症：
第一指標：小腿圍 (Calf Circumference) → 權重佔 35% ~ 40%
小腿圍越低，罹患肌少症的機率呈指數型上升。它是肌肉量流失最直觀的外部表現。
第二指標：身高 (Height) → 權重佔 25%
身高與骨骼肌的分佈比例高度相關，在模型中佔據極大權重。
第三指標：年齡 (Age) → 權重佔 12% ~ 14%
隨著年齡增長，自然老化導致的神經肌肉退化，是不可逆的風險因子。


