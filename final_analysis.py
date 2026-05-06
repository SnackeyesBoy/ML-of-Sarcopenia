import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, roc_curve, auc
import warnings

# 隱藏收斂警告以保持輸出整潔
warnings.filterwarnings('ignore')

def main():
    # ==========================================
    # 0. 數據讀取與預處理
    # ==========================================
    file_path = 'sacropneia_F_Student_20260315.csv'
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    except:
        df = pd.read_csv(file_path, encoding='big5')
    
    df.columns = df.columns.str.strip()

    # 預測特徵 (嚴格排除目標變數以防數據洩漏)
    features = ['age', 'calf_circ', 'weight', 'height', 'function_survey']
    
    # 強制轉型確保無 ValueError
    for col in features + ['muscle_power', 'muscle_mass']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 建立目標變數：肌少症 (女性：握力 < 18 且 ASMI <= 5.7)
    df['sarcopenia'] = ((df['muscle_power'] < 18) & (df['muscle_mass'] <= 5.7)).astype(int)
    df_clean = df.dropna(subset=features + ['muscle_power', 'muscle_mass', 'sarcopenia'])

    # ==========================================
    # === 新增：將帶有 Ground Truth 的資料表存檔 ===
    # ==========================================
    output_filename = 'Sarcopenia_Labeled_Data_2026.csv'
    df_clean.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\n[系統提示] 已成功將包含 'sarcopenia' 標籤的最終分析資料集儲存為: {output_filename}\n")
    # ==========================================

    groups = {
        "All_Women": df_clean.copy(),
        "Older_Women_65plus": df_clean[df_clean['age'] >= 65].copy()
    }

    for name, data in groups.items():
        print(f"\n\n{'#'*20} {name} 分析報告 (樣本數={len(data)}) {'#'*20}")
        X = data[features]
        X_const = sm.add_constant(X)
        
        y_power = data['muscle_power']
        y_mass = data['muscle_mass']
        y_sarco = data['sarcopenia']

        # ==========================================
        # 1. 統計模型 (Linear & Logistic Regression)
        # ==========================================
        print("\n=== 第一部分：統計模型 ===")
        
        ols_power = sm.OLS(y_power, X_const).fit()
        print("\n[預測肌肉力量 (Linear Regression) - 觀察 P-value]")
        print(ols_power.summary().tables[1])
        
        ols_mass = sm.OLS(y_mass, X_const).fit()
        print("\n[預測肌肉質量 (Linear Regression) - 觀察 P-value]")
        print(ols_mass.summary().tables[1])
        
        logit_sarco = sm.Logit(y_sarco, X_const).fit(disp=0)
        logit_prob = logit_sarco.predict(X_const)
        print("\n[預測肌少症存在 (Logistic Regression) - 觀察 P-value]")
        print(logit_sarco.summary().tables[1])

        # ==========================================
        # 2. 機器學習模型與評估
        # ==========================================
        print("\n=== 第二部分：機器學習模型與評估 ===")
        
        # --- A. 迴歸模型評估 (預測力量與質量) ---
        print("\n[迴歸模型評估 (MAE, RMSE, R2)]")
        models_reg = {
            "Ridge Regression": Ridge(alpha=1.0, random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        for target_name, y_target in [("肌肉力量", y_power), ("肌肉質量", y_mass)]:
            print(f"\n目標：{target_name}")
            for ml_name, ml_model in models_reg.items():
                ml_model.fit(X, y_target)
                y_pred = ml_model.predict(X)
                mae = mean_absolute_error(y_target, y_pred)
                rmse = np.sqrt(mean_squared_error(y_target, y_pred))
                r2 = r2_score(y_target, y_pred)
                print(f"  - {ml_name:17} | MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}")

        # --- B. 分類模型評估 (預測肌少症) ---
        print("\n[分類模型評估 (AUC)]")
        rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X, y_sarco)
        rf_prob = rf_clf.predict_proba(X)[:, 1]
        
        xgb_clf = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, eval_metric='logloss').fit(X, y_sarco)
        xgb_prob = xgb_clf.predict_proba(X)[:, 1]
        
        print(f"  - Logistic Regression AUC: {roc_auc_score(y_sarco, logit_prob):.4f}")
        print(f"  - Random Forest AUC      : {roc_auc_score(y_sarco, rf_prob):.4f}")
        print(f"  - XGBoost AUC            : {roc_auc_score(y_sarco, xgb_prob):.4f}")

        # ==========================================
        # 3. 特徵重要性與簡約模型建議
        # ==========================================
        print("\n=== 第三部分：特徵重要性 ===")
        importances = pd.Series(xgb_clf.feature_importances_, index=features).sort_values(ascending=False)
        print("\n[特徵相對重要性 (XGBoost)]")
        print(importances.apply(lambda x: f"{x*100:.2f}%"))

        # ==========================================
        # 4. 繪製 4 張 ROC 曲線圖 (3 獨立 + 1 綜合)
        # ==========================================
        model_results = [
            (logit_prob, 'Logistic_Regression', 'green'),
            (rf_prob, 'Random_Forest', 'blue'),
            (xgb_prob, 'XGBoost', 'red')
        ]
        
        # A. 產出 3 張獨立圖
        for prob, label, color in model_results:
            fpr, tpr, _ = roc_curve(y_sarco, prob)
            cur_auc = auc(fpr, tpr)
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, color=color, lw=2, label=f'AUC = {cur_auc:.4f}')
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
            plt.title(f'ROC Curve - {label.replace("_", " ")} ({name})')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc='lower right')
            plt.grid(alpha=0.3)
            plt.savefig(f'ROC_{name}_Single_{label}.png')
            plt.close()

        # B. 產出 1 張綜合對比圖
        plt.figure(figsize=(8, 6))
        for prob, label, color in model_results:
            fpr, tpr, _ = roc_curve(y_sarco, prob)
            cur_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2, label=f'{label.replace("_", " ")} (AUC = {cur_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.title(f'Comprehensive ROC Comparison - {name}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.savefig(f'ROC_{name}_Comparison_All.png')
        plt.close()
        
        print(f"\n  => 已成功為 {name} 儲存 4 張 ROC 曲線圖檔！")

if __name__ == "__main__":
    main()