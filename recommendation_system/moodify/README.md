# Spotify Emotional Data Analysis

此專案旨在探索 Spotify Emotional Data 資料集，進行機器學習模型的實作、應用與優化，特別是在處理較少特徵但筆數眾多的資料集上的經驗。透過這些實驗，我們期望能建立一套在未來面對更多特徵或更大規模資料集時的分析策略。  
https://github.com/orzanai/Moodify  
https://www.kaggle.com/datasets/abdullahorzan/moodify-dataset   

## 專案目標

* **模型實作與應用：** 探索並實踐多種機器學習演算法，理解它們在特定資料集上的表現。
* **超參數調優與模型選擇：** 利用 `GridSearchCV` 和交叉驗證 (Cross-Validation) 來尋找模型的最佳參數，並比較不同模型的優劣。
* **降維技術應用：** 實驗 `PCA` 等降維方法，以壓縮資料、降低模型訓練時間並減少過擬合。
* **特徵工程與解釋性分析：** 運用 `Tree-based feature importance` 和 `SHAP` 等方法來理解特徵的重要性與模型決策。
* **建立通用分析策略：** 透過實驗經驗，為未來處理大規模資料集（多特徵、多樣本）建立更有效的分析方法。

## 資料集

* **名稱：** Spotify Emotional Data (包含近 28 萬筆記錄)
* **特徵數量：** 相對較少
* **樣本數量：** 約 28 萬筆
* **標籤：** 4 個不同的情緒標籤（可透過視覺化明顯區分）
* **來源：** `./Data/278k_song_labelled.csv` (假設路徑)
* **預處理：**
    * 載入 CSV 檔案，移除可能的 index column。
    * 將 'labels' 欄位作為目標變數。
    * 使用 `train_test_split` 將資料分割為訓練集 (80%) 和測試集 (20%)，並保持類別比例 (`stratify=y`)。
    * 使用 `StandardScaler` 對特徵進行標準化。
    * *(待辦)* 可考慮進行資料分群 (Data Grouping) 並降維至二維以進行視覺化 (e.g., t-SNE, UMAP, PCA)。

## 實驗與模型

### 1. 機器學習模型實驗

使用了以下幾種機器學習模型進行實驗：

* **已跑完的模型：**
    * **SVM (SVC):** 在大型資料集上運行較慢。
    * **Random Forest Classifier:** 訓練和調參時間較長，需仔細設定 `n_estimators`、`max_depth`、`min_samples_split`、`min_samples_leaf` 等參數。
    * **Logistic Regression:** 適合作為基準模型，或用於線性可分情況。
    * **Naive Bayes (GaussianNB):** 假設特徵之間獨立，計算速度快。
* **遇到套件問題的模型：**
    * **XGBoost:** 執行時遇到套件問題 (可能與 `libomp` 等環境設定有關)。

### 2. 超參數調優與交叉驗證

* **方法：**
    * 使用 `GridSearchCV` 進行參數搜尋。
    * 應用 `KFold` 進行交叉驗證 (`cv=5` 或 `cv=10`)，以確保模型穩定性並獲得更可靠的效能評估。
* **評估指標：** 主要使用 `accuracy`，並輔以 `f1_score` (macro average，對類別不平衡更穩健)。
* **`n_estimators` (Random Forest) 的影響：**
    * `n_estimators` 代表隨機森林中決策樹的數量。
    * **影響：** 增加 `n_estimators` 通常會提高模型的精確度，同時降低過擬合的風險，但會增加計算時間。
    * **建議：**
        * **20 vs 50 vs 100:**
            * `20`: 運行速度最快，但可能無法捕捉資料中的複雜模式，準確率可能較低。
            * `50`: 相較於 20，應有更好的表現，但仍可能不如 `100`。
            * `100`: 通常是較好的平衡點，提供了不錯的準確率，且訓練時間尚可接受。
            * **建議：** 可以從 `n_estimators=100` 開始，如果時間允許，可以嘗試 `200` 或更高，觀察準確率是否還有顯著提升。如果時間受限，`50` 或 `100` 是合理的起始點。
* **`RandomForestClassifier` 設定的考量：**
    * `class_weight='balanced'`: 在類別不平衡的資料集上，此設定有助於模型更好地學習較少類別的樣本，提高整體穩健性。
    * `max_depth`, `min_samples_split`, `min_samples_leaf`: 這些參數用於控制決策樹的生長，對防止過擬合至關重要。`GridSearchCV` 會嘗試不同的組合。

### 3. 降維 (PCA)

* **目的：**
    * **特徵萃取 (Feature Extraction):** 透過線性組合創建新的、較少數量的特徵（主成分），以捕捉原始數據中的主要變異性。
    * **降低模型訓練時間。**
    * **降低過擬合風險。**
* **選擇降維維度 k 的方法：**
    1.  **累加解釋變異性比例 (Cumulative Explained Variance Ratio):**
        * 繪製累積解釋變異性對主成分數量的圖（Scree Plot）。
        * 選擇「肘部」(elbow point)，即累積變異性增加率顯著變緩的位置。
    2.  **基於模型表現選擇：**
        * 使用不同的 k 值，訓練模型 (e.g., Random Forest, Naive Bayes, Logistic Regression)，並以 `accuracy` 作為評判指標。
        * 選擇能讓模型表現最好的 k 值。
* **實驗發現：** 在視覺化分析中，儘管不一定出現明確的「拐點」，但觀察顯示選擇 **k=7** 可以保留大部分資訊，並能用於後續的模型訓練。

### 4. 特徵重要性分析

* **Tree-based Feature Importance:**
    * 使用訓練好的 Tree-based 模型（如 Random Forest）來分析特徵的重要性。
    * **範例發現：**
        * `energy` (0.206)
        * `instrumentalness` (0.197)
        * `acousticness` (0.174)
        * 這表明 `energy`、`instrumentalness` 和 `acousticness` 是區分不同情緒標籤的關鍵特徵。
* **SHAP (SHapley Additive exPlanations):**
    * *(待辦)* 計劃使用 SHAP 來進行更深入的模型解釋，理解每個特徵如何影響模型預測。

### 5. 神經網絡 (NN) 模型

* **平台：** TensorFlow / Keras
* **模型架構：** Sequential Model，包含多層 `Dense` 和 `Dropout`，使用 `relu` 激活函數，輸出層使用 `softmax` 進行多分類。
* **超參數調優：**
    * 使用 `Keras Tuner` (RandomSearch) 進行自動超參數搜尋，優化 `val_accuracy`。
    * 搜尋的超參數包括：`units_layer1`, `units_layer2` (神經元數量), `dropout` 率, `learning_rate`, `optimizer` (adam/rmsprop)。
    * 找到最佳超參數組合後，重新構建模型並在所有訓練數據上進行訓練，使用 `EarlyStopping` 和 `ModelCheckpoint` 來保存最佳模型。
* **評估：** 在測試集上評估模型的 `loss`, `accuracy`, `f1_score` (macro)，並生成 `confusion_matrix` 和 `classification_report`。
* **[TODO] GPU 環境：** 計劃在有 GPU 的環境下進行 NN 模型分析，以加速訓練和實驗。

## 後續計畫 (TODO)

* **資料分群與降維視覺化：** 使用 t-SNE, UMAP, PCA 將資料降至二維進行視覺化分析。
* **SHAP 解釋性分析：** 應用 SHAP 庫深入理解模型決策。
* **ETL (Extract, Transform, Load)：** 建立更標準化的資料處理流程。
* **Docker 化：** 將專案容器化，便於部署和再現。
* **物件導向程式設計 (OOP)：** 為推薦系統等應用設計 OOP 架構。
* **環境輸出：** 紀錄並輸出模型訓練和評估所需的環境配置。
* **API 開發：** 將訓練好的模型部署為 API 服務。
* **其他降維方法：** 實驗 UMAP, t-SNE 等非線性降維方法。

## 實驗記錄 (Experiment Log)

* **時間：** 9/13 和 9/14 (約 10 小時)
* **輔助工具：** Gemini Flash 2.5
* **筆記來源：**
    * Jupyter Notebook (`feature_extract.ipynb`)
    * `experiment_qa.txt` (待整理)

---