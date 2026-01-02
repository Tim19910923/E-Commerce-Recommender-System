# E-Commerce Recommendation System based on Vector Space Model
> 基於向量空間模型與代數邏輯的電商推薦引擎

## 1. Project Overview (專案概述)
本專案實作了一個基於 **Memory-based Collaborative Filtering** 的推薦系統。針對 **UCI Online Retail Data Set** 中的高維稀疏數據，本系統利用線性代數中的矩陣運算，解決了認知科學中經典的 **「兩匹馬問題」 (Two Horse Problem)** ，即透過建立個體索引 (Indexing) 來區分不同使用者，並預測其潛在的購買意圖。

核心目標在於證明：即使不依賴深度語義理解，透過幾何空間中的 **插值 (Interpolation)** 與 **代數運算 (Algebraic Manipulation)**，系統仍能實現高準確度的行為泛化 。

## 2. Theoretical Framework (理論架構)
本系統將使用者行為表徵為 $n$ 維空間中的向量 (Vector Space Model)。
核心演算法採用 **餘弦相似度 (Cosine Similarity)** 來計算向量夾角，公式如下：

$$ \text{similarity}(u, v) = \frac{u \cdot v}{||u|| \times ||v||} = \frac{\sum_{i=1}^{n} u_i v_i}{\sqrt{\sum_{i=1}^{n} u_i^2} \sqrt{\sum_{i=1}^{n} v_i^2}} $$

此演算法在數學上消除了使用者活躍度 (Vector Magnitude) 的偏差，專注於偏好方向 (Orientation) 的一致性。

## 3. Key Features (核心功能)
- **Data Representation**: 處理 390 Users x 1822 Items 的稀疏矩陣 (Sparse Matrix)。
- **Algebraic Operation**: 實作矩陣分解與點積運算，解決 Cold Start 問題。
- **Evaluation**: 使用 **RMSE (Root Mean Square Error)** 量化預測向量與真實向量的歐幾里得距離。

## 4. Tech Stack (技術棧)
- **Language**: Python 3.x
- **Libraries**:
    - `pandas`: 用於建立 User-Item Matrix 與資料清洗。
    - `numpy`: 執行底層矩陣運算。
    - `scikit-learn`: 計算 Cosine Similarity 與 MSE。

## 5. Installation & Usage (安裝與執行)
1. Install dependencies (安裝依賴套件):
   ```bash
   pip install -r requirements.txt
2. Run the main script (執行主程式):
6. Results (實驗結果)
在測試環境下 (Subset N=10,000 transactions)，模型表現如下：
• Matrix Shape: (390, 1822)
• RMSE Score: 0.0985 (顯示出極高的預測收斂度)
Case Study: 針對 User 12395 (烘焙偏好者) 成功推薦出 Cake Stand, Jam Jar 等高度相關商品，驗證了演算法在捕捉隱性特徵 (Latent Features) 上的有效性。
