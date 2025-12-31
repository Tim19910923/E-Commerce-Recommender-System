import pandas as pd

# 1. 讀取資料 (這可能會花幾秒鐘，因為資料有 50 萬筆)
print("正在讀取資料...")
df = pd.read_excel('Online Retail.xlsx')

# 2. 資料清理 (Data Cleaning)
# 去除沒有 UserID 的資料，並去除退貨資料 (Quantity < 0)
df = df[df['CustomerID'].notna()]
df = df[df['Quantity'] > 0]

# 為了示範，我們先取前 10,000 筆資料來跑，避免你電腦跑不動
# 等你熟練後，這行可以註解掉，跑全量數據
df = df.iloc[:10000]

print(f"資料清理完成，剩餘 {len(df)} 筆交易紀錄。")

# 3. 建立「使用者-商品」矩陣 (User-Item Matrix)
# 列 (Index) 是使用者 ID，欄 (Columns) 是商品代碼，值 (Values) 是購買數量
# fillna(0) 是將沒買過的商品補 0
user_item_matrix = df.pivot_table(index='CustomerID', 
                                  columns='StockCode', 
                                  values='Quantity').fillna(0)

# 將購買數量轉換為 0/1 (有買/沒買)，我們暫時只關心「有沒有興趣」，不關心買幾個
user_item_matrix = user_item_matrix.map(lambda x: 1 if x > 0 else 0)

print("矩陣建立完成！")
print(f"矩陣大小：{user_item_matrix.shape}") 
# 這裡會顯示 (使用者數量, 商品數量)
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd # 確保 pandas 被正確引用

# 4. 計算使用者之間的相似度
# 這行程式碼就是在執行報告中的數學公式：u . v / (|u| * |v|)
user_similarity = cosine_similarity(user_item_matrix)

# 將結果轉回 DataFrame，方便我們看是誰跟誰像
user_similarity_df = pd.DataFrame(user_similarity, 
                                  index=user_item_matrix.index, 
                                  columns=user_item_matrix.index)

print("相似度計算完成！")

# --- 讓我們來測試一下 ---
# 假設我們要針對第一個使用者 (我們叫他 Target User) 進行推薦
target_user_id = user_item_matrix.index[0]

# 找出跟 Target User 最像的另一個使用者 (排除自己)
similar_users = user_similarity_df[target_user_id].sort_values(ascending=False)
most_similar_user_id = similar_users.index[3] # index 是自己，所以取 [3]

print(f"與使用者 {target_user_id} 最像的人是： {most_similar_user_id}")
# 5. 產生推薦清單
# 找出「最像的人」買過的所有東西
items_bought_by_similar = set(user_item_matrix.loc[most_similar_user_id].iloc[user_item_matrix.loc[most_similar_user_id].to_numpy().nonzero()].index)

# 找出「目標使用者」買過的所有東西
items_bought_by_target = set(user_item_matrix.loc[target_user_id].iloc[user_item_matrix.loc[target_user_id].to_numpy().nonzero()].index)

# 兩者相減，就是推薦清單 (Difference set)
recommendations = items_bought_by_similar - items_bought_by_target

print(f"為使用者 {target_user_id} 產生的推薦商品代碼：")
print(recommendations)

if not recommendations:
    print("目前沒有推薦，因為這兩人買的東西完全一樣，或是最像的人買得比較少。")
    # --- 6. 優化輸出：顯示商品名稱 ---
# 建立一個 商品代碼 -> 商品名稱 的對照表
item_lookup = df[['StockCode', 'Description']].drop_duplicates().set_index('StockCode').to_dict()['Description']

print(f"\n為使用者 {target_user_id} 推薦的商品詳情：")
for item_code in recommendations:
    # 嘗試從對照表找名稱，找不到就顯示 'Unknown'
    item_name = item_lookup.get(item_code, 'Unknown Item')
    print(f"- [{item_code}] {item_name}")
    # --- 7. 模型評估 (簡單版 RMSE) ---
from sklearn.metrics import mean_squared_error
import numpy as np

# 為了評估，我們通常會預測矩陣中的值，然後跟實際值比對
# 這裡我們簡單計算一下：利用相似度矩陣預測使用者的購買行為
# 預測分數 = 相似度矩陣 * 使用者商品矩陣 / 相似度總和
# (這是一個標準的加權平均預測公式)

print("\n正在計算模型誤差 (RMSE)...")

# 處理除以零的微小數值避免報錯
user_similarity_sums = np.abs(user_similarity).sum(axis=1).reshape(-1, 1)
user_similarity_sums[user_similarity_sums == 0] = 1.0 # 避免除以 0

# 預測矩陣
prediction_matrix = user_similarity.dot(user_item_matrix) / user_similarity_sums

# 計算 RMSE (只計算那些原本就有資料的地方，但在這裡我們簡化計算全矩陣誤差)
# 實際上我們通常會把資料拆成「訓練集」跟「測試集」，但第一階段先跑通為主
mse = mean_squared_error(user_item_matrix, prediction_matrix)
rmse = np.sqrt(mse)

print(f"模型的 RMSE (均方根誤差): {rmse:.4f}")
print("RMSE 越低，代表預測越準確（在這個 0/1 矩陣中，通常會很低）。")