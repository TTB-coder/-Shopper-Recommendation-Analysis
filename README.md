# 🛒 ShopperLense: Shopper Personalized Recommendation System
![Screenshot 2025-04-01 at 8 29 43 PM](https://github.com/user-attachments/assets/a673a8a8-4e58-4e78-97dd-0de58aca24fa)

---

## 🌟 Project Overview
**ShopperLens** is an advanced AI-powered recommendation system designed to enhance the online shopping experience through **personalized product suggestions**. By leveraging **machine learning and data-driven analytics**, ShopperLens delivers **highly relevant recommendations** to users based on their behavior, past purchases, and session interactions.
![ProjectPPT - Trishna_page-0002](https://github.com/user-attachments/assets/144007d6-1f55-4cce-bdbe-dac02bf2cdee)


### 🔥 **Key Features**
- ✅ **Three-Branch Model Architecture**: Memory-Based, Collaborative Filtering, and Item-Based approaches.
- ✅ **Hyperparameter-Tuned Models**: LightGBM, SVD, and KNN for enhanced performance.
- ✅ **Data Preprocessing & Feature Engineering**: Handling imbalances, encoding categorical data, and scaling numerical values.
- ✅ **User-Centric Testing**: Sample data inputs showcase real-time recommendation workflow.
- ✅ **Optimized Performance**: Model evaluation metrics guide model selection and refinements.

### 🏗 **Tech Stack**
🚀 **Languages**: Python  
📊 **Libraries**: LightGBM, Scikit-learn, Pandas, NumPy, Seaborn, Surprise, Matplotlib  
🛠 **Tools**: Jupyter Notebook, GitHub

---

## 📌 Table of Contents
1. [🚀 Project Motivation](#-project-motivation)
2. [📊 Dataset Overview](#-dataset-overview)
3. [📈 Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
4. [🛠 Data Preprocessing](#-data-preprocessing)
5. [📡 Model Architecture](#-model-architecture)
- [Branch 1: Memory-Based Learning (LightGBM)](#-branch-1-memory-based-learning-lightgbm)
- [Branch 2: Collaborative Filtering (SVD + LightGBM)](#-branch-2-collaborative-filtering-svd--lightgbm)
- [Branch 3: Item-Based Filtering (KNN)](#-branch-3-item-based-filtering-knn)
7. [🧪 Sample Data & Testing](#-sample-data--testing)
8. [📊 Performance Metrics](#-performance-metrics)
9. [💡 Key Insights](#-key-insights)
10. [🚀 Future Enhancements](#-future-enhancements)
11. [📚 References](#-references)

---
![ProjectPPT - Trishna_page-0003](https://github.com/user-attachments/assets/740fc2a3-6ef8-430f-bad5-96038eeb80f1)

## 🚀 Project Motivation
E-commerce platforms face challenges in delivering **highly relevant recommendations** due to:
- **Sparse user-item interaction data**, affecting prediction accuracy.
- **Cold start problem** for new users and products.
- **Balancing personalization vs. diversity** in suggestions.

ShopperLens tackles these issues by employing a **multi-model recommendation system** that improves accuracy and engagement.
![ProjectPPT - Trishna_page-0004](https://github.com/user-attachments/assets/27c62f3b-8c48-4ac6-a76f-7f947f812030)

---

## 📊 Dataset Overview
The dataset used is **E-Shop Clothing 2008**, consisting of **shopping behavior data collected from an online retail store**. 
([UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/553/clickstream+data+for+online+shopping)).
It includes:
- ✔ **User Sessions** – Tracks clicks, views, and purchase interactions.
- ✔ **Product Attributes** – Details such as color, category, price, metadata.
- ✔ **Transaction History** – Captures orders, session IDs, and timestamps.  

📌 **Variable Descriptions & Mappings**:  
- **Country Mapping**: `1 -> Australia, 2 -> Austria, 3 -> Belgium, ... 42 -> USA, etc.`  
- **Color Mapping**: `1 -> Beige, 2 -> Black, 3 -> Blue, 4 -> Brown, ...`  
- **Main Categories**: `1 -> Trousers, 2 -> Skirts, 3 -> Blouses, 4 -> Sale`  
![ProjectPPT - Trishna_page-0005](https://github.com/user-attachments/assets/a0f4dc89-bc48-4893-9917-6b7dab9990dc)

### 📌 Data Processing Steps
🔹 **Handling Missing Values** – Using Mean/Mode imputation.  
🔹 **Encoding Categorical Variables** – Applying OneHotEncoder & LabelEncoder.  
🔹 **Balancing Data** – Addressing class imbalance with **ADASYN & SMOTE**.  
🔹 **Feature Engineering** – Generating explicit rating bins for Item-Based Filtering.  
![ProjectPPT - Trishna_page-0008](https://github.com/user-attachments/assets/a19d301e-d8e3-495d-a931-5b55c3560c00)


---

## 📈 Exploratory Data Analysis (EDA)
Before training the models, we conducted **EDA to understand data distributions, patterns, and correlations**.
![ProjectPPT - Trishna_page-0009](https://github.com/user-attachments/assets/3de27890-ed13-4abf-8c80-fe74c97ae01b)

### 📊 Key Findings
- **Most frequently purchased product categories** were analyzed.
- **Price distribution across different countries** was visualized.
- **User behavior trends** were identified through session duration analysis.
- **Feature correlations** were analyzed to select the most impactful attributes for modeling.

📌 **Visualizations:**
- **Purchase Frequency Distribution**
- **Top Selling Product Categories**
- **Distributions of Color and other features**
![ProjectPPT - Trishna_page-0010](https://github.com/user-attachments/assets/893aec2a-57d2-4df3-81d3-90c067ae2c60)
![ProjectPPT - Trishna_page-0011](https://github.com/user-attachments/assets/52c1b8ae-bed2-4ea0-98eb-5cd859e84b89)
![ProjectPPT - Trishna_page-0012](https://github.com/user-attachments/assets/0ee2c83e-af45-4e2c-8c9d-55abc653da4c)


---

## 🛠 Model Architecture
ShopperLens follows a **multi-branch modeling approach**, consisting of:
- 1️⃣ **Memory-Based Model (LightGBM)** – Analyzes session attributes.
- 2️⃣ **Collaborative Filtering (SVD + LightGBM)** – Learns user-item interaction patterns.
- 3️⃣ **Item-Based Filtering (KNN)** – Recommends similar products using cosine similarity.
📌 **Overall System Flowchart**  
<div align = "center">
<img src="https://github.com/user-attachments/assets/bfd13c0c-f577-40b3-bcc1-8ca1c254de81" style="width:50%;">    
</div>

### **Branch 1: Memory-Based Learning (LightGBM)**
- **Objective:** Predict user preferences based on session attributes.
- **Hyperparameters:** `max_depth=3, n_estimators=4, learning_rate=0.06`
- **Result:** Achieved **80% accuracy** and **AUC score of 0.98**.
- **Use Case:** Works well for users with clear session behavior.
![ProjectPPT - Trishna_page-0014](https://github.com/user-attachments/assets/91807f10-46ee-4da7-8cba-c1d1857be242)

📌 **Outputs:**  
![ProjectPPT - Trishna_page-0015](https://github.com/user-attachments/assets/19a4cdd1-9f3d-431d-b4b5-f45da8ef9339)
![ProjectPPT - Trishna_page-0016](https://github.com/user-attachments/assets/ecfca371-77ff-402a-8093-7d8f17168627)

### **Branch 2: Collaborative Filtering (SVD + LightGBM)**
- **Objective:** Learn user-item relationships for recommendations.
- **Hyperparameters:** `SVD + LightGBM for further refinement`
- **Result:** Achieved **75% accuracy** and **AUC score of 0.97**.
- **Use Case:** Useful for repeat customers with historical data.
![ProjectPPT - Trishna_page-0017](https://github.com/user-attachments/assets/08f2ba94-81e1-41c5-bf30-158eab769dd2)

📌 **Outputs:**  
![ProjectPPT - Trishna_page-0018](https://github.com/user-attachments/assets/5b08d975-3e11-4817-86c2-41482eb3b0c5)
![ProjectPPT - Trishna_page-0019](https://github.com/user-attachments/assets/ccc01cac-aaee-4d90-aca1-41378806ff95)


### **Branch 3: Item-Based Filtering (KNN)**
- **Objective:** Find similar products based on cosine similarity.
- **Hyperparameters:** `K=10 neighbors`
- **Result:** Achieved **RMSE = 0.4777, MAE = 0.3880**.
- **Use Case:** Ideal for similar product suggestions.
![ProjectPPT - Trishna_page-0020](https://github.com/user-attachments/assets/824d07b5-f5f4-46f2-8db1-24b946eb7925)

📌 **Outputs:**  
![ProjectPPT - Trishna_page-0021](https://github.com/user-attachments/assets/7cde6caf-1196-418d-bb4c-ce8b6c6883b8)
![ProjectPPT - Trishna_page-0022](https://github.com/user-attachments/assets/8c370925-50f9-4cbd-85da-d20a470a8c51)


---

## 🧪 Sample Data & Testing
ShopperLens uses sample test cases to **validate recommendation accuracy**.

### 📌 Sample Test Case Workflow:
- 1️⃣ **New User Data Input** – Generates recommendations based on limited session history.
- 2️⃣ **Session-Based Predictions** – Provides product recommendations using LightGBM.
- 3️⃣ **Personalized Recommendation Output** – Displays top predicted products based on model rankings.  

📌 **Sample Code Execution:**
```python
new_data_point = pd.DataFrame({
    'year': [2008], 'month': [5], 'day': [10], 'price': [20],
    'order': [3], 'country': [1], 'page 1 (main category)': [2],
    'page 2 (clothing model)': ['A11'], 'location': [2],
    'model photography': [1], 'price 2': [1], 'page': [2]
})
recommendations = lgb_model.predict(preprocessor.transform(new_data_point))
print("Recommended Product Colors:", recommendations)
```
📌 **Visualization of Recommendations:**

![ProjectPPT - Updated Reco_page-0001](https://github.com/user-attachments/assets/90aab7e2-a448-4d53-bb20-a8e61a6fa5ee)
![ProjectPPT - Updated Reco_page-0002](https://github.com/user-attachments/assets/120d7df1-596a-4636-a1f1-377e54ae3d00)


---

## 📊 Performance Metrics
| **Model** | **Accuracy / RMSE** | **AUC Score** | **Strengths** | **Weaknesses** |
|-----------|-----------------|------------|------------|-------------|
| LightGBM | **80%** | **0.98** | Session-based, high accuracy | Limited interpretability |
| SVD + LightGBM | **75%** | **0.97** | Works well with sparse data | Cold start issue |
| KNN | **RMSE 0.4777** | N/A | Finds similar products | Sensitive to K hyperparameter |

The results indicate that **LightGBM provides the highest accuracy**, making it suitable for session-based predictions, while **SVD + LightGBM effectively captures user-item relationships**. **KNN serves as a complementary approach**, improving recommendations based on item similarity.

---

## 💡 Key Insights
📌 **Memory-Based Learning** provides high accuracy but requires well-defined session attributes.  
📌 **Collaborative Filtering** captures hidden patterns but is affected by data sparsity.  
📌 **Item-Based Filtering** is highly interpretable but sensitive to hyperparameter tuning.  

---

## 🚀 Future Enhancements
![ProjectPPT - Trishna_page-0025](https://github.com/user-attachments/assets/af279c58-ebe0-4300-baf6-9e4c27f4eb7e)

---

## 📚 References
- 📖 [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- 📖 [Surprise Library for Collaborative Filtering](https://surpriselib.com/)
- 📖 [Scikit-Learn Documentation](https://scikit-learn.org/)
- 📖 [Dataset (UC Irvine Machine Learning Repository)](https://archive.ics.uci.edu/dataset/553/clickstream+data+for+online+shopping)  
---

## 📌 Get in Touch
🔗 [LinkedIn](https://www.linkedin.com/in/ttb1)  
🔗 [GitHub](https://github.com/TTB-coder)
![Screenshot 2025-04-01 at 8 23 53 PM](https://github.com/user-attachments/assets/f66d2ef5-ec69-4143-a643-59fdc6c23569)





