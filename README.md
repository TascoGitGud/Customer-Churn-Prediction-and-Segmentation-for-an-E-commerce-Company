# 🔮 Python / Machine Learning | Customer Churn Prediction & Segmentation for an E-commerce Company

![Python](https://img.shields.io/badge/Language-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Tool-Jupyter_Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square)

---

<p align="center">
  <img src="Churn_Prediction/Banner.png" width="100%">
</p>

_Predict which customers are likely to leave - and group them into segments so the company can run the right promotions for each group._

- 🎯 **Business Question:** Who will churn next, and what kind of customer are they?
- 🏬 **Domain:** E-commerce
- 🛠️ **Tools:** Python (pandas, scikit-learn, seaborn, matplotlib)

👤 Author: Bạch Minh Nam

---

## 📑 Table of Contents
1. [📌 Background & Overview](#-background--overview)
2. [📂 Dataset Description & Data Structure](#-dataset-description--data-structure)
3. [🗂️ Project Structure](#️-project-structure)
4. [⚒️ Main Process](#️-main-process)
5. [🔎 Final Conclusion & Recommendations](#-final-conclusion--recommendations)
6. [🚀 How to Run This Project](#-how-to-run-this-project)

---

## 📌 Background & Overview

### 🎯 Objective

An e-commerce company wants to reduce the number of customers who stop using their platform (churned users). To do this, the team needs to:

✔️ Understand **how churned customers behave** - so the team knows what warning signs to look for.

✔️ Build a **machine learning model** to predict which customers are likely to churn.

✔️ **Segment churned customers** into smaller groups - so the marketing team can design targeted promotions for each group instead of a one-size-fits-all campaign.

### 👤 Who is this project for?

✔️ Marketing and CRM teams - to design better retention programs.

✔️ Data analysts - to understand churn behavior patterns.

✔️ Business decision-makers - to prioritize which customer groups to act on first.

---

## 📂 Dataset Description & Data Structure

### 📌 Data Source
- 📁 Dataset: `churn_prediction`
- 📄 Format: `.xlsx`

### 📊 Data Structure

#### 1️⃣ Tables Used
One main table containing customer behavior and account information.

#### 2️⃣ Table Schema

| Column Name | Data Type | Description |
|---|---|---|
| `CustomerID` | INT | Unique ID for each customer |
| `Churn` | INT | 1 = churned, 0 = stayed |
| `Tenure` | FLOAT | How long the customer has been with the company |
| `PreferredLoginDevice` | TEXT | Device the customer usually logs in with |
| `CityTier` | INT | City tier (1, 2, or 3) |
| `WarehouseToHome` | FLOAT | Distance from warehouse to customer's home |
| `PreferPayment` | TEXT | Preferred payment method |
| `Gender` | TEXT | Customer gender |
| `HourSpendOnApp` | FLOAT | Hours spent on the app or website |
| `NumberOfDeviceRegistered` | INT | Number of devices registered |
| `PreferedOrderCat` | TEXT | Preferred product category in the last month |
| `SatisfactionScore` | INT | Customer satisfaction score |
| `MaritalStatus` | TEXT | Marital status |
| `NumberOfAddress` | INT | Number of saved addresses |
| `Complain` | INT | 1 = filed a complaint in the last month |

---

## 🗂️ Project Structure

```
Churn_Prediction/
    ├── Banner.png                          
    ├── Customer_Churn_Prediction.ipynb     # Main notebook (EDA + ML + Clustering)
    └── churn_prediction.xlsx               # Raw dataset
```

---

## ⚒️ Main Process

### 🔍 Part A - EDA (Exploratory Data Analysis)

#### 🧹 Step 1: Data Cleaning

Before any analysis, the data needs to be clean and complete.

- 🔲 **Missing values:** Filled with the median for numeric columns, and the mode (most common value) for text columns.
- 🗑️ **Duplicate rows:** Removed completely.
- 📐 **Outliers:** Detected using the IQR method and removed for columns like `Tenure`, `WarehouseToHome`, `HourSpendOnApp`, `OrderCount`, and others. This step prevents extreme values from distorting the model.

#### 📈 Step 2: Univariate Analysis

Looking at each column individually to understand the overall data shape.

Key findings:
- ⚠️ The dataset is **imbalanced** - customers who stayed (`Churn` = 0) are about 5 times more than those who churned (`Churn` = 1).
- 🕐 Customers tend to churn **early in their time with the platform** - `Tenure` distribution is right-skewed, meaning new customers leave more often.

#### 🔗 Step 3: Bivariate Analysis - Churn Behavior by Group

This step compares churned vs. non-churned customers across different dimensions to find which groups are most at risk.

**Key observations:**

- 😡 **`Complain`:** Customers who filed complaints churned at a much higher rate than those who didn't. This is the clearest behavioral signal for churn.
- ⏳ **`Tenure`:** Churned customers tend to have shorter time with the company. Customers in their first few months are the most likely to leave.
- 🏙️ **`CityTier`:** Customers in Tier 1 and Tier 3 cities churn more than Tier 2. This may be due to differences in delivery quality or service coverage.
- 📱 **`PreferedOrderCat`:** Customers who often buy **Mobile Phones** and **Laptops & Accessories** show higher churn - these are high-value categories where a bad experience likely has a bigger impact.
- 💍 **`MaritalStatus`:** Single customers churn more than married or divorced ones - possibly because they have less long-term commitment to a platform.
- 👨 **`Gender`:** Male customers show slightly higher churn in both count and rate.

#### 📐 Step 4: Outlier Treatment

Outliers were removed using the IQR (Interquartile Range) method on 8 numeric columns. Keeping outliers would skew both the EDA findings and the machine learning model's performance.

#### 🌲 Step 5: Feature Importance with Random Forest (Base Model)

Before building the full model, a basic Random Forest was trained to see which features matter most for predicting churn. This helps confirm that the analysis in Step 3 is on the right track.

---

### 🤖 Part B - Supervised Learning (Churn Prediction)

The goal here is to build a model that can predict whether a customer will churn or not.

#### ✂️ Step 1: Train-Test Split

The dataset was split into 80% for training and 20% for testing. `stratify=y` was used to make sure the churn ratio is the same in both sets - important because the data is imbalanced.

#### 🌲 Step 2: Train Random Forest Model

A **Random Forest Classifier** was chosen because it handles mixed data types well, is not easily affected by noise, and provides feature importance scores.

Settings used:
- `n_estimators=200` - 200 decision trees
- `max_depth=12` - limits tree depth to avoid overfitting
- `class_weight='balanced'` - tells the model to pay more attention to the minority class (churned customers), since there are far fewer of them

#### 📊 Step 3: Model Evaluation

The model was evaluated using three metrics:

- 📋 **Classification Report** - shows precision, recall, and F1-score for both churn and non-churn groups.
- 📉 **ROC-AUC Score** - measures how well the model separates churned from non-churned. Closer to 1.0 is better.
- 🟩 **Confusion Matrix** - shows how many churned customers the model correctly caught vs. missed.

#### ⚙️ Step 4: Hyperparameter Tuning

To improve the model, **GridSearchCV** was used to test different combinations of settings:

| Parameter | Values Tested |
|---|---|
| `n_estimators` | 50, 100, 200 |
| `max_depth` | None, 10, 20 |
| `min_samples_split` | 2, 5 |
| `bootstrap` | True, False |

The best settings found were then applied to produce the final tuned model.

---

### 🔵 Part C - Unsupervised Learning (Customer Segmentation)

Once churned customers are identified, the next question is: **are all churned customers the same?** If not, the company can offer different promotions to different groups.

Only churned customers (`Churn` = 1) were used in this part.

#### 🔢 Step 1: Find the Right Number of Clusters

Using all numeric columns at once did not produce clear clusters - the Elbow Method showed no obvious "bend" and Silhouette Scores were too low.

The solution was to test **4 different feature sets** and compare results:

| Feature Set | Description | Result |
|---|---|---|
| Set1 - Behavioral & Value | `Tenure`, `Cashback`, `SatisfactionScore`, `Complain`, `OrderCount`, `DaySinceLastOrder`, `HourSpendOnApp` | ✅ Best - clear peak at K=6, acceptable at K=3 |
| Set2 - Logistics & Service | `WarehouseToHome`, `Complain`, `SatisfactionScore`, `CouponUsed`, `DaySinceLastOrder` | ⚠️ OK but weaker than Set1 |
| Set3 - All Numeric | All numeric columns combined | ❌ Silhouette too low (0.07–0.11) |
| Set4 - Behavioral (no Satisfaction) | Same as Set1 but without `SatisfactionScore` | ❌ Score keeps rising with K - no clear peak |

➡️ **Final choice: Set1 with K = 3**

K=3 was chosen over K=6 because it gives 3 distinct, meaningful groups that are practical for the marketing team to act on.

#### 🏷️ Step 2: Run Final Clustering & Label Each Group

KMeans was run with K=3 on Set1 features (after `StandardScaler` normalization). Each churned customer was assigned to one of three clusters.

#### 🧩 Step 3: Cluster Profiling - Who is in Each Group?

| Metric | Cluster 0 | Cluster 1 | Cluster 2 |
|---|---|---|---|
| Count | 247 customers | 220 customers | 281 customers |
| `Tenure` (avg) | 3.51 | Medium | 3.99 (highest) |
| `CashbackAmount` (avg) | 163.8 (highest) | Medium | 131.5 (lowest) |
| `Complain` rate | ~98% | 0% | ~55% |
| `SatisfactionScore` | Medium | 3.79 (highest) | Medium |
| `DaySinceLastOrder` | Medium | 3.03 (highest - long gap) | Medium |
| `OrderCount` | Medium | Medium | 1.06 (lowest) |

🔴 **Cluster 0 - "Long-term, high-value customers with complaints"**
These customers stayed the longest and got the most cashback - but nearly all of them filed a complaint. They also live farthest from the warehouse. They are leaving because of **service and delivery problems**, not because they don't like the platform.

🟡 **Cluster 1 - "Happy but quietly drifting away"**
No complaints, highest satisfaction - but the longest gap since their last order. They are not angry, they are just **becoming less active** over time and eventually stopping.

🟢 **Cluster 2 - "Loyal but under-rewarded"**
Highest `Tenure` (longest time with the company), but lowest `CashbackAmount` and lowest `OrderCount`. They have been around the longest but seem to feel they are **not getting enough value back** for their loyalty.

---

## 🔎 Final Conclusion & Recommendations

📍 Key Takeaways:

🔴 **For Cluster 0 - "Long-term, high-value customers with complaints"**

✔️ Reach out personally to resolve their complaints - these are high-value customers and losing them hurts the most.

✔️ 🚚 Offer free shipping or faster delivery to reduce the impact of long warehouse distances.

✔️ 💰 Send a special cashback voucher as a "thank you for staying with us" to bring them back.

🟡 **For Cluster 1 - "Happy but quietly drifting away"**

✔️ Don't apologize - there's nothing wrong with their experience. Instead, **re-engage** them with time-limited flash sales or new product alerts. ⚡

✔️ 📋 Send a short survey to understand why they stopped buying - it may be a change in needs, not dissatisfaction.

🟢 **For Cluster 2 - "Loyal but under-rewarded"**

✔️ 💸 Increase cashback rates or send discount vouchers for their next purchase - they are currently getting the least value back.

✔️ 🏆 Create a **loyalty program** specifically for long-`Tenure` customers so they feel recognized for their commitment.

✔️ 🎯 Send personalized product recommendations to encourage more orders, since their `OrderCount` is the lowest.

---

## 🚀 How to Run This Project

**1. Clone the repository**
```bash
https://github.com/TascoGitGud/Customer-Churn-Prediction-and-Segmentation-for-an-E-commerce-Company.git
```

**2. Install the required libraries**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

**3. Open the notebook**
```bash
jupyter notebook Customer_Churn_Prediction.ipynb
```
