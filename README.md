# customer-segmentation-app
AI-Powered Customer Segmentation Dashboard using K-Means Clustering and Streamlit. Built as an enterprise analytics tool during my tech industry internship.
# 📊 Enterprise AI Customer Segmentation Dashboard

An interactive, production-ready machine learning web application built to analyze retail customer behavior using unsupervised learning. This engine segments customers into distinct behavioral groups to automate data-driven marketing strategies for enterprise corporate teams.

---

## 🚀 Live Visuals & Overview
This system takes raw transactional or demographic customer data, performs automatic preprocessing, scales features equally, and applies the **K-Means Clustering Algorithm**. The output is mapped onto a beautiful, responsive 3-tab layout via **Streamlit**, helping non-technical executives make direct marketing decisions.

---

## 🛠️ Technology Stack Used
* **Language:** Python 3.10+
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning Engine:** Scikit-Learn (K-Means, StandardScaler)
* **Statistical Visualization:** Matplotlib, Seaborn
* **Deployment Interface:** Streamlit Engine

---

## 🔍 Architecture & Mathematical Workflow

### 1. Data Cleaning & Diagnostics
* Handles missing/null records automatically using column **median imputation** to ensure the clustering pipeline never crashes.
* Features analyzed: `Annual Income (k$)` and `Spending Score (1-100)`.

### 2. Feature Standardization (Scaling)
Since `Annual Income` ranges in thousands and `Spending Score` ranges from 1-100, the algorithm utilizes **`StandardScaler`** to transform the features to a common scale (0 to 1). This prevents distance calculation biases during centroid adjustments.

### 3. Optimal Cluster Selection (The Elbow Method)
The hyperparameter $K$ was determined by plotting the **Within-Cluster Sum of Squares (WCSS)**. The maximum structural variance break point was identified at **$K = 5$**, providing the perfect trade-off between mathematical variance and business execution feasibility.

---

## 🎯 Executive Strategy Matrix

The application dynamically breaks down the 5 clusters into actionable corporate marketing directives:

| Cluster Icon | Cluster Name | Spending vs Income Profile | Business Strategic Action Plan |
| :---: | :--- | :--- | :--- |
| 💎 | **VIP Customers** | High Income + High Spending | **Retention:** Premium loyalty clubs, exclusive early previews of luxury arrivals, and dedicated support. |
| 💰 | **High Potential** | High Income + Low Spending | **Activation:** Targeted product awareness campaigns and personalized premium catalogs. |
| 📊 | **Standard Shoppers**| Average Income + Average Spending| **Stability:** Seasonal value packs, generic store-wide coupons, and standard newsletters. |
| ⚡ | **Impulsive Buyers** | Low Income + High Spending | **Conversion:** Real-time push notifications for lightning flash sales and BOGO deals. |
| 📉 | **Budget Shoppers** | Low Income + Low Spending  | **Optimization:** Low-cost targeted ads focusing strictly on baseline utility item bundles. |

---

## ⚙️ Installation & Local Setup

Follow these simple steps to run this enterprise dashboard on your local machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/customer-segmentation-app.git](https://github.com/YOUR_USERNAME/customer-segmentation-app.git)
   cd customer-segmentation-app
