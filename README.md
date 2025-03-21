# Predicting Term Deposit Subscription 

## 📌 Project Description  
This project aims to predict whether a customer will subscribe to a term deposit using machine learning. It includes data analysis, preprocessing, model training, and evaluation of multiple classifiers such as Decision Trees, Logistic Regression, Ensemble Methods, and K-Nearest Neighbors (KNN). Key steps include handling class imbalance, feature engineering, hyperparameter tuning, and performance comparison using metrics like accuracy, F1-score, and AUC-ROC.

---

## 📂 Dataset Overview 
The dataset (`bank-full.csv`) contains **45,211 instances** and **17 features**, including demographic and banking attributes. The target variable `y` indicates whether a customer subscribed to a term deposit (`yes` or `no`). The dataset exhibits a class imbalance, with 88% of customers not subscribing and 12% subscribing.

---

### Key Features:
- **Numerical**: `age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous`  
- **Categorical**: `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`  
- **Target Variable**: `y` (binary: "yes" or "no")  

---

## 🛠 Libraries Used  
- **Data Handling**: `pandas`  
- **Visualization**: `seaborn`, `matplotlib`  
- **Preprocessing**: `StandardScaler`, `LabelEncoder`  
- **Models**: `DecisionTreeClassifier`, `RandomForestClassifier`, `KNeighborsClassifier`, `LogisticRegression`  
- **Utilities**: `train_test_split`, `cross_val_score`, `GridSearchCV`, `classification_report`, `roc_auc_score`  

---

## 📊 Exploratory Data Analysis (EDA)  

### 1. Target Variable Distribution  
#### Target Distribution  
- The target variable `y` is imbalanced, with most clients not subscribing to the term deposit.  

### 2. Numerical Feature Correlations  
#### Heatmap
- Key correlated features: `duration` (call duration) shows a moderate correlation with the target.  

### 3. Dataset Statistics  
- **Statistical Summary**:  
  ```python
  bank.describe()
  ```
  Includes metrics like mean, standard deviation, and quartiles for numerical features.  

---

## 🧠 Model Training  
### Steps:
1. **Preprocessing**:  
   - Handle missing values (if any).  
   - Encode categorical variables using `LabelEncoder`.  
   - Scale numerical features with `StandardScaler`.  

2. **Train-Test Split**:  
   ```python
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
   ```

3. **Model Selection**:  
   - Tested classifiers: Decision Trees, Random Forest, KNN, Logistic Regression.  
   - Hyperparameter tuning using `GridSearchCV`.  

4. **Evaluation**:  
   - Metrics: Accuracy, ROC-AUC, Confusion Matrix.  
   - Example output:  
     ```text
     Accuracy: 0.89
     ROC-AUC: 0.76
     ```

---

## 📈 Results  
- **Best Model**: Random Forest achieved the highest accuracy (89%) and AUC score (0.76).  
- **Key Insights**:  
  - Call duration (`duration`) is a critical predictor.  
  - Class imbalance may require techniques like SMOTE or class weighting.  

---

## 🚀 Possible Improvements  
1. **Handle Class Imbalance**: Use oversampling (SMOTE) or adjust class weights.  
2. **Feature Engineering**: Create interaction terms (e.g., `campaign` × `poutcome`).  
3. **Advanced Models**: Experiment with gradient boosting (XGBoost, LightGBM).  
4. **Explainability**: Use SHAP values to interpret model predictions.  

--- 

