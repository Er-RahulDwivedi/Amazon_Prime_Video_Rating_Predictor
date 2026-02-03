Amazon Prime Video: Content Analysis & Predictive Modeling
This project provides an end-to-end analytical and machine learning framework to evaluate content performance on Amazon Prime Video. By analyzing metadata from over 9,000 titles and 124,000 credits, the system identifies key success drivers and predicts audience reception with high accuracy.

📌 Project Overview
The streaming industry relies on data-driven content acquisition. This project transitions from intuitive decision-making to a quantitative framework, utilizing Exploratory Data Analysis (EDA) to uncover trends and Gradient Boosting models to predict if a title will be a "Hit" (High Rated) or "Average."

🚀 Key Features
Comprehensive EDA: Univariate, Bivariate, and Multivariate analysis of content types, release trends, and geographical production hubs.

Statistical Hypothesis Testing: Implementation of T-Tests and Chi-Square tests to validate the significance of features like age certification and format.

Advanced NLP Pipeline: Processing of textual descriptions using Lemmatization, POS Tagging, and TF-IDF Vectorization.

Dimensionality Reduction: Application of PCA to manage high-dimensional text data while retaining maximum variance.

Imbalanced Data Handling: Utilization of SMOTE to balance the target classes for more robust training.

Machine Learning: Comparative analysis of Random Forest, XGBoost, and LightGBM.

🛠️ Tech Stack
Language: Python

Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

NLP: NLTK, Scipy

Algorithms: LightGBM (Champion Model), Random Forest, XGBoost

Deployment: Joblib (Model Serialization)

📊 Results
The final model, built using LightGBM and optimized via GridSearchCV, achieved the following benchmarks:

Accuracy: 92%

F1-Score: 0.90

Primary Predictors: TMDB Popularity, Runtime, and Genre Diversity were identified as the most influential factors for high IMDb ratings.

📁 Repository Structure
AmazonPrime_EDA_Submission.ipynb: Detailed data exploration, visualization, and hypothesis testing.

AmazonPrime_ML_Submission.ipynb: Feature engineering, NLP pipeline, model training, and evaluation.

best_lgbm_model.pkl: The serialized production-ready model.

📥 How to Run
Clone the repository:

Bash
https://github.com/Er-RahulDwivedi/Amazon_Prime_Video_Rating_Predictor.git
Install dependencies:

Bash
pip install -r requirements.txt
Run the Jupyter Notebooks to view the full pipeline and results.

📝 Conclusion
This project demonstrates that machine learning can effectively mitigate the financial risks of content acquisition. By identifying "Hidden Gems" and optimizing runtimes, streaming platforms can maximize user retention and maintain a competitive edge in the global market.
