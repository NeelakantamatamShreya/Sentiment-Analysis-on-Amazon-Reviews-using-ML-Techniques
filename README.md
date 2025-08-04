# Sentiment-Analysis-on-Amazon-Reviews-using-ML-Techniques
A machine learning project that classifies Amazon product reviews into positive , negative or neutral sentiment using various classification algorithms.

# 📊 Sentiment Analysis on Amazon Reviews using Machine Learning

This project performs sentiment classification on Amazon product reviews using various machine learning techniques. It explores both CountVectorizer and TF-IDF vectorization methods across multiple classifiers and identifies the most effective model through evaluation and fine-tuning.

---

## 📌 Abstract

Customer reviews are a rich source of feedback for businesses. This project aims to automatically classify Amazon reviews into **positive**, **negative**, or **neutral** sentiments using classical machine learning approaches. After experimenting with several models and vectorizers, the final result shows that **Nu-SVM** delivers the best performance, achieving an accuracy of **94%**.

---

## 🚀 Features

- Preprocessing of raw Amazon review data from Kaggle
- Feature extraction using CountVectorizer and TF-IDF Vectorizer
- Training four classifiers:
  - Naive Bayes
  - Logistic Regression
  - Random Forest
  - Support Vector Machine
- Hyperparameter tuning using `RandomizedSearchCV` and `Nu-SVM`
- Final result with **Nu-SVM (94% accuracy)**

---

## 🛠️ Tech Stack

- Python
- scikit-learn
- pandas, NumPy
- Matplotlib / Seaborn
- Jupyter Notebook / Google Colab

---

## 📊 Results & Model Comparison

### 🔹 CountVectorizer Results

| Algorithm          | Accuracy | Best Precision | Best Recall | Best F1-Score |
|-------------------|----------|----------------|-------------|---------------|
| Naive Bayes        | 92%      | Positive (0.92) | Positive (1.0) | Positive (0.96) |
| Logistic Regression| 90%      | Neutral (1.0)  | Positive (1.0) | Positive (0.94) |
| Random Forest      | 93%      | Neutral (1.0)  | Positive (1.0) | Positive (0.96) |
| SVM                | 93%      | Negative (0.96) | Positive (1.0) | Positive (0.96) |

### 🔹 TF-IDF Vectorizer Results

| Algorithm          | Accuracy | Best Precision | Best Recall | Best F1-Score |
|-------------------|----------|----------------|-------------|---------------|
| Naive Bayes        | 91%      | Positive (0.93) | Positive (0.98) | Positive (0.95) |
| Logistic Regression| 92%      | Positive (0.93) | Positive (0.99) | Positive (0.96) |
| Random Forest      | 93%      | Neutral (1.0)  | Positive (1.0) | Positive (0.96) |
| SVM                | 93%      | Positive (0.95) | Positive (0.98) | Positive (0.97) |

### 🔹 Fine-Tuned Models

| Algorithm                  | Accuracy | Precision (Positive) | Recall (Positive) | F1-Score (Positive) |
|---------------------------|----------|------------------------|--------------------|----------------------|
| Random Forest (Tuned)     | 90%      | 0.90                   | 1.0                | 0.95                 |
| **Nu-SVM (Best)**         | **94%**  | **0.94**               | **0.99**           | **0.97**             |

---


## 📂 Dataset

> The dataset used for this project is from Kaggle and is not included in this repository due to licensing.

📥 [Download from Kaggle](https://www.kaggle.com/datasets) (search "Amazon Reviews")  
After downloading, place the dataset in the `data/` folder.



---

## 📚 Publication

This project has been formally published at **Springer LNNS – ISBM 3rd World Conference (2024)**.

📖 [Springer Link to Publication](https://link.springer.com/chapter/10.1007/978-981-96-1747-0_46)

---

## 📌 Conclusion

- Both Random Forest and SVM performed well with TF-IDF (93% accuracy).
- Fine-tuning revealed Nu-SVM to be the most effective with **94% accuracy**.
- This demonstrates that **vectorization strategy + model tuning** has significant impact on sentiment analysis accuracy.

---

## 🔮 Future Scope

- Explore deep learning methods (e.g., LSTM, BERT)
- Aspect-based sentiment analysis
- Multilingual sentiment classification

---

## 🙌 Contributors

- Neelakantamatam Shreya  
- Indukuri Varsha  
- Gotumukkala Kavya

---

## 📬 Contact

📧 [shreyanm6@gmail.com](mailto:shreyanm6@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/shreya-nm-05n03/)


