
# 🧠 Sentiment Analysis on Amazon Reviews using Machine Learning

A machine learning project that classifies Amazon product reviews into **positive**, **negative**, or **neutral** sentiments using classical ML algorithms and feature extraction techniques.

---

## 📖 Abstract

In today’s digital era, customer feedback is a critical source of business intelligence. This project aims to **automate sentiment analysis** of Amazon product reviews using **machine learning techniques**.
After experimenting with multiple classifiers and vectorization methods, the **Nu-SVM** model achieved the **highest accuracy of 94%**, proving to be the most effective for sentiment prediction.
This analysis provides valuable insights into customer opinions, helping e-commerce platforms and sellers make **data-driven decisions**.

---

## 🚀 Features

* ✅ Data preprocessing and cleaning of raw Amazon review data
* 🧩 Feature extraction using **CountVectorizer** and **TF-IDF Vectorizer**
* ⚙️ Model training using multiple ML algorithms:

  * Naive Bayes
  * Logistic Regression
  * Random Forest
  * Support Vector Machine (SVM)
* 🎯 Hyperparameter tuning using **RandomizedSearchCV** and **Nu-SVM**
* 📈 Model evaluation using accuracy, precision, recall, and F1-score
* 📊 Visualization with confusion matrices and sentiment distribution charts

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Libraries:** scikit-learn, pandas, NumPy, Matplotlib, Seaborn
* **Environment:** Jupyter Notebook / Google Colab

---

## 📂 Dataset

The dataset was sourced from **[Kaggle](https://www.kaggle.com/)** under the search term *“Amazon Reviews.”*
Due to licensing restrictions, the dataset is **not included** in this repository.

After downloading, place it inside the `data/` folder as:

```
data/amazon_reviews.csv
```

---

## 🧩 Methodology

1. **Data Collection:** Downloaded the Amazon reviews dataset from Kaggle.
2. **Preprocessing:**

   * Handled missing values.
   * Combined `review_title` and `review_text` into one column.
   * Labeled sentiments based on ratings:

     * ⭐ 4–5 → Positive
     * ⭐ 3 → Neutral
     * ⭐ 1–2 → Negative
3. **Feature Extraction:** Applied CountVectorizer and TF-IDF to convert text into numerical form.
4. **Model Training:** Trained multiple classifiers — Naive Bayes, Logistic Regression, Random Forest, and SVM.
5. **Evaluation:** Compared all models using accuracy, precision, recall, and F1-score.
6. **Fine-Tuning:** Optimized Random Forest using RandomizedSearchCV and SVM with Nu-SVM.
7. **Visualization:** Plotted sentiment distribution and confusion matrices for analysis.

---

## 📊 Results & Model Comparison

### 🔹 CountVectorizer Results

| Algorithm           | Accuracy | Best Precision  | Best Recall    | Best F1-Score   |
| ------------------- | -------- | --------------- | -------------- | --------------- |
| Naive Bayes         | 92%      | Positive (0.92) | Positive (1.0) | Positive (0.96) |
| Logistic Regression | 90%      | Neutral (1.0)   | Positive (1.0) | Positive (0.94) |
| **Random Forest**   | **93%**  | Neutral (1.0)   | Positive (1.0) | Positive (0.96) |
| **SVM**             | **93%**  | Negative (0.96) | Positive (1.0) | Positive (0.96) |

> ✅ **Observation:** Both **Random Forest** and **SVM** achieved the **highest accuracy of 93%** using the **CountVectorizer** approach, outperforming Naive Bayes and Logistic Regression.

---

### 🔹 TF-IDF Vectorizer Results

| Algorithm           | Accuracy | Best Precision  | Best Recall     | Best F1-Score   |
| ------------------- | -------- | --------------- | --------------- | --------------- |
| Naive Bayes         | 91%      | Positive (0.93) | Positive (0.98) | Positive (0.95) |
| Logistic Regression | 92%      | Positive (0.93) | Positive (0.99) | Positive (0.96) |
| **Random Forest**   | **93%**  | Neutral (1.0)   | Positive (1.0)  | Positive (0.96) |
| **SVM**             | **93%**  | Positive (0.95) | Positive (0.98) | Positive (0.97) |

> ✅ **Observation:** Similarly, both **Random Forest** and **SVM** maintained the **same top accuracy (93%)** with the **TF-IDF Vectorizer**, showing consistent performance across feature extraction methods.

---

### 🔹 Fine-Tuned Models

| Algorithm             | Accuracy | Precision (Positive) | Recall (Positive) | F1-Score (Positive) |
| --------------------- | -------- | -------------------- | ----------------- | ------------------- |
| Random Forest (Tuned) | 90%      | 0.90                 | 1.0               | 0.95                |
| 🏆 **Nu-SVM (Best)**  | **94%**  | **0.94**             | **0.99**          | **0.97**            |

> 🧠 **Conclusion:** Both **Random Forest** and **SVM** performed equally well with 93% accuracy on both vectorizers, but after fine-tuning, **Nu-SVM** achieved the **highest accuracy of 94%**, making it the most effective model overall.

---

## 📚 Publication

This project was **accepted for publication** at the
🎓 *3rd World Conference on Information Systems for Business Management (ISBM)* — **Springer LNNS Series (2024)**.

📖 [Springer Publication Link — Coming Soon]()

---

## 📌 Conclusion

* Both **Random Forest** and **SVM** achieved **93% accuracy** across both CountVectorizer and TF-IDF vectorization methods.
* Fine-tuning with **Nu-SVM** improved performance to **94%**, confirming it as the most efficient model.
* This project demonstrates how **vectorization strategies** and **hyperparameter tuning** significantly influence model performance in sentiment analysis.

---

## 🔮 Future Scope

* 🤖 Integrate **deep learning models** like **LSTM** and **BERT** for contextual understanding.
* 🌍 Expand the system for **multilingual** sentiment classification.
* 🧭 Develop an **aspect-based sentiment analyzer** for more granular insights.
* 💻 Build a **web interface or dashboard** for real-time sentiment tracking and visualization.

---

## 🙌 Contributors

👩‍💻 **Neelakantamatam Shreya**
👩‍💻 **Indukuri Varsha**
👩‍💻 **Gottumukkala Kavya**

---

## 📬 Contact

📧 **Email:** [shreyanm6@gmail.com](mailto:shreyanm6@gmail.com)
🔗 **LinkedIn:** [linkedin.com/in/neelakantamatam-shreya](https://linkedin.com/in/neelakantamatam-shreya)

---

### ⭐ Don’t forget to star this repository if you found it helpful!

---

Would you like me to make it even more **visually appealing** for GitHub (with shields/badges for Python, scikit-learn, license, etc.)? That helps your profile stand out even more to recruiters.


