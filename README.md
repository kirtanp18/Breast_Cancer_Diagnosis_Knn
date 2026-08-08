![Breast Cancer Machine Learning Header](../Visuals/bim.png)

# 🩺 Breast Cancer Diagnosis: KNN From Scratch vs. Sklearn

> **A custom-built Machine Learning engine classifying breast tumor data (Benign vs. Malignant) using the K-Nearest Neighbors (KNN) algorithm from scratch.**

This project focuses on demystifying the "black box" of machine learning. By building the KNN engine entirely from scratch using fundamental Euclidean distance mathematics, this project proves a practical understanding of algorithm design before benchmarking it against the industry-standard `scikit-learn` implementation.

This is part of an ongoing initiative to build end-to-end machine learning applications through iterative development, pushing further into modular code architecture, object-oriented programming (OOP), and custom algorithm engineering.

---

## 🚀 Key Objectives

* **Algorithmic Transparency:** Build a custom KNN classification engine from scratch to understand distance metrics, sorting, and voting mechanisms without relying on external libraries.
* **Performance Benchmarking:** Compare the training and testing accuracy of the custom-built engine directly against Scikit-Learn's built-in `KNeighborsClassifier`.
* **Modular Engineering:** Engineer robust, reusable Python modules by separating data preprocessing, model architecture, and data visualization into distinct directories and classes.
* **Visual Evaluation:** Generate clean, interpretable confusion matrix heatmaps using Matplotlib to visually evaluate the models' true positive and false positive rates.

---

## 🛠 Tech Stack

* **Core Libraries:** Python, Pandas, NumPy
* **Machine Learning:** Scikit-learn (`MinMaxScaler`, `train_test_split`, `KNeighborsClassifier`, `metrics`)
* **Visualization:** Matplotlib
* **Architecture:** Object-Oriented Programming (OOP), Modular Python Scripts

---

## 📁 Folder Structure

```text
Breast_Cancer_Diagnosis/
│
├── Data/
│   └── breast_cancer.csv           # Raw dataset (cleaned of missing '?' values)
├── Model/
│   ├── __init__.py
│   └── knn_classification.py       # Custom KNN engine built from scratch
├── Preprocess/
│   ├── __init__.py
│   └── Preprocessing.py            # Data loading, cleaning, and MinMax scaling
├── Visualization/
│   ├── __init__.py
│   └── Visuals.py                  # Matplotlib heatmap generation logic
├── requirements.txt                # Project dependencies
├── main.py                         # Main execution script
└── README.md                       # This file
```

---

## 📊 Sample Insights & Results

### Confusion Matrix Insights

![KNN Prediction Heatmap](../Visuals/Figure_1.png)

Visual analysis of the generated heatmap revealed highly accurate classifications by the KNN model, demonstrating a strong ability to distinguish between tumor types:
* **High True Classification Rate:** The stark dark blue top-left quadrant (True Benign) and the solid medium blue bottom-right quadrant (True Malignant) indicate that the vast majority of both classes were predicted correctly.
* **Clinical Relevance:** In medical diagnostics, minimizing false negatives (predicting a tumor is benign when it is actually malignant) is paramount. The faintness of the bottom-left quadrant demonstrates the model's reliability in catching malignant cases.

### Model Performance on Test Data

The project successfully demonstrated that a custom-built mathematical engine can match the performance of enterprise-grade libraries.

| Model | Implementation Type | Distance Metric | Performance |
| :--- | :--- | :--- | :--- |
| **Custom KNN Engine** | Built from scratch (NumPy) | Squared Euclidean | *Matches Built-in* |
| **Scikit-Learn KNN** | Built-in (`KNeighborsClassifier`) | Minkowski (Standard) | *Baseline Target* |

---

## 📌 Key Learnings & Future Enhancements

* **Algorithm Under the Hood:** Successfully implemented KNN mathematics from scratch, proving that the underlying logic matches the accuracy of optimized, enterprise-grade libraries.
* **Modular Architecture:** Transitioned from notebook-style scripting to a professional, multi-file software engineering structure.
* **Next Steps:** Package this trained model and preprocessing pipeline into a simple API (e.g., using FastAPI or Flask) and deploy it as an interactive web app locally.

---

## 👨‍💻 About Me

I am a second-year Computer Science Engineering student specializing in backend development, and a self-taught AI enthusiast deep-diving into the mathematics of Machine Learning. Rather than relying on black-box libraries, I build algorithms entirely from scratch to master the core fundamentals. I blend formal Python software engineering with raw data science to build the future of tech—one model at a time.

🔗 **Connect with me on LinkedIn:** [Kirtan Pandya](https://www.linkedin.com/in/kirtan-pandya-2036233b2)