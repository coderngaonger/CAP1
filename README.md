**Sentiment & Aspect-Based Analysis on Hasaki Customer Reviews**

**📌 Overview**

This project is a comprehensive application of NLP, machine learning, and data visualization to analyze customer sentiments and specific aspects in product reviews from the e-commerce cosmetics website Hasaki.vn.

**🎯 Objectives**
Classify sentiment polarity (positive/neutral/negative) of customer comments.

Identify review aspects (Service, Price, Packaging, etc.).

Visualize insights via dashboards (Power BI & Tableau).

Build a web-based system that automatically crawls, analyzes, and visualizes review data.

**Project Pipeline**

_1. Data Collection_

Crawled over 10,000 customer reviews from Hasaki.vn using Selenium.

Stored structured data (product info, comments, ratings, timestamps) in Microsoft SQL Server.

_2. Preprocessing_

Lowercasing, emoji-to-text conversion.

Punctuation & special character removal.

Word segmentation and stopword removal.

_3. Labeling_

Used Gemini API & manual methods to tag comments with corresponding aspects:

Service, Price, Packaging, Store, Others

Mapped comments to sentiment labels.

_4. Feature Engineering_

Word2Vec embeddings using Skip-gram and CBOW.

Sentence-level vector aggregation.

_5. Model Training_

Algorithms: SVM, Logistic Regression, Random Forest, Naive Bayes, and Neural Network.

Evaluation using K-Fold Cross Validation, Confusion Matrix, Accuracy, Classification Report.
_6. Best Model by Aspect_

Aspect	Best Model	Accuracy
Service	Neural Network	92.3%
Others	Random Forest	88.23%
Store	Random Forest	90.1%
Packaging	Random Forest	89.72%
Price	Neural Network	85.4%

**📊 Data Visualization**

Power BI: Interactive dashboards for KPI tracking, aspect analysis.

Tableau: Visual insights by sentiment trends and aspect frequency.

**🌐 Website Demo**

Web app features:

Paste Hasaki product URL.

Automatically crawls and preprocesses review data.

Applies trained models for sentiment & aspect analysis.

Displays charts & summaries of findings.

**🛠️ Tech Stack**

Language: Python

Libraries: Selenium, Pandas, NumPy, Scikit-learn, TensorFlow/Keras, WordCloud

DBMS: Microsoft SQL Server

Visualization: Power BI, Tableau

Web: Flask / (or other framework depending on final deployment)
