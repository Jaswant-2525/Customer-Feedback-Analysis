📘 **Customer Feedback Analysis using NMF and Sentiment Analysis**

Topic Modeling + Sentiment Classification on Customer Reviews

📌 **Project Overview**
1. This project performs automatic analysis of customer feedback using a combination of:
2. NLP preprocessing
3. TF-IDF vectorization
4. NMF (Non-negative Matrix Factorization) topic modeling
5. VADER sentiment analysis
6. Evaluation metrics (Reconstruction Error, Topic Coherence, Silhouette Score)
7. Visualizations such as topic distribution & word cloud.

The goal is to understand what customers are talking about and how they feel about different aspects of a product/service.

This project uses a CSV file containing customer review texts.

📂 **Dataset Description**

The dataset consists of a CSV file with customer reviews:

**Column	Description**

review_text	The customer review text (string)

**Example review entries:**

"The product quality is excellent and very durable."

"Delivery was late and the package was damaged."

"Amazing value for money! Totally worth the price."

🎯 **Objectives**
1. Identify key topics discussed in customer reviews
2. Determine sentiment (Positive / Negative / Neutral)
3. Cluster reviews based on dominant themes
4. Evaluate model performance using topic modeling metrics
5. Visualize topic–word distributions and sentiment insights

🧠 **Techniques & Algorithms Used**

**1. Text Preprocessing**
1. Lowercasing
2. Removing punctuation & special symbols
3. Removing extra spaces

**2. TF-IDF Vectorization**

Converts textual reviews into numerical feature vectors.

**3. NMF Topic Modeling**

Extracts hidden themes by decomposing the TF-IDF matrix into:

   W: Document–Topic matrix
   
   H: Topic–Word matrix

**Outputs:**
1. Top Words per Topic
2. Dominant Topic per Review

**4. Sentiment Analysis**

Uses VADER to classify each review as:
- Positive
- Negative
- Neutral

**5. Evaluation Metrics**
1. Reconstruction Error	: Measures NMF model fit
2. Topic Coherence (C_v) : Measures semantic quality of topics
3. Silhouette Score : Measures clustering separation

**6. Visualizations**
1. Topic distribution bar chart
2. Word clouds for each topic

🚀 **Project Workflow**
1. Load CSV containing customer reviews
2. Clean and preprocess the text
3. Convert text into TF-IDF vectors
4. Apply NMF to extract topics
5. Extract top words per topic
6. Determine dominant topic per review
7. Run sentiment analysis
8. Calculate evaluation metrics
9. Create visualizations
10. Save results to customer_feedback_nmf_results.csv

📊 **Sample Output**

**Top Words per Topic**
- Topic 1: delivery, package, late, damaged
- Topic 2: quality, excellent, durable, product
- Topic 3: service, wrong, disappointed, rude

**Dominant Topic Example**
Review: "Delivery was late and package was damaged."
→ Dominant Topic: 1 (Delivery Issues)
→ Sentiment: Negative


▶️ **How to Run the Project**

**1. Install dependencies**

pip install pandas numpy scikit-learn vaderSentiment gensim wordcloud matplotlib seaborn

**2. Run the Python script**
   
python customer_feedback_nmf_sentiment.py

**3. View the results**

Check the attached csv file.
