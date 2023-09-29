# Sentiment-Analysis-for-Predicting-Suicidal-Intent

INTRODUCTION

The prevalence of mental health challenges and the increasing use of social media platforms to express emotions have raised concerns about identifying individuals at risk of suicide. This project aims to address this issue by developing a text classifier for detecting suicidal intent in posts from social media platforms. The project focuses on utilizing the Naive Bayes theorem to perform sentiment analysis on text data collected from "SuicideWatch" and "depression" subreddits on Reddit.
Objective of this project
This project is to design a text classifier capable of predicting suicidal intent in social media posts. By analyzing the language patterns and sentiments expressed in the posts, the classifier will aid in early detection of individuals at risk, enabling timely intervention and support.
Hypothesis
Naive Bayes theorem for sentiment analysis will allow us to accurately identify posts containing suicidal ideation based on the emotional cues and language patterns present in the text.
Review of Literature
The Prior research has explored the use of sentiment analysis in various domains, including mental health. Studies have demonstrated the effectiveness of natural language processing techniques and machine learning algorithms in detecting suicidal intent in social media posts. The project builds upon this existing knowledge to develop a robust and efficient text classifier.


THEORY RELATED TO PROJECT

Sentiment Analysis
Sentiment analysis, also known as opinion mining, is a natural language processing (NLP) technique aimed at determining the sentiment or emotional tone conveyed by a piece of text. The primary goal is to identify whether the text expresses positive, negative, or neutral emotions. Sentiment analysis finds extensive applications in understanding customer feedback, social media monitoring, market research, and, in our case, mental health analysis.

Naive Bayes Theorem
Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem, which provides a way to update probabilities based on new evidence. The "naive" assumption underlying the algorithm is that each feature in the dataset is independent of the others, given the class label. Despite this simplifying assumption, Naive Bayes has proven to be remarkably effective in text classification tasks.

Bayes' Theorem
Bayes' theorem calculates the probability of an event based on prior knowledge. It can be represented as:
P(A|B) = (P(B|A) * P(A)) / P(B)
where:
P(A|B) is the probability of event A given event B,
P(B|A) is the probability of event B given event A,
P(A) is the prior probability of event A,
P(B) is the prior probability of event B.
Applying Naive Bayes to Text Classification
In the context of text classification, we aim to determine the probability that a given document belongs to a particular class (e.g., suicidal or non-suicidal). Naive Bayes assumes that the features (words or tokens) in the document are conditionally independent,

given the class label. This allows us to calculate the posterior probability of a class given the features in the document:
P(Class | Features) ∝ P(Features | Class) * P(Class)
By comparing the posterior probabilities for each class, we can assign the document to the class with the highest probability, making it a powerful algorithm for sentiment analysis and text classification tasks.

Other Essential Concepts

Data Preprocessing
Data preprocessing plays a critical role in preparing the textual data for analysis. Techniques such as lowercasing, punctuation removal, stop word removal, tokenization, and stemming are commonly used to clean and normalize the text.\

TF-IDF Vectorizer
The Term Frequency-Inverse Document Frequency (TF-IDF) is a numerical representation of the importance of each word in a document relative to a collection of documents. It assigns higher weights to words that are more relevant to a specific document while downgrading commonly occurring words.
Model Selection and Evaluation Metrics
To assess the performance of the text classifier, we use evaluation metrics such as accuracy, precision, recall, and F1-score. These metrics provide insights into the model's ability to correctly classify suicidal and non-suicidal posts.

DESIGN

3.1 Introduction
The design and simulation process involved in developing the Sentiment Analysis for Predicting Suicidal Intent text classifier. We describe the software used, data preprocessing techniques, and the steps taken to optimize the model's parameters for efficient and accurate classification. Additionally, we discuss the simulation process and the selection of evaluation metrics to assess the performance of the classifier.

3.2 Software Description
For this project, we utilized Python as the primary programming language due to its extensive libraries and tools for natural language processing and machine learning. The main libraries used include:

 Pandas: Used for data manipulation and preprocessing, Pandas allowed us to efficiently handle the large dataset containing posts from "SuicideWatch" and "depression" subreddits.
 Scikit-learn: This powerful machine learning library provided various algorithms for text classification, including Naive Bayes, Random Forest, and Decision Tree.
 Matplotlib and Seaborn: These visualization libraries allowed us to create plots and visualize the performance metrics, confusion matrices, and other relevant data.

3.3 Data Preprocessing
The dataset obtained from the "SuicideWatch" and "depression" subreddits contains over 200,000 rows, making it computationally expensive for training and testing various models. To reduce complexity and computational time while maintaining data representativeness, we randomly sampled 10,000 data points for our analysis.
Data preprocessing was essential to transform the raw text data into a suitable format for sentiment analysis. We performed the following preprocessing steps:
 Lowercasing: All text was converted to lowercase to ensure consistency and avoid the distinction between uppercase and lowercase words.
 Punctuation Removal: Punctuation marks were removed as they do not contribute much semantic value to the text and can interfere with downstream NLP tasks.
 Stopword Removal: Commonly occurring stopwords such as "the," "and," "a," and "in" were removed, as they have little semantic value and can be safely excluded from analysis.
 Tokenization: The text was tokenized, breaking it into individual words or tokens, to prepare it for further analysis.
 Stemming: The words were reduced to their base or root form using stemming, which helps in reducing the vocabulary size and simplifying the text representation.

3.4 Machine Learning - Model Selection
We employed various machine learning models to build the text classifier for predicting suicidal intent:

 Naive Bayes (Voting Classifier): As the core of our analysis, we used the Naive Bayes theorem for sentiment analysis. We utilized the Voting Classifier, which combines multiple Naive Bayes classifiers to improve performance.

 Random Forest: We employed the Random Forest algorithm, an ensemble learning technique, to assess its performance in comparison to the Naive Bayes classifier.
 Decision Tree: Decision Trees were also utilized as another baseline model for comparison.

3.5 TF-IDF Vectorizer
To represent the text data numerically, we used the TF-IDF Vectorizer, which calculates the importance of each word in a document relative to the entire collection of documents. This vectorization technique allowed us to transform the textual data into a format suitable for machine learning models.

3.6 Simulation and Evaluation
We split the preprocessed data into training and testing sets to simulate the training and validation process of the text classifier. We trained each model using the training set and evaluated their performance on the testing set.
For evaluation, we used various metrics, including accuracy, precision, recall, and F1-score, to assess the classifier's ability to correctly classify posts as suicidal or non-suicidal.

RESULT & DISCUSSION

Discussion
From the above we can say that out of all models,
Naive Bayes (Voting Classifier) is best fit model for the dataset.
Training score: 0.899271324474925
Testing score: 0.8753333333333333

SUMMARY & CONCLUSION

Summary
The Sentiment Analysis for Predicting Suicidal Intent project aimed to develop a text classifier capable of detecting suicidal intent in social media posts from the "SuicideWatch" and "depression" subreddits on Reddit. The project leveraged natural language processing techniques and machine learning algorithms to analyze textual data and predict whether a post indicates suicidal thoughts or not. This chapter provides a concise summary of the project's key objectives, methodology, findings, and contributions.
The project began with data collection using the Pushshift API, resulting in a dataset of over 2 lakh rows. To optimize computational resources, a random sample of 10,000 data points was selected for analysis. The data underwent preprocessing, including lowercasing, punctuation removal, stopword removal, tokenization, and stemming, to prepare it for sentiment analysis.
Three machine learning models, namely Naive Bayes (Voting Classifier), Random Forest, and Decision Tree, were evaluated for their performance in classifying posts into suicidal and non-suicidal categories. The Naive Bayes (Voting Classifier) demonstrated superior performance, achieving high precision, recall, F1-score, and accuracy on the testing set.
The practical implications of the text classifier's performance in early suicide detection and mental health support on social media platforms were discussed. Additionally, ethical considerations related to working with sensitive mental health-related data were emphasized.
Conclusion
The Sentiment Analysis for Predicting Suicidal Intent project represents a significant step towards addressing mental health challenges on social media. The development of an efficient text classifier using the Naive Bayes (Voting Classifier) model showcases the potential of machine learning in identifying individuals at risk of suicide and providing timely support.
The project's contributions lie in its application of natural language processing and machine learning techniques to analyze textual data from social media platforms. By predicting suicidal intent, the text classifier can assist in early intervention and facilitate mental health support for individuals in need.
However, it is essential to recognize the limitations of the project, including the sensitivity of working with mental health-related data and the potential for false positives and negatives in the classification process. Ethical considerations must be prioritized to protect user privacy and handle the data responsibly.
36
As future work, the project could be extended to explore advanced natural language processing techniques, such as deep learning models, to enhance the classifier's performance further. Additionally, a more extensive and diverse dataset could be utilized to improve the generalizability of the model.
Overall, the Sentiment Analysis for Predicting Suicidal Intent project underscores the significance of digital mental health research and its potential impact on suicide prevention efforts.
