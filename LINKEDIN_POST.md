# LinkedIn post

I’m excited to share **SupportFlow AI**, my Machine Learning Task 2 project for
my internship with **Future Interns**.

Customer-support teams often spend valuable time manually reading, sorting and
prioritising incoming tickets. I built an end-to-end decision-support system
that automatically:

• classifies IT tickets into 8 operational categories  
• assigns the appropriate support team  
• predicts High, Medium or Low business priority  
• recommends a response-time target  
• sends uncertain predictions for human review

The category model combines word and character TF-IDF, supervised similarity,
LSA semantic features, a Linear SVM and a neural-model blend. On 9,567 untouched
test tickets, it achieved **87.60% overall accuracy**. With confidence-based
routing, it achieved **93.23% automatic-routing accuracy** at **86.06%
coverage**.

For priority prediction, I trained a leakage-safe Histogram Gradient Boosting
model using structured operational-impact features. On 7,500 untouched test
tickets, it achieved **97.45% overall accuracy** and **97.10% macro F1**. Its
selective workflow achieved **98.97% automatic-priority accuracy** at **95.35%
coverage**.

I also built a responsive React + Vite interface with a Flask API, animated
confidence visualisations and separate human-review controls for category and
priority decisions.

This project taught me that a useful ML system is more than a high score—it also
needs leakage-safe evaluation, uncertainty handling, explainable business
inputs and an interface people can confidently use.

Thank you **Future Interns** for the opportunity to work on a realistic NLP and
support-operations use case.

GitHub: [add your repository link]  
Demo: [add your demo link]

#MachineLearning #ArtificialIntelligence #NLP #ScikitLearn #ReactJS #Flask
#DataScience #MLOps #AIML #InternshipProject #FutureInterns #GitHubPortfolio
