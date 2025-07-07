# Email-spam-detection-using-ml
📧 Email Spam Detection using Machine Learning
This project is designed to automatically classify emails as spam or not spam (ham) using NLP techniques and machine learning models.

🧠 Objective
To build a reliable spam filter that classifies text messages/emails using traditional ML algorithms and text vectorization.

📂 Dataset
Dataset: SMS Spam Collection Dataset

Total messages: ~5,500

Format:

label: "ham" or "spam"

message: The content of the message/email

🔧 Technologies & Tools
Python 🐍

Pandas & NumPy

NLTK for NLP preprocessing

Scikit-learn for ML models

Streamlit (for optional web app)

📊 Features Extracted
Lowercasing

Removing punctuation

Removing stop words

Stemming

TF-IDF Vectorization

📈 Machine Learning Models Used
Naive Bayes (Multinomial)

Logistic Regression

Support Vector Machine (SVM)

Random Forest

✅ Accuracy Results (example)
Model	Accuracy
Multinomial Naive Bayes	97%
Logistic Regression	95%
SVM	96%

🧪 How to Run the Project
Step 1: Clone the Repository
bash
Copy
Edit
git clone https://github.com/yourusername/spam-email-detector.git
cd spam-email-detector
Step 2: Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
Step 3: Run the Script or App
To run the notebook:

bash
Copy
Edit
jupyter notebook spam_classifier.ipynb
To run the Streamlit app:

bash
Copy
Edit
streamlit run app.py
📁 File Structure
kotlin
Copy
Edit
📦 spam-email-detector
├── data/
│   └── spam.csv
├── spam_classifier.ipynb
├── app.py
├── vectorizer.pkl
├── model.pkl
├── requirements.txt
└── README.md
💡 Streamlit App Demo (Optional)
Input an email or message

Click "Predict"

See if it’s Spam or Not Spam

🚀 Future Improvements
Use deep learning (e.g., LSTM)

Deploy using Flask/FastAPI

Collect real-time email data

Multi-language spam detection

📚 References
Scikit-learn documentation

NLTK documentation

Kaggle Dataset
