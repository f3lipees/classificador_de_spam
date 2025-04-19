import pandas as pd
import re
import joblib
import logging
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords

logging.basicConfig(level=logging.INFO)

class SpamClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=10000,
                stop_words=stopwords.words('portuguese'),
                token_pattern=r'(?u)\b[^\d\W][a-záàâãéèêíïóôõöúçñ]+\b'
            )),
            ('clf', SGDClassifier(
                loss='log_loss',
                penalty='l2',
                alpha=1e-5,
                random_state=42,
                max_iter=1000
            ))
        ])

    def _preprocess(self, df):
        df['processed'] = df['message'].str.lower()
        df['processed'] = df['processed'].apply(lambda x: re.sub(r'[^\w\sáàâãéèêíïóôõöúçñ]', '', x))
        return df

    def train(self, data_path):
        try:
            data = pd.read_csv(data_path, encoding='latin-1', usecols=[0, 1])
            data.columns = ['label', 'message']
            data = self._preprocess(data)
            
            X_train, X_test, y_train, y_test = train_test_split(
                data['processed'], 
                data['label'].map({'ham': 0, 'spam': 1}),
                test_size=0.2,
                stratify=data['label'],
                random_state=42
            )
            
            self.pipeline.fit(X_train, y_train)
            y_pred = self.pipeline.predict(X_test)
            
            logging.info(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
            logging.info("Classification Report:\n%s", classification_report(y_test, y_pred))
            
            joblib.dump(self.pipeline, 'spam_model.pkl', compress=9)
            
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            raise

    def predict(self, message):
        try:
            processed = re.sub(r'[^\w\sáàâãéèêíïóôõöúçñ]', '', message.lower())
            return "Spam" if self.pipeline.predict([processed])[0] else "Ham"
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return "Error"

if __name__ == "__main__":
    classifier = SpamClassifier()
    classifier.train('C:\\spam.csv')
    
    test_messages = [
        'Parabéns! Você ganhou R$1 milhão',
        'Oi, como vai você?',
        'Urgente! Ligue agora para nossa oferta especial',
        'Deixei minhas chaves na sua casa ontem',
        'Ganhe um iPhone grátis! Clique neste link'
    ]
    
    for msg in test_messages:
        logging.info(f'"{msg}" -> {classifier.predict(msg)}')
