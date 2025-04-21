import pandas as pd
import re
import joblib
import logging
import os
import hashlib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from nltk.stem import RSLPStemmer
from nltk.corpus import stopwords
import nltk
from datetime import datetime
import uuid
import secrets
import multiprocessing
from functools import partial
from sklearn.linear_model import SGDClassifier
from scipy.sparse import vstack
import threading
import queue
import time
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("spam_classifier.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("SpamClassifier")

class SpamClassifier:
    def __init__(self, model_dir="models", online_learning=True, feedback_collection=True):
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, f"spam_model_{datetime.now().strftime('%Y%m%d')}.pkl")
        self.pipeline = None
        self.stemmer = None
        self.online_learning = online_learning
        self.feedback_collection = feedback_collection
        self.feedback_queue = queue.Queue() if feedback_collection else None
        self.feedback_data = os.path.join(model_dir, "feedback_data.csv")
        self.online_update_lock = threading.Lock()
        self.update_threshold = 100  # Número de novas amostras antes da atualização do modelo completo
        self.last_full_update = datetime.now()
        self.full_update_interval = 3600 * 24  # Atualiza o modelo a cada 24 horas de forma completa
        self.tfidf_vectorizer = None  
        self.online_classifier = None  
        
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('rslp', quiet=True)
        self.stemmer = RSLPStemmer()
        
        if self.feedback_collection:
            self._init_feedback_storage()
            self._start_feedback_thread()
    
    def _init_feedback_storage(self):
        if not os.path.exists(self.feedback_data):
            pd.DataFrame(columns=['message', 'label', 'timestamp', 'confidence']).to_csv(
                self.feedback_data, index=False
            )
        
    def _start_feedback_thread(self):
        def process_feedback():
            while True:
                try:
                    try:
                        feedback_items = []
                        while True:
                            try:
                                feedback_items.append(self.feedback_queue.get_nowait())
                            except queue.Empty:
                                break
                        
                        if feedback_items:
                            with self.online_update_lock:
                                self._process_feedback_batch(feedback_items)
                            
                            self._save_feedback_data(feedback_items)
                            
                            # Função de verificação se é necessária uma atualização completa do modelo
                            current_time = datetime.now()
                            seconds_since_update = (current_time - self.last_full_update).total_seconds()
                            feedback_count = len(feedback_items)
                            
                            if feedback_count >= self.update_threshold or seconds_since_update >= self.full_update_interval:
                                logger.info(f"Triggering full model update with {feedback_count} new samples")
                                self._update_full_model()
                                self.last_full_update = current_time
                    except queue.Empty:
                        pass
                        
                    time.sleep(5)  
                except Exception as e:
                    logger.error(f"Error in feedback thread: {str(e)}", exc_info=True)
                    time.sleep(60)
        
        feedback_thread = threading.Thread(target=process_feedback, daemon=True)
        feedback_thread.start()
        logger.info("Feedback processing thread started")
    
    def _save_feedback_data(self, feedback_items):
        try:
            feedback_df = pd.DataFrame(feedback_items)
            if os.path.exists(self.feedback_data):
                feedback_df.to_csv(self.feedback_data, mode='a', header=False, index=False)
            else:
                feedback_df.to_csv(self.feedback_data, index=False)
        except Exception as e:
            logger.error(f"Failed to save feedback data: {str(e)}", exc_info=True)
    
    def _process_feedback_batch(self, feedback_items):
        if not self.pipeline or not self.online_learning:
            return
            
        try:
            messages = [item['message'] for item in feedback_items]
            labels = [1 if item['label'] == 'spam' else 0 for item in feedback_items]
            
            processed_messages = []
            for message in messages:
                processed = re.sub(r'[^\w\sáàâãéèêíïóôõöúçñ]', '', message.lower())
                if self.stemmer:
                    words = processed.split()
                    processed = ' '.join([self.stemmer.stem(word) for word in words if len(word) > 2])
                processed_messages.append(processed)
            
            X = self.tfidf_vectorizer.transform(processed_messages)
            y = np.array(labels)
            
            self.online_classifier.partial_fit(X, y, classes=[0, 1])
            
            logger.info(f"Online model updated with {len(feedback_items)} new samples")
        except Exception as e:
            logger.error(f"Error updating online model: {str(e)}", exc_info=True)
    
    def _update_full_model(self):
        try:
            if not os.path.exists(self.feedback_data) or os.path.getsize(self.feedback_data) == 0:
                logger.info("No feedback data available for model update")
                return
                
            feedback_df = pd.read_csv(self.feedback_data)
            if feedback_df.empty:
                return
                
            logger.info(f"Updating full model with {len(feedback_df)} samples")
            
            feedback_df = self._preprocess(feedback_df.rename(columns={'message': 'message', 'label': 'label'}))
            
            X = self.tfidf_vectorizer.transform(feedback_df['processed'])
            y = feedback_df['label'].map({'ham': 0, 'spam': 1}).values
            
            rf_classifier = self.pipeline.named_steps['clf']
            rf_classifier.fit(X, y)
            
            self.pipeline.named_steps['clf'] = rf_classifier
            
            model_hash = self._get_model_hash()
            timestamp = datetime.now().isoformat()
            
            metadata = {
                'timestamp': timestamp,
                'model_hash': model_hash,
                'training_samples': len(feedback_df),
                'update_type': 'incremental',
                'id': str(uuid.uuid4())
            }
            
            update_path = os.path.join(self.model_dir, f"spam_model_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
            joblib.dump((self.pipeline, metadata), update_path, compress=9)
            logger.info(f"Updated model saved to {update_path}")
            
            archive_path = os.path.join(self.model_dir, f"feedback_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            feedback_df.to_csv(archive_path, index=False)
            
            pd.DataFrame(columns=['message', 'label', 'timestamp', 'confidence']).to_csv(self.feedback_data, index=False)
            
        except Exception as e:
            logger.error(f"Failed to update full model: {str(e)}", exc_info=True)
    
    def add_feedback(self, message, label, confidence=1.0):
        if not self.feedback_collection:
            logger.warning("Feedback collection is disabled")
            return False
            
        if label not in ['spam', 'ham']:
            logger.error(f"Invalid label: {label}. Must be 'spam' or 'ham'")
            return False
            
        try:
            feedback_item = {
                'message': message,
                'label': label,
                'timestamp': datetime.now().isoformat(),
                'confidence': float(confidence)
            }
            
            self.feedback_queue.put(feedback_item)
            return True
        except Exception as e:
            logger.error(f"Failed to add feedback: {str(e)}")
            return False
    
    def _preprocess(self, df):
        df = df.copy()
        df['processed'] = df['message'].str.lower()
        df['processed'] = df['processed'].apply(lambda x: re.sub(r'[^\w\sáàâãéèêíïóôõöúçñ]', '', str(x)))
        df['processed'] = df['processed'].fillna('')
        
        cores = max(1, multiprocessing.cpu_count() - 1)
        with multiprocessing.Pool(cores) as pool:
            stem_func = partial(self._stem_text)
            df['processed'] = pool.map(stem_func, df['processed'])
        
        return df
    
    def _stem_text(self, text):
        words = text.split()
        stemmed_words = [self.stemmer.stem(word) for word in words if len(word) > 2]
        return ' '.join(stemmed_words)
    
    def _sanitize_input(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        if not file_path.endswith('.csv'):
            raise ValueError("Only CSV files are supported")
        
        abs_path = os.path.abspath(file_path)
        if not os.path.dirname(abs_path).startswith(os.path.abspath(os.getcwd())):
            raise ValueError("Access to files outside the working directory is not allowed")
        
        return abs_path
    
    def train(self, data_path, test_size=0.2, random_state=None):
        if random_state is None:
            random_state = secrets.randbelow(1000)
        
        try:
            sanitized_path = self._sanitize_input(data_path)
            start_time = datetime.now()
            logger.info(f"Started training at {start_time}")
            
            data = pd.read_csv(sanitized_path, encoding='latin-1')
            
            required_columns = ['label', 'message']
            if not all(col in data.columns for col in required_columns):
                logger.error("CSV file must contain 'label' and 'message' columns")
                raise ValueError("CSV file must contain 'label' and 'message' columns")
            
            data = data[required_columns]
            
            if data.isnull().values.any():
                logger.warning(f"Found {data.isnull().sum().sum()} null values in dataset, filling with empty strings")
                data = data.fillna('')
            
            unique_labels = data['label'].unique()
            if not all(label in ['ham', 'spam'] for label in unique_labels):
                logger.error(f"Invalid labels found: {unique_labels}. Only 'ham' and 'spam' are allowed.")
                raise ValueError(f"Invalid labels found: {unique_labels}. Only 'ham' and 'spam' are allowed.")
            
            data = self._preprocess(data)
            
            X_train, X_test, y_train, y_test = train_test_split(
                data['processed'], 
                data['label'].map({'ham': 0, 'spam': 1}),
                test_size=test_size,
                stratify=data['label'],
                random_state=random_state
            )
            
            logger.info(f"Training data size: {X_train.shape[0]}, Test data size: {X_test.shape[0]}")
            
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words=stopwords.words('portuguese'),
                token_pattern=r'(?u)\b[^\d\W][a-záàâãéèêíïóôõöúçñ]+\b',
                ngram_range=(1, 2),
                max_features=10000,
                min_df=2,
                max_df=0.95
            )
            
            self.online_classifier = SGDClassifier(
                loss='log_loss',
                penalty='l2',
                alpha=1e-5,
                random_state=random_state,
                max_iter=1000,
                warm_start=True,
                class_weight='balanced'
            )
            
            self.pipeline = Pipeline([
                ('tfidf', self.tfidf_vectorizer),
                ('clf', RandomForestClassifier(
                    n_estimators=200,
                    random_state=random_state,
                    n_jobs=-1,
                    class_weight='balanced'
                ))
            ])
            
            self.pipeline.fit(X_train, y_train)
            
            X_train_features = self.tfidf_vectorizer.transform(X_train)
            self.online_classifier.fit(X_train_features, y_train)
            
            y_pred = self.pipeline.predict(X_test)
            y_prob = self.pipeline.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            
            logger.info(f"Accuracy: {accuracy*100:.2f}%")
            logger.info(f"ROC AUC: {auc:.4f}")
            logger.info("Classification Report:\n%s", classification_report(y_test, y_pred))
            
            model_hash = self._get_model_hash()
            
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'accuracy': accuracy,
                'auc': auc,
                'model_hash': model_hash,
                'training_samples': X_train.shape[0],
                'online_learning': self.online_learning,
                'id': str(uuid.uuid4())
            }
            
            joblib.dump((self.pipeline, metadata, self.online_classifier), self.model_path, compress=9)
            logger.info(f"Model saved to {self.model_path}")
            logger.info(f"Training completed in {datetime.now() - start_time}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise
    
    def _get_model_hash(self):
        if not self.pipeline:
            return None
        model_bytes = joblib.dumps(self.pipeline)
        return hashlib.sha256(model_bytes).hexdigest()
    
    def load_model(self, model_path=None):
        try:
            if model_path is None:
                model_files = [f for f in os.listdir(self.model_dir) if f.startswith('spam_model_') and f.endswith('.pkl')]
                if not model_files:
                    raise FileNotFoundError("No model files found")
                model_path = os.path.join(self.model_dir, sorted(model_files)[-1])
            
            logger.info(f"Loading model from {model_path}")
            loaded_data = joblib.load(model_path)
            
            if isinstance(loaded_data, tuple) and len(loaded_data) >= 2:
                if len(loaded_data) == 3:
                    self.pipeline, metadata, self.online_classifier = loaded_data
                else:
                    self.pipeline, metadata = loaded_data
                    # Se não for encontrado no modelo salvo ele irá iniciar um novo
                    self.online_classifier = SGDClassifier(
                        loss='log_loss',
                        penalty='l2',
                        alpha=1e-5,
                        warm_start=True,
                        class_weight='balanced'
                    )
                
                self.tfidf_vectorizer = self.pipeline.named_steps['tfidf']
                
                logger.info(f"Model loaded: {metadata.get('id')}, accuracy: {metadata.get('accuracy', 'N/A')}")
                return metadata
            else:
                raise ValueError("Invalid model format")
                
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            raise
    
    def predict(self, message, threshold=0.5, collect_feedback=True):
        if not self.pipeline:
            try:
                self.load_model()
            except Exception as e:
                logger.error(f"No model available: {str(e)}")
                return {"error": "Model not loaded", "prediction": None, "probability": None}
        
        try:
            if not isinstance(message, str):
                message = str(message)
                
            processed = re.sub(r'[^\w\sáàâãéèêíïóôõöúçñ]', '', message.lower())
            
            if self.stemmer:
                words = processed.split()
                processed = ' '.join([self.stemmer.stem(word) for word in words if len(word) > 2])
            
            rf_probability = float(self.pipeline.predict_proba([processed])[0, 1])
            
            if self.online_learning and self.online_classifier is not None:
                # Transform the input using the same vectorizer
                X = self.tfidf_vectorizer.transform([processed])
                online_probability = float(self.online_classifier.predict_proba(X)[0, 1])
                
                combined_probability = 0.4 * rf_probability + 0.6 * online_probability
            else:
                combined_probability = rf_probability
            
            prediction = "spam" if combined_probability >= threshold else "ham"
            
            result = {
                "prediction": prediction,
                "probability": combined_probability,
                "threshold": threshold,
                "feedback_id": str(uuid.uuid4()) if collect_feedback else None
            }
            
            return result
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}", exc_info=True)
            return {"error": str(e), "prediction": None, "probability": None}
    
    def batch_predict(self, messages, threshold=0.5, collect_feedback=True):
        results = []
        for msg in messages:
            results.append(self.predict(msg, threshold, collect_feedback))
        return results
    
    def confirm_prediction(self, message, actual_label, feedback_id=None):
        if not self.feedback_collection:
            return False
            
        if actual_label not in ['spam', 'ham']:
            logger.error(f"Invalid label: {actual_label}. Must be 'spam' or 'ham'")
            return False
            
        try:
            
            return self.add_feedback(message, actual_label, 1.0)
        except Exception as e:
            logger.error(f"Failed to confirm prediction: {str(e)}")
            return False
    
    def get_model_info(self):
        if not self.pipeline:
            return {"status": "Not loaded"}
            
        try:
            online_samples = 0
            if os.path.exists(self.feedback_data):
                feedback_df = pd.read_csv(self.feedback_data)
                online_samples = len(feedback_df)
                
            return {
                "status": "Loaded",
                "online_learning": self.online_learning,
                "feedback_collection": self.feedback_collection,
                "pending_feedback_samples": online_samples,
                "last_full_update": self.last_full_update.isoformat(),
                "update_threshold": self.update_threshold,
                "rf_feature_importance": self.pipeline.named_steps['clf'].feature_importances_.tolist()[:10] if hasattr(self.pipeline.named_steps['clf'], 'feature_importances_') else None,
                "time_to_next_update": str(datetime.now() - self.last_full_update)
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {str(e)}")
            return {"status": "Error", "error": str(e)}


class SpamDetectionAPI:
    def __init__(self):
        self.classifier = SpamClassifier(online_learning=True, feedback_collection=True)
        
    def initialize(self, data_path=None, model_path=None):
        try:
            if model_path:
                self.classifier.load_model(model_path)
            elif data_path:
                self.classifier.train(data_path)
            else:
                try:
                    self.classifier.load_model()
                except FileNotFoundError:
                    logger.warning("No model found. Please provide a data_path to train a new model.")
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            raise
    
    def analyze_message(self, message, threshold=0.5):
        result = self.classifier.predict(message, threshold)
        return result
    
    def provide_feedback(self, message, correct_label):
        return self.classifier.confirm_prediction(message, correct_label)
    
    def get_status(self):
        return self.classifier.get_model_info()
    
    def analyze_batch(self, messages, threshold=0.5):
        return self.classifier.batch_predict(messages, threshold)


def evaluate_model(data_path, model=None, test_size=0.2):
    if model is None:
        model = SpamClassifier()
        model.train(data_path, test_size=test_size)
    else:
        model.load_model()
    
    try:
        data = pd.read_csv(data_path, encoding='latin-1')
        required_columns = ['label', 'message']
        data = data[required_columns]
        data = model._preprocess(data)
        
        X_train, X_test, y_train, y_test = train_test_split(
            data['processed'],
            data['label'].map({'ham': 0, 'spam': 1}),
            test_size=test_size,
            random_state=42
        )
        
        y_pred = np.array([1 if model.predict(msg)['prediction'] == 'spam' else 0 for msg in X_test])
        
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Evaluation Accuracy: {accuracy*100:.2f}%")
        logger.info("Evaluation Report:\n%s", classification_report(y_test, y_pred))
        
        return accuracy, classification_report(y_test, y_pred, output_dict=True)
    
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    api = SpamDetectionAPI()
    
    try:
        data_path = os.path.join(os.getcwd(), 'spam.csv')
        if os.path.exists(data_path):
            api.initialize(data_path=data_path)
        else:
            api.initialize()
            
        logger.info("Spam detection system initialized")
        logger.info(f"System status: {json.dumps(api.get_status(), indent=2)}")
            
        test_messages = [
            'Parabéns! Você ganhou R$1 milhão',
            'Oi, como vai você?',
            'Urgente! Ligue agora para nossa oferta especial',
            'Deixei minhas chaves na sua casa ontem',
            'Ganhe um iPhone grátis! Clique neste link'
        ]
        
        results = api.analyze_batch(test_messages)
        
        for msg, result in zip(test_messages, results):
            prediction = result["prediction"]
            probability = result.get("probability", 0)
            logger.info(f'"{msg}" -> {prediction} (Probability: {probability:.4f})')
            
            # Simulação para demonstração
            if "iPhone" in msg or "milhão" in msg or "Urgente" in msg:
                api.provide_feedback(msg, "spam")
                logger.info(f"Provided feedback: 'spam' for message '{msg}'")
            elif "chaves" in msg or "como vai" in msg:
                api.provide_feedback(msg, "ham")
                logger.info(f"Provided feedback: 'ham' for message '{msg}'")
                
        new_messages = [
            'Oferta imperdível! Últimas unidades!',
            'Me avise quando chegar em casa',
            'Não perca essa chance única! Promoção relâmpago!',
            'Vamos nos encontrar no restaurante às 8h',
            'Prêmio exclusivo aguardando seu clique!'
        ]
        
        logger.info("\nAnalyzing new messages after feedback:")
        new_results = api.analyze_batch(new_messages)
        
        for msg, result in zip(new_messages, new_results):
            prediction = result["prediction"]
            probability = result.get("probability", 0)
            logger.info(f'"{msg}" -> {prediction} (Probability: {probability:.4f})')
            
        # Show updated system status
        logger.info(f"\nUpdated system status: {json.dumps(api.get_status(), indent=2)}")
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
