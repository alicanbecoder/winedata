import time
from sklearn.metrics import accuracy_score, f1_score
import joblib

def evaluate_model(model, X_test, y_test):
    """
    Modelin performansını değerlendirmek için doğruluk ve F1 skoru hesaplar.
    """
    start_time = time.time()
    predictions = model.predict(X_test)
    end_time = time.time()
    
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')
    duration = end_time - start_time
    
    return accuracy, f1, duration

def evaluate_models_performance(X_train, X_test, y_train, y_test):
    """
    Modellerin performansını değerlendirir ve sürelerini kaydeder.
    """
    # Eğitimli modelleri yükleme
    log_reg = joblib.load('logistic_regression_model.pkl')
    rf_clf = joblib.load('random_forest_model.pkl')
    
    # Model Performansını Değerlendirme
    log_reg_accuracy, log_reg_f1, log_reg_eval_duration = evaluate_model(log_reg, X_test, y_test)
    rf_clf_accuracy, rf_clf_f1, rf_clf_eval_duration = evaluate_model(rf_clf, X_test, y_test)
    
    # Performans Sonuçlarını Yazdırma
    print(f"Logistic Regression - Accuracy: {log_reg_accuracy}, F1 Score: {log_reg_f1}, Evaluation Duration: {log_reg_eval_duration}")
    print(f"Random Forest - Accuracy: {rf_clf_accuracy}, F1 Score: {rf_clf_f1}, Evaluation Duration: {rf_clf_eval_duration}")
    
    return {
        "logistic_regression": {
            "accuracy": log_reg_accuracy,
            "f1_score": log_reg_f1,
            "evaluation_duration": log_reg_eval_duration
        },
        "random_forest": {
            "accuracy": rf_clf_accuracy,
            "f1_score": rf_clf_f1,
            "evaluation_duration": rf_clf_eval_duration
        }
    }

if __name__ == "__main__":
    # Örnek kullanım
    from model_training import train_models
    file_path = r'C:\Users\Alican\Desktop\Code\BLM5110-PROJE\data\winequality-red.csv'
    data = data_loader.load_data(file_path)
    X_train, X_test, y_train, y_test = train_models(data)
    
    # Modellerin performansını değerlendirme
    results = evaluate_models_performance(X_train, X_test, y_train, y_test)
    print(results)