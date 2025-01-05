import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import data_loader
from model_evaluation import evaluate_model  # Model değerlendirme fonksiyonu
from results_saver import save_results_to_csv, create_results_folder  # Sonuçları kaydetme

# Dosya yollarını belirleyelim
file_path_red = r'C:\Users\Alican\Desktop\Code\BLM5110-PROJE\data\winequality-red.csv'
file_path_white = r'C:\Users\Alican\Desktop\Code\BLM5110-PROJE\data\winequality-white.csv'

# Veri setlerini yükleyelim
datared = data_loader.load_data(file_path_red)
datawhite = data_loader.load_data(file_path_white)

# Kırmızı şarap için özellikler ve etiketler
X_red = datared.drop('quality', axis=1)
y_red = datared['quality']

# Beyaz şarap için özellikler ve etiketler
X_white = datawhite.drop('quality', axis=1)
y_white = datawhite['quality']

# Modelleri tanımlayalım
svm_model = SVC(probability=True)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
logreg_model = LogisticRegression(max_iter=500)

# Red Wine için modelleri değerlendirelim
X_train, X_test, y_train, y_test = train_test_split(X_red, y_red, test_size=0.3, random_state=42)
print("Red Wine için Modeller Eğitiliyor...")
folder_name = create_results_folder()  # Sonuçların kaydedileceği klasörü oluştur

models_red = []  # Red wine için yerel model listesi
models_red = evaluate_model('SVM', svm_model, X_train, X_test, y_train, y_test, X_red.columns, folder_name, 'Red', models_red)
models_red = evaluate_model('Random Forest', rf_model, X_train, X_test, y_train, y_test, X_red.columns, folder_name, 'Red', models_red)
models_red = evaluate_model('Logistic Regression', logreg_model, X_train, X_test, y_train, y_test, X_red.columns, folder_name, 'Red', models_red)

# White Wine için modelleri değerlendirelim
X_train, X_test, y_train, y_test = train_test_split(X_white, y_white, test_size=0.3, random_state=42)
print("White Wine için Modeller Eğitiliyor...")

models_white = []  # White wine için yerel model listesi
models_white = evaluate_model('SVM', svm_model, X_train, X_test, y_train, y_test, X_white.columns, folder_name, 'White', models_white)
models_white = evaluate_model('Random Forest', rf_model, X_train, X_test, y_train, y_test, X_white.columns, folder_name, 'White', models_white)
models_white = evaluate_model('Logistic Regression', logreg_model, X_train, X_test, y_train, y_test, X_white.columns, folder_name, 'White', models_white)

# Sonuçları CSV dosyasına kaydedelim
all_models = models_red + models_white  # Her iki wine türünün sonuçlarını birleştiriyoruz
save_results_to_csv(all_models, folder_name)
