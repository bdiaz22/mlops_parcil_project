import pandas as pd
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
import joblib  # Importar joblib

def train_model():
    mlflow.set_experiment("mlops_demo")
    with mlflow.start_run():
        # Cargar los datos
        data = pd.read_csv('data/processed/train.csv')

        # Verifica los tipos de datos
        print(data.dtypes)

        # Elimina o convierte las columnas no numéricas (si es necesario)
        data = pd.get_dummies(data, drop_first=True)

        # Elimina las columnas no numéricas (como un ID o cualquier columna no útil)
        # data = data.drop(columns=['id_column_name'])

        # Verifica los valores nulos y rellena si es necesario
        data = data.fillna(0)

        # Divide los datos en características (X) y la variable objetivo (y)
        X = data.drop('target', axis=1)
        y = data['target']

        # Entrena el modelo
        model = LogisticRegression(max_iter=1000)  # Aumenta max_iter si es necesario
        model.fit(X, y)

        # Guarda el modelo utilizando joblib
        joblib.dump(model, 'models/model.pkl')  # Guardar el modelo en el archivo 'models/model.pkl'
        
        # Registra el modelo en MLflow
        mlflow.sklearn.log_model(model, "model")
        print("Model trained, saved to 'models/model.pkl', and logged to MLflow.")

if __name__ == "__main__":
    train_model()
