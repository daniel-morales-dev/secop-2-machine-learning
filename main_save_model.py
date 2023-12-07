import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def entrenar_modelo(df, entrenar=True):
    # Seleccionamos las columnas que vamos a usar para predecir
    columns = ['Entidad', 'Departamento Entidad', 'Ciudad Entidad', 'OrdenEntidad', 'Entidad Centralizada', 'Fase', 'Precio Base', 'Modalidad de Contratacion', 'Duracion', 'Unidad de Duracion', 'Tipo de Contrato', 'Subtipo de Contrato']

    # Preprocesamiento de los datos
    # Convertimos las columnas categóricas a valores numéricos
    le = LabelEncoder()

    le_adjudicado = LabelEncoder()
    df['Adjudicado'] = le_adjudicado.fit_transform(df['Adjudicado'])

    le_entidad = LabelEncoder()
    df['Entidad'] = le_entidad.fit_transform(df['Entidad'])

    le_departamento = LabelEncoder()
    df['Departamento Entidad'] = le_departamento.fit_transform(df['Departamento Entidad'])

    le_entidad = LabelEncoder()
    df['Entidad'] = le_entidad.fit_transform(df['Entidad'])

    le_departamento = LabelEncoder()
    df['Departamento Entidad'] = le_departamento.fit_transform(df['Departamento Entidad'])

    le_ciudad = LabelEncoder()
    df['Ciudad Entidad'] = le_ciudad.fit_transform(df['Ciudad Entidad'])

    le_orden = LabelEncoder()
    df['OrdenEntidad'] = le_orden.fit_transform(df['OrdenEntidad'])

    le_centralizada = LabelEncoder()
    df['Entidad Centralizada'] = le_centralizada.fit_transform(df['Entidad Centralizada'])

    le_fase = LabelEncoder()
    df['Fase'] = le_fase.fit_transform(df['Fase'])

    le_modalidad = LabelEncoder()
    df['Modalidad de Contratacion'] = le_modalidad.fit_transform(df['Modalidad de Contratacion'])

    le_duracion = LabelEncoder()
    df['Unidad de Duracion'] = le_duracion.fit_transform(df['Unidad de Duracion'])

    le_tipo = LabelEncoder()
    df['Tipo de Contrato'] = le_tipo.fit_transform(df['Tipo de Contrato'])

    le_subtipo = LabelEncoder()
    df['Subtipo de Contrato'] = le_subtipo.fit_transform(df['Subtipo de Contrato'])

    X = df[columns]
    y = df['Adjudicado']

    # Dividimos los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if entrenar:
        # Creamos el modelo
        model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Entrenamos el modelo
        model.fit(X_train, y_train)

        # Guardamos el modelo entrenado
        joblib.dump(model, 'modelo_entrenado.pkl')

        # Hacemos predicciones con el conjunto de prueba
        y_pred = model.predict(X_test)

        # Evaluamos el modelo
        print('Precisión del modelo:', accuracy_score(y_test, y_pred))
    else:
        # Cargamos el modelo previamente entrenado
        model = joblib.load('modelo_entrenado.pkl')

    return model

# Cargamos los datos
df = pd.read_csv('./SECOP_II_-_Procesos_de_Contrataci_n (1).csv',nrows=50000)

# Entrenamos el modelo
model = entrenar_modelo(df, entrenar=True)

# Usamos el modelo para hacer predicciones
model = entrenar_modelo(df, entrenar=False)