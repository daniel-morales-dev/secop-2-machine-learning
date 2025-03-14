# pip install -r requirements.txt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve,  precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import time

# Definición de la función transform_with_unknowns
def transform_with_unknowns(column, encoder):
    # Transforma los datos conocidos
    known_mask = column.isin(encoder.classes_)
    column_known = column[known_mask]
    transformed_known = encoder.transform(column_known)

    # Asigna -1 a los datos desconocidos
    transformed = column.copy()
    transformed[known_mask] = transformed_known
    transformed[~known_mask] = -1

    return transformed

start_time = time.time()

# Cargamos los datos
df = pd.read_csv('./SECOP_II_-_Procesos_de_Contrataci_n (1).csv',nrows=1000000)

# Muestreo de datos, solo usamos el 10% de los datos, una mustra. 300.000
df = df.sample(frac=0.2)

# Seleccionamos las columnas que vamos a usar para predecir
columns = [ 'Departamento Entidad', 'Ciudad Entidad', 'OrdenEntidad', 'Entidad Centralizada', 'Fase', 'Precio Base', 'Modalidad de Contratacion', 'Duracion', 'Unidad de Duracion', 'Tipo de Contrato', 'Subtipo de Contrato','Justificación Modalidad de Contratación','Proveedores Invitados']

# Preprocesamiento de los datos
# Convertimos las columnas categóricas a valores numéricos
le = LabelEncoder()

le_adjudicado = LabelEncoder()
df['Adjudicado'] = le_adjudicado.fit_transform(df['Adjudicado'])

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

le_justificacion = LabelEncoder()
df['Justificación Modalidad de Contratación'] = le_justificacion.fit_transform(df['Justificación Modalidad de Contratación'])

le_proveedores = LabelEncoder()
df['Proveedores Invitados'] = le_proveedores.fit_transform(df['Proveedores Invitados'])


# Eliminar la columna 'Entidad' después de las transformaciones
df.drop('Entidad', axis=1, inplace=True)


X = df[columns]
y = df['Adjudicado']

# Dividimos los datos en conjuntos de entrenamiento y prueba, 20% de los datos se usaran como prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creamos el modelo, usamos Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenamos el modelo
model.fit(X_train, y_train)

# Hacemos predicciones con el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluamos el modelo
print('Precisión del modelo [Conjunto prueba]:', accuracy_score(y_test, y_pred))

# Precisión en el conjunto de entrenamiento
y_train_pred = model.predict(X_train)
print('Precisión en el conjunto de entrenamiento:', accuracy_score(y_train, y_train_pred))

# Calculamos la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión:")
print(conf_matrix)


# Calculamos la precisión, recall y F1-score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precisión: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
# Calculamos la precisión, recall y F1-score
precision, recall, thresholds = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])

print(f"Precisión: {precision}")
print(f"Recall: {recall}")

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva de Precisión-Recall - Random Forest')
plt.show()

# Aplicamos cross-validation
""" scores = cross_val_score(model, X.sample(frac=0.1), y.sample(frac=0.1), cv=3)
print('Precisión de cross-validation:', scores.mean()) """

# Guardar el modelo en un archivo
joblib.dump(model, 'modelo_entrenado.joblib')

# PRUEBA MANUAL #
# Selecciona una fila aleatoria del DataFrame completo
fila_aleatoria = df[df['Adjudicado'] == 1].sample()


# Guarda la etiqueta verdadera
etiqueta_verdadera = fila_aleatoria['Adjudicado']
print(etiqueta_verdadera)

# Selecciona solo las columnas que necesitas para la predicción
fila_aleatoria = fila_aleatoria[columns]


# Aplica las mismas transformaciones que aplicaste a tus datos de entrenamiento
fila_aleatoria['OrdenEntidad'] = transform_with_unknowns(fila_aleatoria['OrdenEntidad'], le_orden)
fila_aleatoria['Entidad Centralizada'] = transform_with_unknowns(fila_aleatoria['Entidad Centralizada'], le_centralizada)
fila_aleatoria['Fase'] = transform_with_unknowns(fila_aleatoria['Fase'], le_fase)
fila_aleatoria['Modalidad de Contratacion'] = transform_with_unknowns(fila_aleatoria['Modalidad de Contratacion'], le_modalidad)
fila_aleatoria['Unidad de Duracion'] = transform_with_unknowns(fila_aleatoria['Unidad de Duracion'], le_duracion)
fila_aleatoria['Tipo de Contrato'] = transform_with_unknowns(fila_aleatoria['Tipo de Contrato'], le_tipo)
fila_aleatoria['Subtipo de Contrato'] = transform_with_unknowns(fila_aleatoria['Subtipo de Contrato'], le_subtipo)

# Haz una predicción con tu modelo
prediccion = model.predict(fila_aleatoria)

# Imprime la etiqueta verdadera y la predicción
print('Etiqueta verdadera:', etiqueta_verdadera)
print('Predicción del modelo:', prediccion)

probabilidades = model.predict_proba(fila_aleatoria)
# La segunda columna representa la probabilidad de la clase "adjudicado"
probabilidad_adjudicacion = probabilidades[0, 1]

# Imprime la probabilidad
print('Probabilidad de Adjudicación:', probabilidad_adjudicacion)

# 1. ANALISIS DE AFINIDAD
fila_afinidad = fila_aleatoria[columns]
# Calcula la similitud con otras filas usando alguna métrica, por ejemplo, la correlación
similitud = X_train.apply(lambda row: np.sum(row[columns] == fila_afinidad.iloc[0][columns]), axis=1)


# Agrega la columna de similitud al DataFrame original
X_train['Similitud'] = similitud

# Muestra contratos similares ordenados por similitud
contratos_similares = X_train.sort_values('Similitud', ascending=False).head(10)

columns_to_show = ['Ciudad Entidad','Tipo de Contrato','Precio Base']


# Revierte las transformaciones para obtener los valores reales
contratos_similares['Ciudad Entidad'] = le_ciudad.inverse_transform(contratos_similares['Ciudad Entidad'])
contratos_similares['OrdenEntidad'] = le_orden.inverse_transform(contratos_similares['OrdenEntidad'])
contratos_similares['Entidad Centralizada'] = le_centralizada.inverse_transform(contratos_similares['Entidad Centralizada'])
contratos_similares['Fase'] = le_fase.inverse_transform(contratos_similares['Fase'])
contratos_similares['Modalidad de Contratacion'] = le_modalidad.inverse_transform(contratos_similares['Modalidad de Contratacion'])
contratos_similares['Unidad de Duracion'] = le_duracion.inverse_transform(contratos_similares['Unidad de Duracion'])
contratos_similares['Tipo de Contrato'] = le_tipo.inverse_transform(contratos_similares['Tipo de Contrato'])
contratos_similares['Subtipo de Contrato'] = le_subtipo.inverse_transform(contratos_similares['Subtipo de Contrato'])

print('Contratos Similares:')
print(contratos_similares[columns_to_show])

end_time = time.time()
print('Tiempo total de ejecución:', end_time - start_time, 'segundos')