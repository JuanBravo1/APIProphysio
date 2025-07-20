from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)


modelo_tipo = joblib.load('modelo_tipo_terapia.joblib')
modelo_sesiones = joblib.load('Sesiones_red_neuronal.joblib')
modelo_semanas = joblib.load('Semanas_red_neuronal.joblib')
modelo_costos = joblib.load('Costos_red_neuronal.joblib')

@app.route('/predecir', methods=['POST'])
def predecir():
    data = request.get_json()

    # Extraer variables necesarias
    enfermedad = data.get('Enfermedad_base')
    diagnostico = data.get('Diagnostico_inicial')
    edad = data.get('Edad')
    tiempo = data.get('Tiempo_con_los_sintomas')
    dolor = data.get('Dolor')
    movilidad = data.get('Nivel_movilizacion_actual')

    # Modelo 1: Tipo de terapia
    tipo_input = [[enfermedad, diagnostico, edad]]
    tipo_pred = modelo_tipo.predict(tipo_input)[0]

    # Modelo 2: Número de sesiones
    sesiones_input = [[tiempo, dolor, movilidad]]
    sesiones_pred = modelo_sesiones.predict(sesiones_input)[0]

    # Modelo 3: Semanas de recuperación
    semanas_input = [[sesiones_pred]]
    semanas_pred = modelo_semanas.predict(semanas_input)[0]

    # Modelo 4: Costo
    costo_input = [[sesiones_pred]]
    costo_pred = modelo_costos.predict(costo_input)[0]

    return jsonify({
        'Tipo_terapia': tipo_pred,
        'Sesiones_estimadas': sesiones_pred,
        'Semanas_recuperacion': semanas_pred,
        'Costo_estimado': costo_pred
    })

if __name__ == '__main__':
    app.run(debug=True)