import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from io import StringIO
import folium
from folium.plugins import MarkerCluster
import streamlit as st
import pandas as pd

def leer_datos(archivo=None, url=None):
    """
    Lee datos desde un archivo local o desde una URL.
    """
    if archivo:
        df = pd.read_csv(archivo)
    elif url:
        response = requests.get(url)
        df = pd.read_csv(StringIO(response.text))
    else:
        raise ValueError("Debe proporcionar una ruta de archivo o una URL.")
    
    return df

def interpolar_valores(df, metodo='linear', order=None):
    """
    Rellena los valores faltantes en un DataFrame utilizando la interpolación.
    """
    if df.isnull().values.any():
        if metodo == 'polynomial' and order is not None:
            df_interpolado = df.interpolate(method=metodo, axis=0, order=order)
        else:
            df_interpolado = df.interpolate(method=metodo, axis=0)
        return df_interpolado
    else:
        return df

def analizar_correlacion(df):
    """
    Realiza el análisis de correlación entre edad e ingreso anual.
    """
    correlacion_global = df[['Edad', 'Ingreso_Anual_USD']].corr().iloc[0, 1]
    correlacion_genero = df.groupby('Género')[['Edad', 'Ingreso_Anual_USD']].corr().iloc[0, 1]
    correlacion_frecuencia = df.groupby('Frecuencia_Compra')[['Edad', 'Ingreso_Anual_USD']].corr().iloc[0, 1]

    return {
        'Correlación Global': correlacion_global,
        'Correlación por Género': correlacion_genero,
        'Correlación por Frecuencia de Compras': correlacion_frecuencia
    }

def mostrar_graficos(df):
    """
    Muestra gráficos de dispersión de Edad vs Ingreso Anual.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Edad', y='Ingreso_Anual_USD')
    plt.title("Gráfico Global de Edad vs Ingreso_Anual_USD")
    st.pyplot(plt)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Edad', y='Ingreso_Anual_USD', hue='Género', palette="Set1")
    plt.title("Gráfico de Edad vs Ingreso Anual por Género")
    st.pyplot(plt)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Edad', y='Ingreso_Anual_USD', hue='Frecuencia_Compra', palette="Set2")
    plt.title("Gráfico de Edad vs Ingreso Anual por Frecuencia de Compras")
    st.pyplot(plt)

def crear_mapa(df, segmento='global'):
    """
    Crea mapas de ubicación de clientes usando Folium y los muestra en el notebook.
    Permite ver la ubicación de clientes a nivel global, por género o por frecuencia de compra.

    Args:
        df (pd.DataFrame): El DataFrame que contiene los datos.
        segmento (str, optional): El segmento a mostrar en el mapa. Puede ser 'global', 'genero' o 'frecuencia'. Defaults to 'global'.
    """
    # Verificar si las columnas de latitud y longitud no contienen valores nulos
    if df['Latitud'].isnull().any() or df['Longitud'].isnull().any():
        print("Advertencia: Algunas filas tienen valores nulos en las columnas 'Latitud' o 'Longitud'.")

    # Crear un DataFrame para Streamlit con las columnas 'Latitud' y 'Longitud'
    mapa_df = df[['Latitud', 'Longitud']].dropna()

    # Mostrar el mapa usando Streamlit
    st.map(mapa_df)

    # Si quieres usar Folium para agregar más características (por ejemplo, clusters):
    if segmento == 'global':
        # Inicializar el mapa con folium (usando el promedio de las coordenadas)
        mapa = folium.Map(location=[df['Latitud'].mean(), df['Longitud'].mean()], zoom_start=10)
        marker_cluster = MarkerCluster().add_to(mapa)
        
        for idx, row in df.iterrows():
            if pd.notnull(row['Latitud']) and pd.notnull(row['Longitud']):
                folium.Marker(location=[row['Latitud'], row['Longitud']], 
                              popup=f"ID: {row['ID_Cliente']}, Nombre: {row['Nombre']}").add_to(marker_cluster)

        # Mostrar el mapa de Folium (en Streamlit)
        st.write(mapa._repr_html_(), unsafe_allow_html=True)
def crear_mapa(df, segmento='global'):
    """
    Crea mapas de ubicación de clientes usando Folium y los muestra en el notebook.
    Permite ver la ubicación de clientes a nivel global, por género o por frecuencia de compra.

    Args:
        df (pd.DataFrame): El DataFrame que contiene los datos.
        segmento (str, optional): El segmento a mostrar en el mapa. Puede ser 'global', 'genero' o 'frecuencia'. Defaults to 'global'.
    """
    # Verificar si las columnas de latitud y longitud no contienen valores nulos
    if df['Latitud'].isnull().any() or df['Longitud'].isnull().any():
        print("Advertencia: Algunas filas tienen valores nulos en las columnas 'Latitud' o 'Longitud'.")

    # Crear un DataFrame para Streamlit con las columnas 'Latitud' y 'Longitud'
    mapa_df = df[['Latitud', 'Longitud']].dropna()

    # Mostrar el mapa usando Streamlit
    st.map(mapa_df)

    # Si quieres usar Folium para agregar más características (por ejemplo, clusters):
    if segmento == 'global':
        # Inicializar el mapa con folium (usando el promedio de las coordenadas)
        mapa = folium.Map(location=[df['Latitud'].mean(), df['Longitud'].mean()], zoom_start=10)
        marker_cluster = MarkerCluster().add_to(mapa)
        
        for idx, row in df.iterrows():
            if pd.notnull(row['Latitud']) and pd.notnull(row['Longitud']):
                folium.Marker(location=[row['Latitud'], row['Longitud']], 
                              popup=f"ID: {row['ID_Cliente']}, Nombre: {row['Nombre']}").add_to(marker_cluster)

        # Mostrar el mapa de Folium (en Streamlit)
        st.write(mapa._repr_html_(), unsafe_allow_html=True)


def procesar_datos(metodo_interpolacion='linear', order=None, archivo=None, url=None):
    """
    Función principal que procesa los datos, permite leer desde un archivo o URL,
    y rellena los valores faltantes con la interpolación seleccionada.
    """
    df = leer_datos(archivo, url)

    df['Edad'] = pd.to_numeric(df['Edad'], errors='coerce')
    df['Ingreso_Anual_USD'] = pd.to_numeric(df['Ingreso_Anual_USD'], errors='coerce')

    df_interpolado = interpolar_valores(df, metodo_interpolacion, order)

    correlaciones = analizar_correlacion(df_interpolado)

    mostrar_graficos(df_interpolado)
    crear_mapa(df_interpolado, segmento='global')

    return df_interpolado, correlaciones

def main():
    st.title("Análisis de Datos con Streamlit")

    metodo = st.selectbox("¿Cómo te gustaría proporcionar los datos?", ["archivo", "url"])

    archivo = None
    url = None
    if metodo == "archivo":
        archivo = st.file_uploader("Sube tu archivo CSV", type="csv")
    elif metodo == "url":
        url = st.text_input("Ingresa la URL del archivo CSV")

    metodo_interpolacion = st.selectbox("¿Qué método de interpolación te gustaría usar?", ["linear", "polynomial"])
    order = None
    if metodo_interpolacion == "polynomial":
        order = st.number_input("Especifica el orden del polinomio", min_value=1, value=2)

    if metodo == "archivo" and archivo is not None:
        df_resultado, correlaciones = procesar_datos(metodo_interpolacion, order, archivo=archivo)
    elif metodo == "url" and url:
        df_resultado, correlaciones = procesar_datos(metodo_interpolacion, order, url=url)
    
    if df_resultado is not None:
        st.write("Datos procesados:", df_resultado)
        st.write("Correlación Global:", correlaciones['Correlación Global'])
        st.write("Correlación por Género:", correlaciones['Correlación por Género'])
        st.write("Correlación por Frecuencia de Compras:", correlaciones['Correlación por Frecuencia de Compras'])

if __name__ == "__main__":
    main()
