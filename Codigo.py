import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import folium
from folium.plugins import MarkerCluster
from streamlit.components.v1 import html

def leer_datos(archivo=None, url=None):
    if archivo:
        df = pd.read_csv(archivo)
    elif url:
        response = requests.get(url)
        df = pd.read_csv(StringIO(response.text))
    else:
        raise ValueError("Debe proporcionar una ruta de archivo o una URL.")
    
    return df


def interpolar_valores(df, metodo='linear', order=None):
    if df.isnull().values.any():
        if metodo == 'polynomial' and order is not None:
            df_interpolado = df.interpolate(method=metodo, axis=0, order=order)
        else:
            df_interpolado = df.interpolate(method=metodo, axis=0)
        return df_interpolado
    else:
        return df


def analizar_correlacion(df):
    correlacion_global = df[['Edad', 'Ingreso_Anual_USD']].corr().iloc[0, 1]
    correlacion_genero = df.groupby('Género')[['Edad', 'Ingreso_Anual_USD']].corr().iloc[0, 1]
    correlacion_frecuencia = df.groupby('Frecuencia_Compra')[['Edad', 'Ingreso_Anual_USD']].corr().iloc[0, 1]

    return {
        'Correlación Global': correlacion_global,
        'Correlación por Género': correlacion_genero,
        'Correlación por Frecuencia de Compras': correlacion_frecuencia
    }


def mostrar_graficos(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Edad', y='Ingreso_Anual_USD')
    plt.title("Gráfico Global de Edad vs Ingreso Anual")
    plt.xlabel('Edad')
    plt.ylabel('Ingreso Anual')
    st.pyplot(plt)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Edad', y='Ingreso_Anual_USD', hue='Género', palette="Set1")
    plt.title("Gráfico de Edad vs Ingreso Anual por Género")
    plt.xlabel('Edad')
    plt.ylabel('Ingreso Anual')
    st.pyplot(plt)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Edad', y='Ingreso_Anual_USD', hue='Frecuencia_Compra', palette="Set2")
    plt.title("Gráfico de Edad vs Ingreso Anual por Frecuencia de Compras")
    plt.xlabel('Edad')
    plt.ylabel('Ingreso Anual')
    st.pyplot(plt)


def crear_mapa(df, segmento='global'):
    if df['Latitud'].isnull().any() or df['Longitud'].isnull().any():
        st.warning("Algunas filas tienen valores nulos en las columnas 'Latitud' o 'Longitud'.")

    mapa = folium.Map(location=[df['Latitud'].mean(), df['Longitud'].mean()], zoom_start=10)
    marker_cluster = MarkerCluster().add_to(mapa)

    if segmento == 'global':
        for idx, row in df.iterrows():
            if pd.notnull(row['Latitud']) and pd.notnull(row['Longitud']):
                folium.Marker(location=[row['Latitud'], row['Longitud']], popup=f"ID: {row['ID_Cliente']}, Nombre: {row['Nombre']}").add_to(marker_cluster)
    elif segmento == 'genero':
        for idx, row in df.iterrows():
            if pd.notnull(row['Latitud']) and pd.notnull(row['Longitud']):
                folium.Marker(location=[row['Latitud'], row['Longitud']], popup=f"Género: {row['Género']}, ID: {row['ID_Cliente']}, Nombre: {row['Nombre']}").add_to(marker_cluster)
    elif segmento == 'frecuencia':
        for idx, row in df.iterrows():
            if pd.notnull(row['Latitud']) and pd.notnull(row['Longitud']):
                folium.Marker(location=[row['Latitud'], row['Longitud']], popup=f"Frecuencia: {row['Frecuencia_Compra']}, ID: {row['ID_Cliente']}, Nombre: {row['Nombre']}").add_to(marker_cluster)

    # Muestra el mapa en Streamlit usando HTML
    map_html = mapa._repr_html_()
    html(map_html, height=500)


def procesar_datos(metodo_interpolacion='linear', order=None, archivo=None, url=None):
    df = leer_datos(archivo, url)

    df['Edad'] = pd.to_numeric(df['Edad'], errors='coerce')
    df['Ingreso_Anual_USD'] = pd.to_numeric(df['Ingreso_Anual_USD'], errors='coerce')

    df_interpolado = interpolar_valores(df, metodo_interpolacion, order)

    correlaciones = analizar_correlacion(df_interpolado)

    mostrar_graficos(df_interpolado)

    crear_mapa(df_interpolado, segmento='global')

    return df_interpolado, correlaciones


st.title('Análisis de Datos de Clientes')

metodo = st.radio("¿Cómo te gustaría proporcionar los datos?", ('Archivo', 'URL'))

if metodo == 'Archivo':
    archivo = st.file_uploader("Sube un archivo CSV", type="csv")
    metodo_interpolacion = st.selectbox("Método de interpolación", ('linear', 'polynomial'))
    if metodo_interpolacion == 'polynomial':
        order = st.number_input("Especifica el orden del polinomio", min_value=1, max_value=10)
    else:
        order = None

    if archivo:
        df_resultado, correlaciones = procesar_datos(metodo_interpolacion=metodo_interpolacion, order=order, archivo=archivo)

elif metodo == 'URL':
    url = st.text_input("Ingresa la URL del archivo CSV")
    metodo_interpolacion = st.selectbox("Método de interpolación", ('linear', 'polynomial'))
    if metodo_interpolacion == 'polynomial':
        order = st.number_input("Especifica el orden del polinomio", min_value=1, max_value=10)
    else:
        order = None

    if url:
        df_resultado, correlaciones = procesar_datos(metodo_interpolacion=metodo_interpolacion, order=order, url=url)

st.write("Datos procesados:", df_resultado)
st.write("Correlación Global:", correlaciones['Correlación Global'])
st.write("Correlación por Género:", correlaciones['Correlación por Género'])
st.write("Correlación por Frecuencia de Compras:", correlaciones['Correlación por Frecuencia de Compras'])
