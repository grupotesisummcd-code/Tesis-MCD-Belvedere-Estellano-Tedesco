# %% [markdown]
# ## Librerías

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from pathlib import Path
from scipy.stats import mode
from ydata_profiling import ProfileReport

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from catboost import CatBoostRegressor, Pool

# %% [markdown]
# ## Importar Dataset

# %%
# Importar archivo excel desde el Drive

ruta_archivo = Path(r"/Users/valetedesco/Library/CloudStorage/GoogleDrive-grupotesisummcd@gmail.com/.shortcut-targets-by-id/1aI4ids63D_lROjgqkvdwMPmzc-h0_Cc_/Tesis | MCD | UM/turismo_receptivo.xlsx")
#ruta_archivo = Path(r"H:\.shortcut-targets-by-id\1aI4ids63D_lROjgqkvdwMPmzc-h0_Cc_\Tesis   MCD   UM\turismo_receptivo.xlsx")

df = pd.read_excel(ruta_archivo)

df.head()

# %%
# Ver resumen de df

df.info()

# %%
# Cantidad de variables por tipo de dato

df.dtypes.value_counts()

# %% [markdown]
# ## Análisis Exploratorio de Datos

# %%
# Eliminar columnas de 'Id'

cols_to_drop = [col for col in df.columns if col.startswith('Id')] + ['IsEstudio']
df = df.drop(columns=cols_to_drop)

# %%
df.info()

# %%
# Verificar composición de variable 'Estadia''

print(df['Estadia'].unique())

# %%
# Cantidad de valores de la variable estadia

df['Estadia'].value_counts().sort_index()

# %%
# Redondear los valores de 'Estadia' y convertir a entero, asegurando que el mínimo sea 1

df['Estadia'] = np.floor(df['Estadia'] + 0.5).astype(int)

print(df['Estadia'].sort_values().unique())

# %%
# Valores de 'Estadia' menores a 50 días

count_menor_50 = df[df['Estadia'] <= 50].shape[0]
print(f"Número de registros con 'Estadia' menor a 50 días: {count_menor_50}") 

# %%
# Valores de 'Estadia' mayor a 50 días

count_mayor_50 = df[df['Estadia'] > 50].shape[0]
print(f"Número de registros con 'Estadia' mayor a 50 días: {count_mayor_50}")  

# %%
# Distribución de la variable 'Estadia'

plt.figure(figsize=(10, 6))
sns.histplot(df['Estadia'], bins=365, kde=True)
plt.xlabel('Estadia')
plt.ylabel('Frecuencia')
plt.show()

# %%
# Distribución de la variable 'Estadia' menor a 50 dias

plt.figure(figsize=(10, 6))
sns.histplot(df[df['Estadia'] < 50]['Estadia'], bins=50, kde=True)
plt.xlabel('Estadia')
plt.ylabel('Frecuencia')
plt.show()

# %%
# Porcentaje de la distribucion de 'Estadia' menor a 50 dias

porcentaje_menor_50 = (df[df['Estadia'] < 50].shape[0] / df.shape[0]) * 100
print(f"Porcentaje de registros con 'Estadia' menor a 50 días: {porcentaje_menor_50:.2f}%") 

# %%
# Verificar composición de varibale 'Gente'

print(df['Gente'].unique())

# %%
# Distribución de la variable 'Gente'

plt.figure(figsize=(8, 5))
orden_gente = sorted(df['Gente'].unique())
sns.countplot(data=df, x='Gente', order=orden_gente)
plt.xlabel('Gente')
plt.ylabel('Frecuencia')
plt.show()

# %%
# Estadísticos descriptivos de la variable 'Gasto Total'

df['GastoTotal'].describe()

# %%
# Verificar si hay valores nulos en 'GastoTotal'

nulos = df['GastoTotal'].isnull().sum()
print(f"Número de valores nulos en 'GastoTotal': {nulos}")

# %%
# Distribución de la variable 'Gasto Total'

plt.figure(figsize=(10, 6))
sns.histplot(df['GastoTotal'].dropna(), bins=50, kde=True)
plt.xlabel('Gasto Total')
plt.ylabel('Frecuencia')
plt.show()

# %%
# Distribución de 'Gasto Total' menor a 2000

plt.figure(figsize=(10, 6))
sns.histplot(df[df['GastoTotal'] < 2000]['GastoTotal'].dropna(), bins=30, kde=True)
plt.xlabel('Gasto Total')
plt.ylabel('Frecuencia')
plt.show()

# %%
# Distribución de 'Gasto Total' acotada entre 0 y 250

plt.figure(figsize=(10, 6))
sns.histplot(df[df['GastoTotal'] <= 250]['GastoTotal'].dropna(), bins=50, kde=True)
plt.xlabel('Gasto Total')
plt.ylabel('Frecuencia')
plt.show()

# %%
# Cantidad de valores con 'GastoTotal' igual a 0

df[df['GastoTotal'] == 0].shape[0]

# %%
# Cantidad de valores con 'GastoTotal' menor o igual a 5

df[df['GastoTotal'] <= 5].shape[0]

# %%
# Eliminar registros con GastoTotal menor o igual a 5

df = df[df['GastoTotal'] > 5].copy()

# %%
# Verificar que se eliminaron correctamente los registros con GastoTotal <= 5

df[df['GastoTotal'] <= 5]

# %%
# Relación entre 'Estadia' y 'GastoTotal'

df.plot.scatter('Estadia', 'GastoTotal')

# %%
# Relación entre 'Gente'y 'GastoTotal'

df.plot.scatter('Gente', 'GastoTotal')

# %%
# Crear columna de Gasto total por día y por persona

df['Gasto/p/d'] = (df['GastoTotal'] / df['Gente'] / df['Estadia']).astype('float64')

# %%
# Estadísticos descriptivos de 'Gasto/p/d'

df['Gasto/p/d'].describe()

# %%
# Cuantiles seleccionados de 'Gasto/p/d'

quantiles = df['Gasto/p/d'].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
print("Cuantiles seleccionados:\n", quantiles)

# %%
# Eliminar outliers en 'Gasto/p/d'

q_high = df["Gasto/p/d"].quantile(0.99)
df = df[df["Gasto/p/d"] < q_high].copy()

# %%
# Estadísticos descriptivos de 'Gasto/p/d' luego de eliminar outliers

col = 'Gasto/p/d'
print(df[col].describe())

# %%
# Distribucion del 'Gasto/p/d' luego de eliminar outliers

plt.figure(figsize=(10, 6))
sns.histplot(df['Gasto/p/d'], bins=30, kde=True)
plt.xlabel('Gasto/p/d')
plt.ylabel('Frecuencia')
plt.show()

# %%
# Eliminar columnas de Gasto que no se usarán en el modelo

cols_a_eliminar = [
    'GastoTotal', 'GastoAlimentacion','GastoTransporte',
    'GastoCultural','GastoTours','GastoCompras','GastoOtros','Coef','CoefTot'
]
df = df.drop(columns=[col for col in cols_a_eliminar if col in df.columns])

# %%
df.info()

# %%
# Generar informe EDA automático

profile = ProfileReport(df, title="Informe EDA Automático", explorative=True)
profile.to_notebook_iframe()
profile.to_file("informe_eda_automatico.html")

# %%
# Variables categóricas y su relación con 'Gasto/p/d'

variables = ["Motivo", "Pais", "Destino", "Alojamiento", "Ocupacion"]

tabla_final = []

for var in variables:

    temp = (
        df.groupby(var)["Gasto/p/d"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .reset_index()
    )
    
    temp.columns = ["Categoria", "Promedio Gasto/p/d"]
    temp["Variable"] = var
    temp = temp[["Variable", "Categoria", "Promedio Gasto/p/d"]]
    
    tabla_final.append(temp)

tabla_compilada = pd.concat(tabla_final, ignore_index=True)
tabla_compilada

# %%
# Visualizar boxplots de 'Gasto/p/d' para las principales categorías de cada variable categórica

variables = ["Pais", "Destino", "Motivo", "Alojamiento", "Ocupacion"]

fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
axes = axes.flatten()

for ax, var in zip(axes, variables):

    top5_categorias = (
        df.groupby(var)["Gasto/p/d"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .index
    )
    
    temp = df[df[var].isin(top5_categorias)].copy()
    
    orden_categorias = (
        temp.groupby(var)["Gasto/p/d"]
        .mean()
        .sort_values(ascending=False)
        .index
    )

    sns.boxplot(
        data=temp,
        x="Gasto/p/d",
        y=var,
        order=orden_categorias,
        orient="h",
        ax=ax,
        showfliers=True
    )

    ax.set_title(f"Gasto/p/d por {var} (Top 5)", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis='x', labelbottom=True, labelsize=9)
    ax.grid(False)
    sns.despine(ax=ax, top=True, right=True, left=True)

axes[-1].set_visible(False)
plt.tight_layout()
plt.subplots_adjust(hspace=0.35)
plt.show()

# %% [markdown]
# ## Transformaciones

# %% [markdown]
# ### Transformaciones df original

# %%
# Crear columna con el mes de ingreso y egreso

df['Mes_Ingreso'] = df['FechaIngreso'].dt.month

df['Mes_Egreso'] = df['FechaEgreso'].dt.month

# %%
# Crear nuevas columnas con el formato 'Quarter/Year' para Ingreso y Egreso

df['Quarter_Year_Ingreso'] = df['FechaIngreso'].dt.to_period('Q').astype(str).apply(lambda x: int(x.replace('Q','')))
df['Quarter_Year_Egreso'] = df['FechaEgreso'].dt.to_period('Q').astype(str).apply(lambda x: int(x.replace('Q','')))

# %%
# Definir etiquetas de Temporada según mes de ingreso

condiciones = [
    df['Mes_Ingreso'].isin([1, 2, 3, 12]),   # Alta
    df['Mes_Ingreso'].isin([4, 5, 10, 11]),  # Media
    df['Mes_Ingreso'].isin([6, 7, 8, 9])     # Baja
]

valores = ['Alta', 'Media', 'Baja']

df['Temporada'] = np.select(condiciones, valores, default='Desconocido')

df[['Mes_Ingreso', 'Temporada']].drop_duplicates().sort_values('Mes_Ingreso')


# %%
# Crear columna binaria "Gasta en Alojamiento?"

df['Gasta en Alojamiento?'] = np.where(df['GastoAlojamiento'].fillna(0) > 0, 'Si', 'No')

# %%
# Crear columna transito

df['Es Transito?'] = np.where(
    (
        ((df['Motivo'] == 'Transito') & (df['Gasta en Alojamiento?'] == 'No')) |
        ((df['Motivo'] == 'Transito') & (df['Gasta en Alojamiento?'] == 'Si') & (df['Estadia'] <= 1))
    ),
    'Sí',
    'No'
)
print(df['Es Transito?'].unique())

# %%
# Crear columna 'Personas_Dia' como producto de 'Gente' y 'Estadia'

df["Personas_Dia"] = df["Gente"] * df["Estadia"]

# %%
# Calcular el promedio de gasto por día/persona por mes

gasto_por_mes = df.groupby('Mes_Ingreso')['Gasto/p/d'].mean().reset_index()
gasto_por_mes

# %%
# Graficar la tendencia del gasto por día/persona según el mes de ingreso

plt.figure(figsize=(10,6))
sns.lineplot(data=gasto_por_mes, x='Mes_Ingreso', y='Gasto/p/d', marker='o')
plt.xlabel('Mes')
plt.ylabel('Gasto por día / persona')
plt.xticks(range(1,13))
plt.grid(False)
plt.show()

# %%
# Calcular el promedio de gasto por día/persona por temporada

gasto_por_temporada = df.groupby('Temporada')['Gasto/p/d'].mean().reset_index()
gasto_por_temporada

# %%
# Eliminar columnas no relevantes

cols_a_eliminar = [
    'FechaIngreso','FechaEgreso','Estudio','Otra Localidad','Otro Departamento','GastoAlojamiento'
]
df = df.drop(columns=[col for col in cols_a_eliminar if col in df.columns])

# %%
df.info()

# %%
print(f"Dataset limpio: {df.shape[0]} filas, media gasto/día = {round(df['Gasto/p/d'].mean(),2)}")

# %%
# Conteo de nulos y valores únicos

print(f"\n Valores nulos: {df[col].isna().sum()}")

# %% [markdown]
# ### Dividir df original en df_train y df_test

# %%
# Dividir el df original en train y test 80/20

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# %%
print("Shape de df_train:", df_train.shape)
print("Shape de df_test:", df_test.shape)

# %% [markdown]
# ### Transformaciones en df_train

# %% [markdown]
# #### Lugar de Ingreso
# 
# Lugar por donde el turista ingresó a Uruguay.

# %%
# Lugar de Ingreso

lugaresI_principales = ['Colonia','Aeropuerto de Carrasco','Fray Bentos','Salto','Paysandú',
                       'Puerto de montevideo','Chuy','Rivera','Aeropuerto de Punta del Este']

df_train['Lugar Ingreso'] = df_train['Lugar Ingreso'].apply(lambda x: x if x in lugaresI_principales else 'Otros')

print(df_train['Lugar Ingreso'].unique())

# %% [markdown]
# #### Transporte Internacional de Ingreso
# 
# Medio de transporte internacional utilizado para ingresar a Uruguay.

# %%
# Transporte Internacional de Ingreso

plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df_train,
    x='Transporte Internacional de Ingreso',
    y='Gasto/p/d',
    order=df_train['Transporte Internacional de Ingreso'].value_counts().index
)
plt.title('Boxplot de Gasto/p/d por Transporte Internacional de Ingreso')
plt.xlabel('Transporte Internacional de Ingreso')
plt.ylabel('Gasto/p/d')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
agrupacion_transporteI = {
    'Aereo': 'Aereo',
    'Maritimo - Fluvial': 'Maritimo',
    'Terrestre Auto': 'Terrestre',
    'Terrestre Bus': 'Terrestre',
    'Terrestre Otros': 'Terrestre',
    'Otros': 'Terrestre',  
}

df_train['Transporte Internacional de Ingreso'] = df_train['Transporte Internacional de Ingreso'].map(agrupacion_transporteI)

print(df_train['Transporte Internacional de Ingreso'].unique())

# %% [markdown]
# #### Pais
# 
# Nacionalidad de turista que ingresó a Uruguay.

# %%
# Pais

tabla_pais = (
    df_train.groupby('Pais')['Gasto/p/d']
      .agg(['mean', 'median', 'min', 'max', 'count',
            lambda x: x.quantile(0.75) - x.quantile(0.25)])
      .reset_index()
)
tabla_pais.columns = ['Pais', 'Media', 'Mediana', 'Min', 'Max', 'Cantidad', 'Rango IQR']

tabla_pais = tabla_pais.sort_values('Media', ascending=False).reset_index(drop=True)

display(tabla_pais)

# %%
paises = [
    'Uruguay', 'Argentina', 'Brasil', 'Chile', 'Paraguay', 'Otro de America'
]
paises_otro_america = [
    'Mexico', 'Canada', 'Colombia', 'EE.UU.', 'Ecuador', 'Peru', 'Bolivia', 'Cuba', 'Venezuela'
]
paises_resto_mundo = [
    'Otro pais de Asia', 'Japon', 'China', 'Israel', 'Africa u Oceania'
]
paises_europa = [
    'Gran Bretaña', 'Suecia', 'Italia', 'Suiza', 'Francia', 'Alemania', 'España', 'Otro de Europa'
]

def agrupar_pais(pais):
    if pais in paises:
        return pais
    elif pais in paises_otro_america:
        return 'Otro de America'
    elif pais in paises_resto_mundo:
        return 'Resto del mundo'
    elif pais in paises_europa:
        return 'Europa'
    else:
        return pais

df_train['Pais'] = df_train['Pais'].apply(agrupar_pais)

print(df_train['Pais'].unique())

# %% [markdown]
# #### Residencia
# 
# Lugar de residencia de turista que ingresó a Uruguay.

# %%
# Residencia

map_residencia_a_pais = {
    
    'Buenos Aires':'Argentina','Rosario':'Argentina','Cordoba':'Argentina','Corrientes':'Argentina',
    'Pcia. Enre Rios':'Argentina','Otras Pcias. Argentinas':'Argentina','Otros Pcia. Bs. As.':'Argentina',
    'Otros Pcia. Santa Fe':'Argentina','Otras Pcia Cordoba':'Argentina',

    'San Pablo':'Brasil','Otros San Pablo':'Brasil','Rio de Janeiro':'Brasil','Porto Alegre':'Brasil',
    'Santa Catarina':'Brasil','Pelotas - Rio Grande':'Brasil','Rio Grande':'Brasil','Otra  ciudades de Brasil':'Brasil',
    
    'Paraguay':'Paraguay',
    
    'Chile':'Chile',
    
    'Mexico':'Mexico',
    
    'Colombia':'Colombia',
    
    'Peru':'Peru',
    
    'EEUU - Canada':'EEUU-Canada',
    
    'Reino Unido':'Reino Unido','Alemanai':'Alemania','España':'España','Europa':'Europa',
    
    'Otras ciudades America':'Otro America','Otras ciudades Sudamerica':'Otro Sudamerica',
    'Ciudades de Asia':'Asia','Ciudades Africa y Oceania':'Africa-Oceania',
    
    'Sin Datos':'Sin Datos'
}
df_train['Residencia'] = df_train['Residencia'].map(map_residencia_a_pais)

print(df_train['Residencia'].unique())

# %% [markdown]
# #### Motivo
# 
# Motivo del ingreso a Uruguay.

# %%
# Motivo

df_train['Motivo'] = df_train['Motivo'].replace(
    {'MICE':'Trabajo / Profesional / MICE','Trabajo / Profesional':'Trabajo / Profesional / MICE',
     'Estudios':'Otros','Religioso':'Otros','Compras':'Otros','Salud / wellness':'Otros'}
)

print(df_train['Motivo'].unique())

# %% [markdown]
# #### Ocupación
# 
# Ocupación o profesión del turista que ingresó a Uruguay.

# %%
# Ocupación

tabla_ocupacion = (
    df_train.groupby('Ocupacion')['Gasto/p/d']
      .agg(['mean', 'median', 'min', 'max', 'count',
            lambda x: x.quantile(0.75) - x.quantile(0.25)])
      .reset_index()
)
tabla_ocupacion.columns = ['Ocupacion', 'Media', 'Mediana', 'Min', 'Max', 'Cantidad', 'Rango IQR']

tabla_ocupacion = tabla_ocupacion.sort_values('Media', ascending=False).reset_index(drop=True)

display(tabla_ocupacion)

# %%
ocupacion_categoria = {
    
    "Director, gerente": "Profesionales y Patrones",
    "Patron, Com, Ind, Prod Agrop": "Profesionales y Patrones",
    "Prof, Tecnico, Docente, Artista": "Profesionales y Patrones",
    "Profesional independiente": "Profesionales y Patrones",
    "Profesional dependiente": "Profesionales y Patrones",

    "Empl. Adm, Cajero, Vendedor": "Empleados A",
    "Funcionario Publico": "Empleados A",
    "Deportista, Entrenador, Juez Dep": "Empleados A",
    "Trabajador Independiente": "Empleados A",
    "Militar, policia, Aduanero, Insp, bombero, Marinero": "Empleados A",

    "Trabajador dependiente": "Empleados B",
    "Jefe, Capataz, Encargado": "Empleados B",
    "Obrero esp, Conductor, Artesano": "Empleados B",
 
    "Rentista": "Inactivos con ingresos",
    "Jubilado, Pensionista": "Inactivos con ingresos",
 
    "Estudiante": "No remunerados",
    "Ama de casa": "No remunerados",

    "Mozo, Portero, Serv Dom, Otros Serv": "Otros",
    "Trabajador Agro, Pesca": "Otros",
    "Desocupado": "Otros",
    "Trabajador sin especializacion": "Otros",
    "Otra situacion Inactividad": "Otros",
    "Desconocido / Sin Datos": "Otros",
    "Sin Datos": "Otros",
    "Religioso": "Otros",
    "Otros": "Otros"
}

df_train["Ocupacion"] = df_train["Ocupacion"].map(ocupacion_categoria)
print(df_train['Ocupacion'].unique())

# %% [markdown]
# #### Localidad
# 
# Destino principal elegido por el turista que ingresó a Uruguay.

# %%
# Localidad

df_train['Localidad'] = df_train['Localidad'].replace(to_replace=['Sin Datos','Otros',''],value='Otros').fillna('Otros')
print(df_train['Localidad'].unique())

# %%
tabla_localidad = (
    df_train.groupby('Localidad')['Gasto/p/d']
      .agg(['mean', 'median', 'min', 'max', 'count',
            lambda x: x.quantile(0.75) - x.quantile(0.25)])
      .reset_index()
)
tabla_localidad .columns = ['Localidad', 'Media', 'Mediana', 'Min', 'Max', 'Cantidad', 'Rango IQR']

tabla_localidad  = tabla_localidad .sort_values('Cantidad', ascending=False).reset_index(drop=True)

print((tabla_localidad).head(9))

# %%
loc_principales = ['Montevideo','Punta del Este','Colonia del Sacramento','Transito','Termas del Dayman','Piriapolis', 'Salto', 'Paysandu']

df_train['Localidad'] = df_train['Localidad'].apply(lambda x: x if x in loc_principales else 'Otros')

print(df_train['Localidad'].unique())

# %% [markdown]
# #### Departamento
# 
# Departamento destino elegido por el turista que ingresó a Uruguay.

# %%
# Departamento

deptos_principales = ['Colonia','Montevideo','Maldonado','Salto','Canelones','Transito']

df_train['Departamento'] = df_train['Departamento'].apply(lambda x: x if x in deptos_principales else 'Otros').fillna('Otros')

print(df_train['Departamento'].unique())

# %% [markdown]
# #### Alojamiento
# 
# Alojamiento principal elegido por el turista que ingresó a Uruguay.

# %%
# Alojamiento

tabla_alojamiento = (
    df_train.groupby('Alojamiento')['Gasto/p/d']
      .agg(['mean', 'median', 'min', 'max', 'count',
            lambda x: x.quantile(0.75) - x.quantile(0.25)])
      .reset_index()
)
tabla_alojamiento.columns = ['Alojamiento', 'Media', 'Mediana', 'Min', 'Max', 'Cantidad', 'Rango IQR']

tabla_alojamiento = tabla_alojamiento.sort_values('Cantidad', ascending=False).reset_index(drop=True)

display(tabla_alojamiento)

# %%
mapa_alojamiento = {
    
    'Vivienda familiares/amigos residentes': 'Vivienda familiares/amigos',
    'Vivienda familiares/amigos no residentes': 'Vivienda familiares/amigos',
 
    'Hotel 4 estrellas': 'Hoteles 4 y 5 estrellas',
    'Hotel 5 estrellas': 'Hoteles 4 y 5 estrellas',
 
    'Hotel sin categorizar': 'Otros Hoteles',
    'Hotel 3 estrellas': 'Otros Hoteles',
    'Hotel 1 y 2 estrellas': 'Otros Hoteles',
    'Appart Hotel': 'Otros Hoteles',
    'Hotel/Albergue': 'Otros Hoteles',
 
    'Ninguno': 'Otros',
    'Camping': 'Otros',
    'Motor Home': 'Otros',
    'Tiempo Compartido': 'Otros',
    'Barco, Yate, Crucero': 'Otros',
    'Estancia Turistica': 'Otros',
    'Bed y Breakfast': 'Otros',
    'Sin Datos': 'Otros',
    'Cabañas/Bungalows': 'Otros',
    'Otros': 'Otros',
 
    'Vivienda propia': 'Vivienda propia',
 
    'Vivienda arrendada': 'Vivienda arrendada',
    'Vivienda arrendada por plataforma': 'Vivienda arrendada'
}

df_train['Alojamiento'] = df_train['Alojamiento'].map(mapa_alojamiento)
 
print(df_train['Alojamiento'].unique())

# %% [markdown]
# #### Transporte Local
# 
# Transporte utilizado durante la estadía en Uruguay.

# %%
# Transporte Local

transp_principales = ['Auto propio','Taxi - Bus','Auto familiares / amigos','Ninguno']

df_train['TransporteLocal'] = df_train['TransporteLocal'].apply(lambda x: x if x in transp_principales else 'Otros').fillna('Otros')

print(df_train['TransporteLocal'].unique())

# %% [markdown]
# #### Lugar de Egreso
# 
# Lugar por donde el turista egresó de Uruguay.

# %%
# Lugar de Egreso

lugaresE_principales = ['Colonia','Aeropuerto de Carrasco','Fray Bentos','Salto','Paysandú',
                       'Puerto de montevideo','Chuy','Rivera','Aeropuerto de Punta del Este']

df_train['Lugar Egreso'] = df_train['Lugar Egreso'].apply(lambda x: x if x in lugaresE_principales else 'Otros')

print(df_train['Lugar Egreso'].unique())

# %% [markdown]
# #### Transporte Internacional de Egreso
# 
# Medio de transporte internacional utilizado para egresar de Uruguay.

# %%
# Transporte Internacional de Egreso

plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df_train,
    x='Transporte Internacional de Egreso',
    y='Gasto/p/d',
    order=df_train['Transporte Internacional de Egreso'].value_counts().index
)
plt.title('Boxplot de Gasto/p/d por Transporte Internacional de Egreso')
plt.xlabel('Transporte Internacional de Egreso')
plt.ylabel('Gasto/p/d')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
agrupacion_transporteE = {
    'Aereo': 'Aereo',
    'Maritimo - Fluvial': 'Maritimo',
    'Terrestre Auto': 'Terrestre',
    'Terrestre Bus': 'Terrestre',
    'Terrestre Otros': 'Terrestre',
    'Otros': 'Terrestre',       
    'Sin Datos': 'Terrestre'     
}

df_train['Transporte Internacional de Egreso'] = df_train['Transporte Internacional de Egreso'].map(agrupacion_transporteE)

print(df_train['Transporte Internacional de Egreso'].unique())

# %% [markdown]
# #### Tipo de Viaje

# %% [markdown]
# Variable creada a partir de la concatenación de Motivo + Alojamiento, luego de las agrupaciones realizadas a ambas variables.

# %%
df_train["Tipo_Viaje"] = df_train["Motivo"] + "_" + df_train["Alojamiento"]
df_train["Tipo_Viaje"].unique()

# %%
print(df_train.shape)

# %%
list(df_train.columns)

# %%
# Exportar el DataFrame a un archivo Excel

df_train.to_excel("df_train.xlsx", index=False)

# %% [markdown]
# ### Aplicar transformaciones a df_test

# %%
# Función que aplica mapeos a df_test

def aplicar_mapeos_al_df(df_input):
    dfc = df_input.copy()
    dfc['Lugar Ingreso'] = dfc['Lugar Ingreso'].apply(lambda x: x if x in lugaresI_principales else 'Otros')
    dfc['Transporte Internacional de Ingreso'] = dfc['Transporte Internacional de Ingreso'].map(agrupacion_transporteI).fillna('Terrestre')
    dfc['Pais'] = dfc['Pais'].apply(agrupar_pais)
    dfc['Residencia'] = dfc['Residencia'].map(map_residencia_a_pais).fillna('Sin Datos')
    dfc['Motivo'] = dfc['Motivo'].replace(
        {'MICE':'Trabajo / Profesional / MICE','Trabajo / Profesional':'Trabajo / Profesional / MICE',
         'Estudios':'Otros','Religioso':'Otros','Compras':'Otros','Salud / wellness':'Otros'}
    ).fillna('Otros')
    dfc['Ocupacion'] = dfc['Ocupacion'].map(ocupacion_categoria).fillna('Otros')
    dfc['Localidad'] = dfc['Localidad'].replace(to_replace=['Sin Datos','Otros',''], value='Otros').fillna('Otros')
    dfc['Localidad'] = dfc['Localidad'].apply(lambda x: x if x in loc_principales else 'Otros')
    dfc['Departamento'] = dfc['Departamento'].apply(lambda x: x if x in deptos_principales else 'Otros').fillna('Otros')
    dfc['Alojamiento'] = dfc['Alojamiento'].map(mapa_alojamiento).fillna('Otros')
    dfc['TransporteLocal'] = dfc['TransporteLocal'].apply(lambda x: x if x in transp_principales else 'Otros').fillna('Otros')
    dfc['Lugar Egreso'] = dfc['Lugar Egreso'].apply(lambda x: x if x in lugaresE_principales else 'Otros')
    dfc['Transporte Internacional de Egreso'] = dfc['Transporte Internacional de Egreso'].map(agrupacion_transporteE).fillna('Terrestre')
    return dfc

df_test = aplicar_mapeos_al_df(df_test)

# %%
df_test["Tipo_Viaje"] = df_test["Motivo"] + "_" + df_test["Alojamiento"]
df_test["Tipo_Viaje"].unique()

# %%
print(df_test.shape)

# %%
list(df_test.columns)

# %% [markdown]
# ## Preparación de train y test

# %%
# Preparar X_train, X_test, y_train, y_test

y_train = df_train['Gasto/p/d']
X_train = df_train.drop(columns=['Gasto/p/d'])

y_test = df_test['Gasto/p/d']
X_test = df_test.drop(columns=['Gasto/p/d'])

# %%
# Columnas categóricas

categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

# Rellenar NaN y convertir a string

for col in categorical_cols:
    X_train[col] = X_train[col].fillna("missing").astype(str)
    X_test[col]  = X_test[col].fillna("missing").astype(str)

numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

print("Variables categóricas:", categorical_cols)
print("Variables numéricas:", numeric_cols)

# %%
df_train.dtypes

# %%
df_test.dtypes

# %%
# Definir función de Hit Rate

def hit_rate(y_real, y_pred, umbral):
    """
    Calcula el Hit Rate para un umbral relativo dado.
    umbral debe ser 0.10, 0.20, 0.30, etc.
    """
    return np.mean(np.abs(y_real - y_pred) / y_real <= umbral)

# %%
# Definir función de performance por grupo

def performance_por_grupo(df_test, y_real, y_pred, variable):
    
    df_eval = df_test.copy()
    df_eval["y_real"] = y_real
    df_eval["y_pred"] = y_pred
    df_eval["error_abs"] = np.abs(df_eval["y_real"] - df_eval["y_pred"])
    df_eval["error_rel"] = df_eval["error_abs"] / df_eval["y_real"]

    resultados = []

    for cat, grupo in df_eval.groupby(variable):
        mae = grupo["error_abs"].mean()
        rmse = np.sqrt(((grupo["y_real"] - grupo["y_pred"]) ** 2).mean())

        hit10 = (grupo["error_rel"] <= 0.10).mean()
        hit20 = (grupo["error_rel"] <= 0.20).mean()
        hit30 = (grupo["error_rel"] <= 0.30).mean()

        resultados.append({
            variable: cat,
            "MAE": mae,
            "RMSE": rmse,
            "Hit10": hit10,
            "Hit20": hit20,
            "Hit30": hit30
        })

    return pd.DataFrame(resultados).sort_values("MAE")

# %% [markdown]
# ## Regresión Lineal

# %%
# Preprocesamiento

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ],
    remainder='drop'
)

pipeline = Pipeline([
    ('preproc', preprocessor),
    ('model', LinearRegression())
])

# %%
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_r2_scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='r2')
cv_mae_scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='neg_mean_absolute_error')
cv_rmse_scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')

print("Resultados de validación cruzada (5 folds):")
print(f"R² promedio:  {cv_r2_scores.mean():.4f} ± {cv_r2_scores.std():.4f}")
print(f"MAE promedio: {-cv_mae_scores.mean():.4f} ± {cv_mae_scores.std():.4f}")
print(f"RMSE promedio: {-cv_rmse_scores.mean():.4f} ± {cv_rmse_scores.std():.4f}")

# %% [markdown]
# ### Train

# %%
pipeline.fit(X_train, y_train)

y_train_pred = pipeline.predict(X_train)

r2_train = r2_score(y_train, y_train_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

print("\n Resultados en Train:")
print(f"R²:   {r2_train:.4f}")
print(f"MAE:  {mae_train:.2f}")
print(f"RMSE: {rmse_train:.2f}")

# %% [markdown]
# ### Test

# %%
y_pred = pipeline.predict(X_test)

umbrales = [0.10, 0.20, 0.30]

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n Resultados en Test:")
print(f"R²:   {r2:.4f}")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

for umbral in umbrales:
    hr = hit_rate(y_test, y_pred, umbral)
    print(f"Hit Rate {int(umbral*100)}%: {hr:.4f}")

# %%
# Gráfico de residuos
 
residuals_lin = y_test - y_pred
 
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals_lin, alpha=0.3)
plt.axhline(0, linestyle='--', color='gray')
plt.ylim(-150, 250)
plt.xlabel("Valor predicho")
plt.ylabel("Residuo (real - predicho)")
plt.tight_layout()
plt.show()

 

# %% [markdown]
# ### Feature Importance

# %%
ohe = pipeline.named_steps['preproc'].named_transformers_['cat']
cat_features = ohe.get_feature_names_out(categorical_cols).tolist()
feature_names = numeric_cols + cat_features

coefs = pipeline.named_steps['model'].coef_
coef_df = pd.DataFrame({'feature': feature_names, 'coef': coefs})
coef_df = coef_df.sort_values(by='coef', ascending=False)


print("\n Top 15 coeficientes positivos:")
print(coef_df.head(15))

print("\n Top 15 coeficientes negativos:")
print(coef_df.tail(15))

# %%
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel("Real")
plt.ylabel("Predicho")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

# %%
# Tabla ANOVA

categorical_cols = ['Lugar Ingreso', 'Transporte Internacional de Ingreso', 'Pais', 'Residencia', 
                    'Motivo', 'Ocupacion', 'Localidad', 'Departamento', 'Alojamiento', 
                    'TransporteLocal', 'Lugar Egreso', 'Transporte Internacional de Egreso', 
                    'Destino', 'Temporada', 'Gasta en Alojamiento?', 'Es Transito?', 'Tipo_Viaje']

numeric_cols = ['Estadia', 'Gente', 'Mes_Ingreso', 'Mes_Egreso', 'Quarter_Year_Ingreso', 
                'Quarter_Year_Egreso', 'Personas_Dia']

formula = "Q('Gasto/p/d') ~ " + " + ".join([f"C(Q('{col}'))" for col in categorical_cols]) + " + " + " + ".join(numeric_cols)

modelo = ols(formula, data=df_train).fit()

tabla_anova = anova_lm(modelo, typ=2)
tabla_anova = tabla_anova.sort_values(by='PR(>F)')
tabla_anova

# %%
# Predicciones por grupo

y_pred_lin = pipeline.predict(X_test)

print("Regresión Lineal: desempeño por DESTINO ")
print(performance_por_grupo(df_test, y_test, y_pred_lin, "Destino"))

print("\n")

print("Regresión Lineal: desempeño por ALOJAMIENTO ")
print(performance_por_grupo(df_test, y_test, y_pred_lin, "Alojamiento"))

# %% [markdown]
# ## Catboost

# %% [markdown]
# ### Búsqueda de mejores hiperparámetros

# %% [markdown]
# #### Train

# %%
model = CatBoostRegressor(
    eval_metric='RMSE',
    cat_features=categorical_cols,
    random_seed=42,
    verbose=100,
    od_type="Iter",
    od_wait=50
)

param_dist = {
    'depth': [1, 2, 3, 4, 5, 10],
    'learning_rate': [0.001, 0.01, 0.03, 0.05, 0.1],
    'l2_leaf_reg': [1, 3, 5],
    'iterations': [2000, 2500, 3000],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bylevel': [0.5, 0.7, 1.0],
    'min_data_in_leaf': [1, 5, 10, 20],
    'bootstrap_type': ['Bernoulli']
}

scoring = {
    'r2': 'r2',
    'mae': 'neg_mean_absolute_error',
    'rmse': 'neg_root_mean_squared_error'
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=500,
    scoring=scoring,
    refit='r2',
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)

print("Mejores hiperparámetros:", random_search.best_params_)
print("Mejor R² obtenido:", random_search.best_score_)

cv_results = random_search.cv_results_

print("R² CV :", cv_results['mean_test_r2'])
print("MAE CV:", -cv_results['mean_test_mae'])
print("RMSE CV:", -cv_results['mean_test_rmse'])

# %% [markdown]
# #### Test

# %%
best_model = random_search.best_estimator_

y_pred = best_model.predict(X_test)

umbrales = [0.10, 0.20, 0.30]

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(" Evaluación en test")
print("R²:", round(r2, 2))
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))

for umbral in umbrales:
    hr = hit_rate(y_test, y_pred, umbral)
    print(f"Hit Rate {int(umbral*100)}%: {hr:.4f}")

# %% [markdown]
# ### Mejores parámetros

# %% [markdown]
# #### Train

# %%
model = CatBoostRegressor(
    eval_metric='RMSE',
    cat_features=categorical_cols,
    random_seed=42,
    verbose=100,
    od_type="Iter",
    od_wait=50
)

param_dist = {
    'depth': [10],
    'learning_rate': [0.03],
    'l2_leaf_reg': [1],
    'iterations': [2000],  
    'subsample': [1.0],
    'colsample_bylevel': [0.5],
    'min_data_in_leaf': [5],
    'bootstrap_type': ['Bernoulli']
}

scoring = {
    'r2': 'r2',
    'mae': 'neg_mean_absolute_error',
    'rmse': 'neg_root_mean_squared_error'
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=500,
    scoring=scoring,
    refit='r2',
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)

print("Mejores hiperparámetros:", random_search.best_params_)
print("Mejor R² obtenido:", random_search.best_score_)

cv = random_search.cv_results_

print("R² promedio en CV:", cv['mean_test_r2'].mean())
print("MAE promedio en CV:", -cv['mean_test_mae'].mean())
print("RMSE promedio en CV:", -cv['mean_test_rmse'].mean())


# %% [markdown]
# #### Test

# %%
best_model = random_search.best_estimator_

y_pred = best_model.predict(X_test)

umbrales = [0.10, 0.20, 0.30]

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(" Evaluación en test")
print("R²:", round(r2, 2))
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))

for umbral in umbrales:
    hr = hit_rate(y_test, y_pred, umbral)
    print(f"Hit Rate {int(umbral*100)}%: {hr:.4f}")

# %%
# Gráfico de residuos
 
residuals_cat = y_test - y_pred
 
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals_cat, alpha=0.3)
plt.axhline(0, linestyle='--', color='gray')
plt.ylim(-150, 250)
plt.xlabel("Valor predicho")
plt.ylabel("Residuo (real - predicho)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Feature importance

# %%
importances = best_model.get_feature_importance(prettified=True)

print("\nImportancia de características (CatBoost):")
print(importances)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importances', y='Feature Id', data=importances.sort_values('Importances', ascending=False))
plt.xlabel('Importancia')
plt.ylabel('Característica')
plt.tight_layout()
plt.show()

# %%
# Predicciones por grupo

y_pred_cat = best_model.predict(X_test)

print("CatBoost: desempeño por DESTINO ")
print(performance_por_grupo(df_test, y_test, y_pred_cat, "Destino"))

print("\n")

print("CatBoost: desempeño por ALOJAMIENTO ")
print(performance_por_grupo(df_test, y_test, y_pred_cat, "Alojamiento"))

# %% [markdown]
# ### Observación nueva

# %%
columnas_modelo = [
    'Lugar Ingreso',
    'Transporte Internacional de Ingreso',
    'Pais',
    'Residencia',
    'Motivo',
    'Ocupacion',
    'Localidad',
    'Departamento',
    'Alojamiento',
    'TransporteLocal',
    'Lugar Egreso',
    'Transporte Internacional de Egreso',
    'Destino',
    'Estadia',
    'Gente',
    'Mes_Ingreso',
    'Mes_Egreso',
    'Quarter_Year_Ingreso',
    'Quarter_Year_Egreso',
    'Temporada',
    'Gasta en Alojamiento?',
    'Es Transito?',
    'Personas_Dia',
    'Gasto/p/d',
    'Tipo_Viaje'
]

df_nuevo = pd.DataFrame(columns=columnas_modelo)

df_nuevo.loc[0] = {
    'Lugar Ingreso': 'Aeropuerto de Carrasco',
    'Transporte Internacional de Ingreso': 'Aereo',
    'Pais': 'Otro de America',
    'Residencia': 'Otras ciudades Sudamerica',
    'Motivo': 'Ocio y vacaciones',
    'Ocupacion': 'Profesionales y Patrones',
    'Localidad': 'Montevideo',
    'Departamento': 'Montevideo',
    'Alojamiento': 'Otros Hoteles',
    'TransporteLocal': 'Taxi - Bus',
    'Lugar Egreso': 'Aeropuerto de Carrasco',
    'Transporte Internacional de Egreso': 'Aereo',
    'Destino': 'Montevideo',
    'Estadia': 11,
    'Gente': 2,
    'Mes_Ingreso': 2,
    'Mes_Egreso': 2,
    'Quarter_Year_Ingreso': '20171',
    'Quarter_Year_Egreso': '20171',
    'Temporada': 'Alta',
    'Gasta en Alojamiento?': 'Si',
    'Es Transito?': 'No',
    'Gasto/p/d': np.nan,
}

df_nuevo['Personas_Dia'] = df_nuevo['Gente'] * df_nuevo['Estadia']
df_nuevo['Tipo_Viaje'] = df_nuevo['Motivo'] + "_" + df_nuevo['Alojamiento']
df_nuevo


# %%
X_nuevo = df_nuevo.drop(columns=['Gasto/p/d'])

prediccion = best_model.predict(X_nuevo)

print(f"Predicción estimada de Gasto/p/d: {prediccion[0]:.2f}")

# %%
y_real = [113.18]
y_pred = [prediccion[0]]

mae = mean_absolute_error(y_real, y_pred)
rmse = np.sqrt(mean_squared_error(y_real, y_pred))
r2 = r2_score(y_real, y_pred)

print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# %%
df_test.head()

# %% [markdown]
# ## Random Forest

# %% [markdown]
# ### One Hot Encoding

# %%
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_encoded = pd.DataFrame(encoder.fit_transform(X_train[categorical_cols]))
X_test_encoded = pd.DataFrame(encoder.transform(X_test[categorical_cols]))

X_train_encoded.columns = encoder.get_feature_names_out(categorical_cols)
X_test_encoded.columns = encoder.get_feature_names_out(categorical_cols)

X_train_final = pd.concat([X_train.drop(columns=categorical_cols).reset_index(drop=True),
                           X_train_encoded.reset_index(drop=True)], axis=1)
X_test_final = pd.concat([X_test.drop(columns=categorical_cols).reset_index(drop=True),
                          X_test_encoded.reset_index(drop=True)], axis=1)

# %%
X_train_final.shape

# %% [markdown]
# ### Búsqueda de mejores hiperparámetros

# %% [markdown]
# #### Train

# %%
rf = RandomForestRegressor(random_state=42, n_jobs=-1)

param_dist = {
    'n_estimators': [800, 1200, 1500],
    'max_depth': [6, 8, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 10],
    'max_features': ['sqrt', 0.2, 0.3, 0.5],
    'bootstrap': [True]
}

scoring = {
    'r2': 'r2',
    'mae': 'neg_mean_absolute_error',
    'rmse': 'neg_root_mean_squared_error'
}

rf_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=500,
    cv=5,
    scoring=scoring,
    refit='r2',
    verbose=1,
    n_jobs=-1,
    random_state=42
)

rf_search.fit(X_train_final, y_train)

print("Mejores hiperparámetros:", rf_search.best_params_)
print("Mejor R² CV:", rf_search.best_score_)

cv_results = rf_search.cv_results_
best_idx = rf_search.best_index_

print("\n Métricas de CV del mejor Random Forest:")

print(f"R² CV   : {cv_results['mean_test_r2'][best_idx]:.4f}")
print(f"MAE CV  : {-cv_results['mean_test_mae'][best_idx]:.4f}")
print(f"RMSE CV : {-cv_results['mean_test_rmse'][best_idx]:.4f}")

# %% [markdown]
# #### Test

# %%
best_rf = rf_search.best_estimator_
y_pred = best_rf.predict(X_test_final)

umbrales = [0.10, 0.20, 0.30]

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n Evaluación en test:")
print(f"R²:   {r2:.3f}")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

for umbral in umbrales:
    hr = hit_rate(y_test, y_pred, umbral)
    print(f"Hit Rate {int(umbral*100)}%: {hr:.4f}")

# %% [markdown]
# ### Mejores parámetros

# %% [markdown]
# #### Train

# %%
rf = RandomForestRegressor(random_state=42, n_jobs=-1)

param_dist = {
    'n_estimators': [1200],
    'max_depth': [None],
    'min_samples_split': [10],
    'min_samples_leaf': [1],
    'max_features': [0.3],
    'bootstrap': [True]
}

scoring = {
    'r2': 'r2',
    'mae': 'neg_mean_absolute_error',
    'rmse': 'neg_root_mean_squared_error'
}

rf_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=500,
    cv=5,
    scoring=scoring,
    refit='r2',
    verbose=1,
    n_jobs=-1,
    random_state=42
)

rf_search.fit(X_train_final, y_train)

print("Mejores hiperparámetros:", rf_search.best_params_)
print("Mejor R² CV:", rf_search.best_score_)

cv_results = rf_search.cv_results_
best_idx = rf_search.best_index_

print("\n Métricas de CV del mejor Random Forest:")

print(f"R² CV   : {cv_results['mean_test_r2'][best_idx]:.4f}")
print(f"MAE CV  : {-cv_results['mean_test_mae'][best_idx]:.4f}")
print(f"RMSE CV : {-cv_results['mean_test_rmse'][best_idx]:.4f}")

# %% [markdown]
# #### Test

# %%
best_rf = rf_search.best_estimator_
y_pred = best_rf.predict(X_test_final)

umbrales = [0.10, 0.20, 0.30]

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n Evaluación en test:")
print(f"R²:   {r2:.3f}")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

for umbral in umbrales:
    hr = hit_rate(y_test, y_pred, umbral)
    print(f"Hit Rate {int(umbral*100)}%: {hr:.4f}")

# %%
# Gráfico de residuos
 
residuals_rf = y_test - y_pred
 
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals_rf, alpha=0.3)
plt.axhline(0, linestyle='--', color='gray')
plt.ylim(-150, 250)
plt.xlabel("Valor predicho")
plt.ylabel("Residuo (real - predicho)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Feature importance

# %%
importances = pd.DataFrame({
    'Feature': X_train_final.columns,
    'Importance': best_rf.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=importances.head(20))
plt.show()

# %%
# Predicciones por grupo

y_pred_rf = best_rf.predict(X_test_final)

print("Random Forest: desempeño por DESTINO ")
print(performance_por_grupo(df_test, y_test, y_pred_rf, "Destino"))

print("\n")

print("Random Forest: desempeño por ALOJAMIENTO ")
print(performance_por_grupo(df_test, y_test, y_pred_rf, "Alojamiento"))


