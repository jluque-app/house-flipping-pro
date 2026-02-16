import pandas as pd
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

## Helpers, common variables and functions.

#CIUDAD = "Bar"

municipios_aledanos_madrid = [
    "Alcobendas",
    "Alcorcón",
    "Coslada",
    "Fuenlabrada",
    "Getafe",
    "Leganés",
    "Majadahonda",
    "Madrid",
    "Las Rozas",
    "Móstoles",
    "Paracuellos de Jarama",
    "Parla",
    "Pozuelo de Alarcón",
    "Rivas-Vaciamadrid",
    "San Fernando de Henares",
    "San Sebastián de los Reyes",
    "Torrejón de Ardoz",
    "Tres Cantos",
    "Villanueva de la Cañada",
    "Villaviciosa de Odón"
]    

ingresos_aledanos = {
    "Alcobendas": 19394,            # según elEconomista
    "Alcorcón": 17004,             # INE (renta bruta media 2021)
    "Coslada": 16329,              # INE
    "Fuenlabrada": 13896,          # INE
    "Getafe": 17279,              # INE
    "Leganés": 15658,             # INE
    "Madrid": 22587,              # INE
    "Majadahonda": 21248,         # elEconomista (renta neta per cápita)
    "Móstoles": 14875,            # INE
    "Paracuellos de Jarama": 14000,  # valor aproximado para municipio pequeño (entre 10k y 14k)
    "Parla": 11965,               # INE
    "Pozuelo de Alarcón": 27167,   # elEconomista
    "Rivas-Vaciamadrid": 20386,    # INE
    "San Fernando de Henares": 13000,  # valor aproximado para municipio de tamaño medio
    "San Sebastián de los Reyes": 20876,  # INE
    "Torrejón de Ardoz": 15313,    # INE
    "Tres Cantos": 26657,         # INE
    "Villanueva de la Cañada": 13000,  # valor aproximado (municipio pequeño)
    "Villaviciosa de Odón": 19257  # elEconomista
}


from shapely.geometry import Point, Polygon

def fix_barrio_name(name: str) -> str | None:
    """
    Corrige nombres de barrios con caracteres 'rotos' tipo 'GÃ²tic' -> 'Gòtic'.

    Intenta reinterpretar el string como latin-1 y decodificarlo como utf-8.
    Si algo falla o el nombre es None, lo deja como está.
    """
    if name is None:
        return None
    try:
        # Paso clave: reinterpretar bytes latin-1 como utf-8
        fixed = name.encode("latin1").decode("utf-8")
        return fixed
    except (UnicodeEncodeError, UnicodeDecodeError):
        # Si no funciona, devuelve el original
        return name


def _get_neighborhood(lat, lon, barrios_dict, extents, polygons):
    #if municipality != ciudad:
    #    return None #municipality
    # Pertenencia a barrio. Si no cae en ninguno se descarta.
    point = Point(lon, lat)
    for barrio in barrios_dict.keys():
        extent = extents[barrio]
        polygon = polygons[barrio]
        if lon < extent["min_lon"] or lon > extent["max_lon"] or lat < extent["min_lat"] or lat > extent["max_lat"]:
            continue
        if polygon.contains(point):
            return barrio
    return None 


def _haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r    


def normalizar(x, minimo, maximo):
    if x < minimo:
        return minimo
    elif x > maximo:
        return maximo
    else:
        return (x - minimo)/(maximo - minimo)




def supply_side_V6(NO, SC, betaO, sigmaO, x_bar, x_max, ere, kappa, eta, T=100):
    """
    Traducción de la función 'supply_side' escrita originalmente en MATLAB.
    
    Parámetros:
    -----------
    NO      : int   -> Número de pasos (equiv. a N en el código MATLAB).
    SC      : float -> Parámetro SC.
    betaO   : float -> Parámetro beta.
    sigmaO  : float -> Parámetro sigma.
    x_bar   : float -> Valor mínimo de la región (Klo en MATLAB).
    x_max   : float -> Valor máximo de la región (Kmax en MATLAB).
    ere     : float -> Parámetro ere.
    kappa   : float -> Parámetro kappa.
    eta     : float -> Parámetro eta.

    Retorna:
    --------
    F       : numpy.ndarray -> Equiv. a 'r' final (ó 'R' actualizado).
    K       : numpy.ndarray -> Vector de capitales (K).
    Kprime  : numpy.ndarray -> Policy function (K' óptimo para cada K).
    """

    # -------------------------------------------------------------------------
    # 1. Inicialización y definición de parámetros
    # -------------------------------------------------------------------------
    N = NO
    beta = betaO
    sigma = sigmaO
    
    # Klo y Kmax en MATLAB
    Klo = x_bar
    Kmax = x_max
    
    # Paso entre puntos de la malla de K
    step = (Kmax - Klo) / N
    
    # Vector de K (equivalente a K = Klo:step:Kmax en MATLAB)
    # Para incluir el extremo Kmax, ajustamos np.arange con un pequeño epsilon.
    # Otra opción es usar linspace: np.linspace(Klo, Kmax, N+1).
    K = np.arange(Klo, Kmax + 0.5*step, step)
    
    # Arreglos auxiliares
    R = np.zeros(K.shape)   # Equivale a R(1, size(K,2)) en MATLAB
    r = np.zeros(K.shape)   # Se usará para iterar en cada paso
    m = np.zeros(K.shape, dtype=int)  # Para almacenar índices del máximo
    Kprime = np.zeros(K.shape)
    
    #T = 100        # Número de iteraciones máximas
    toler = 0.001  # Tolerancia
    D = np.zeros(T)
    
    # alpha = (kappa-1)/Klo en MATLAB
    alpha = (kappa - 1) / Klo
    
    # -------------------------------------------------------------------------
    # 2. Cálculo inicial de R con un primer "while" (en MATLAB)
    #    MATLAB: while i <= size(K,2) -> for i in range(len(K))
    # -------------------------------------------------------------------------
    for i in range(len(K)):
        K1max = K[i]
        # Se arma el vector K1 = Klo:step:K1max
        # Para evitar problemas de punto flotante, ajustamos si K1max < Klo
        if K1max < Klo:
            K1 = np.array([Klo])  # Si K1max es menor que Klo, forzamos un único punto
        else:
            K1 = np.arange(Klo, K1max + 0.5*step, step)
        
        # S tendrá el mismo tamaño que K1 (o la mínima dimensión con K)
        S = np.zeros(len(K1))
        
        # Ahora llenamos S con la fórmula dada
        for j in range(len(K1)):
            # CC = alpha*eta*K(i)^(-kappa) [en el código original, lo llama CC en la parte de S]
            CC = alpha * eta * (K[i] ** (-kappa))
            # La expresión que sumas a S:
            # (ere*(x_bar-(K1(j)+SC))^(1-sigma))/(1-sigma)
            # + (eta*K1(j)*(1 - CC*(eta*K1(j))^(1+kappa)/(1+kappa))) * (beta/(1-beta))
            # + beta*(eta*(K1(j))^(1-sigma)/(1-sigma)) * (CC*K(i)^(1+kappa)/(1+kappa))
            #
            # OJO: Asegúrate de que no se produzcan valores negativos en (x_bar - (K1(j)+SC))
            # si sigma < 1. En MATLAB, podrían salir NaNs si esto es negativo.
            
            # Término 1
            term1 = (ere * (x_bar - (K1[j] + SC)) ** (1 - sigma)) / (1 - sigma)
            # Término 2
            term2 = eta * K1[j] * (
                1 - CC * (eta * K1[j]) ** (1 + kappa) / (1 + kappa)
            )
            term2 *= (beta / (1 - beta))
            # Término 3
            term3 = beta * (
                eta * (K1[j] ** (1 - sigma)) / (1 - sigma)
            ) * (CC * (K[i] ** (1 + kappa)) / (1 + kappa))
            
            # S[j] = sumatorio de term1, term2, term3
            S[j] = term1 + term2 + term3
        
        # R(i) = max(S)
        R[i] = np.max(S)

    #print(R)
    
    # -------------------------------------------------------------------------
    # 3. Iteraciones principales (loop h = 1 to T en MATLAB)
    # -------------------------------------------------------------------------
    for h in range(T):
        for i in range(len(K)):
            K1max = K[i]
            if K1max < Klo:
                K1 = np.array([Klo])
            else:
                K1 = np.arange(Klo, K1max + 0.5*step, step)
            
            s = np.zeros(len(K1))
            
            for j in range(len(K1)):
                # CC = alpha*K(i)^(-kappa)
                CC = alpha * (K[i] ** (-kappa))
                
                # s(j) = ...
                term1 = (ere * (x_bar - (K1[j] + SC)) ** (1 - sigma)) / (1 - sigma)
                term2 = eta * K1[j] * (
                    1 - CC * (K1[j] ** (1 + kappa)) / (1 + kappa)
                )
                term2 *= (beta / (1 - beta))
                term3 = beta * R[j] * (
                    CC * (K[j] ** (1 + kappa)) / (1 + kappa)
                )
                
                s[j] = term1 + term2 + term3
            
            # r(i) = max(s) y m(i) = argmax(s)
            r[i] = np.max(s)
            m[i] = np.argmax(s)
        
        # diff = (R - r) / (1 + R)
        diff = (R - r) / (1 + R)
        
        # Chequeo de convergencia usando su máximo valor absoluto
        if np.max(np.abs(diff)) <= toler:
            # Si todos los puntos han convergido dentro de la tolerancia, se rompe el loop
            break
        else:
            R = r.copy()
        
        D[h] = np.max(diff)
    
    # -------------------------------------------------------------------------
    # 4. Salida: F, K, Kprime
    #    F = r
    #    Kprime(i) = K(m(i))
    # -------------------------------------------------------------------------
    F = r.copy()
    for i in range(len(K)):
        Kprime[i] = K[m[i]]  # K' óptimo para cada K[i]
    
    return F, K, Kprime


def armar_supply(cluster):

    num_points= 100
    capital_max_value = 12000 # Fijarse un máximo "razonable".
    capital_min_value = clusters_stats[cluster]["x_min"]
    kappa_param = clusters_stats[cluster]["kappa"]
    beta_param = 0.99
    ere_param = 0.05
    eta_param = 1 # Fixed. # VER SI SACARLO.
    sigma_param = 5.0
    fixed_cost = 0.0
    _, k_arr, k_prima = supply_side_V6(NO=num_points, SC=0.1, betaO=beta_param, 
                                       sigmaO=sigma_param, x_bar=capital_min_value, 
                                       x_max=capital_max_value, ere=ere_param, kappa=kappa_param, eta=eta_param)
    def supply(x):
        indice_cercano = np.abs(k_arr - x).argmin()
        # Usar ese índice para encontrar el valor correspondiente en k_prima
        precio = k_prima[indice_cercano]
        return precio
    return supply

def armar_supply_normalizado(cluster, clusters_stats, sigma=5.0, points=100, iteraciones=100):

    MIN_VALUE = 0.01
    MAX_EUROS = clusters_stats[cluster]["x_max"]
    
    num_points= points
    #capital_max_value = 12000 # Fijarse un máximo "razonable".
    #capital_min_value = clusters_stats[cluster]["x_min"]
    capital_min_value = MIN_VALUE
    capital_max_value = 1.0
    
    kappa_param = clusters_stats[cluster]["kappa"]
    beta_param = 0.99
    ere_param = 0.05
    eta_param = 1 # Fixed. # VER SI SACARLO.
    sigma_param = sigma
    fixed_cost = 0.0
    _, k_arr, k_prima = supply_side_V6(NO=num_points, SC=0.1, betaO=beta_param, 
                                       sigmaO=sigma_param, x_bar=capital_min_value, 
                                       x_max=capital_max_value, ere=ere_param, kappa=kappa_param, eta=eta_param,
                                       T=iteraciones)
    def supply(x):
        if x < clusters_stats[cluster]["x_min"]:
            return np.nan
        if x > MAX_EUROS:
            x_norm = 1.0
        x_norm = MIN_VALUE + (x - clusters_stats[cluster]["x_min"])/(MAX_EUROS - clusters_stats[cluster]["x_min"])        
        indice_cercano = np.abs(k_arr - x_norm).argmin()
        # Usar ese índice para encontrar el valor correspondiente en k_prima
        precio_normalizado = k_prima[indice_cercano]

        precio = (precio_normalizado - MIN_VALUE)*(MAX_EUROS - clusters_stats[cluster]["x_min"]) + clusters_stats[cluster]["x_min"]
        return precio
        
    return supply

def armar_funcion_demanda(gamma=0.5, size_factor_h=1.0, size_factor_ingreso=1.0, debug=False):
    ahorcados = [(gamma**((1.0/gamma)*(1.0 - 1.0/gamma)))*theta**(1.0/gamma + 1.0) for theta in thetas]
    
    #ahorcado = (gamma**((1.0/gamma)*(1.0 - 1.0/gamma)))*slope**(1.0/gamma + 1.0)
    epsilon_1 = (1.0/gamma)*(gamma - 2.0 + 1.0/gamma)
    epsilon_2 = (1.0/gamma)*(1.0/gamma - 1.0)
    epsilon_3 = 1.0/gamma

    print("Epsilon 1", epsilon_1)
    print("Epsilon 2", epsilon_2)
    print("Epsilon 3", epsilon_3)
    print("Ahorcado", ahorcados)
    
    factor_ingreso = []
    for cluster in clusters:
        factor_ingreso.append(0.0)
        for barrio in barrios:
            factor_ingreso[cluster] += dicc_cluster_barrio[(cluster, barrio)]*((ingresos[barrio]*size_factor_ingreso)**epsilon_1)
    #print(factor_ingreso)

    def demanda(x, cluster):
        h = (sizes_cluster[cluster]*size_factor_h)**epsilon_2
        return factor_ingreso[cluster]*h*(x**epsilon_3)*ahorcados[cluster]
    return demanda



def armar_funcion_demanda_normalizado(clusters, barrios, dicc_cluster_barrio, ingresos, thetas_normalizado, sizes_cluster, min_price, max_price, gamma=0.5, size_factor_h=1.0, size_factor_ingreso=1.0, debug=False):
    
    # el Theta debería estar calcularse con valores normalizados...
    #min_theta = 0.0 #min(thetas)
    #max_theta = max(thetas)
    min_ingreso = 0.0 #min([ingresos[barrio] for barrio in ingresos.keys()])
    max_ingreso = max([ingresos[barrio] for barrio in ingresos.keys()])
    min_size = 0.0 #min(sizes_cluster)
    max_size = max(sizes_cluster)
    
    #thetas_normalizados = thetas #thetas_normalizados = [normalizar(theta, min_theta, max_theta) for theta in thetas]
    thetas_normalizados = thetas_normalizado
    #sizes_normalizados = [normalizar(size, min_size, max_size) for size in sizes_cluster]
    sizes_normalizados = sizes_cluster
    ingresos_normalizados = {}
    for barrio in ingresos.keys():
        ingresos_normalizados[barrio] = normalizar(ingresos[barrio], min_ingreso, max_ingreso)
    
    
    ahorcados = [(gamma**((1.0/gamma)*(1.0 - 1.0/gamma)))*theta**(1.0/gamma + 1.0) for theta in thetas_normalizados]
    
    #ahorcado = (gamma**((1.0/gamma)*(1.0 - 1.0/gamma)))*slope**(1.0/gamma + 1.0)
    epsilon_1 = (1.0/gamma)*(gamma - 2.0 + 1.0/gamma)
    epsilon_2 = (1.0/gamma)*(1.0/gamma - 1.0)
    epsilon_3 = 1.0/gamma
    if debug:
        #print("Epsilon 1", epsilon_1)
        #print("Epsilon 2", epsilon_2)
        #print("Epsilon 3", epsilon_3)
        #print("Ahorcado", ahorcados)
        pass
    
    factor_ingreso = []
    for cluster in clusters:
        factor_ingreso.append(0.0)
        for barrio in barrios:
            factor_ingreso[cluster] += dicc_cluster_barrio[(cluster, barrio)]*((ingresos_normalizados[barrio]*size_factor_ingreso)**epsilon_1)
    #print(factor_ingreso)

    def demanda(x, cluster):
        #print("Evaluando",x)
        x_norm = normalizar(x, 0.0, max_price) #normalizar(x, min_price, max_price)
        #print("Normalizado", x_norm)
        #print(f"x {x} x_norm {x_norm} min_price {min_price} max_price {max_price}")
        h = (sizes_normalizados[cluster]*size_factor_h)**epsilon_2
        #print("H", h)
        #print(f"Factor ingreso {factor_ingreso[cluster]} H {h} x {x} x_norm {x_norm} Epsilon_3 {epsilon_3} Ahorcado {ahorcados[cluster]}")
        #print(f"x_norm**epsilon_3 {x_norm**epsilon_3}")
        #print(f"factor_ingreso[cluster]*h*(x_norm**epsilon_3) {factor_ingreso[cluster]*h*(x_norm**epsilon_3)}")
        value = factor_ingreso[cluster]*h*(x_norm**epsilon_3)
        #print(f"value*ahorcados[cluster] {value*ahorcados[cluster]}")
        return value*ahorcados[cluster]
    return demanda



### STEP 1 --- Loading of data


def initial_load(input_path="./data/raw/FotocasaINE.dta", output_path="./data/raw/data_buy.pkl"):
    df = pd.read_stata(input_path)
    df_buy = df[df["operation"]=="buy"]
    df_buy.to_pickle(output_path)


### STEP 2 --- Handling of neighborhood.

def handle_neighborhood(input_path="./data/raw/data_buy.pkl", shp_path='./data/Madrid/Barrios.shp', output_path="./data/paper/data_madrid_barrios.pkl", municipios=municipios_aledanos_madrid, campo_barrio="NOM_BARRI", ciudad=None, origen="epsg:32631"):
    df = pd.read_pickle(input_path)
    print("Registros", len(df))
    #df_aled = df[df.municipality.isin(municipios)]

    if not ciudad is None:
        df = df[df.municipality==ciudad]
    print("Registros", len(df))
    
    import geopandas as gpd
    from shapely.geometry import mapping
    import json
    from pyproj import Transformer
    
    # Cargar el archivo SHP usando geopandas
    gdf = gpd.read_file(shp_path, encoding="latin1")
    gdf = gdf.to_crs(epsg=4326)
    
    # Crear un transformador para convertir de UTM a WGS84
    # Supongamos que las coordenadas están en el sistema UTM zona 30N (EPSG:25830)
    #transformer = Transformer.from_crs("epsg:25830", "epsg:4326", always_xy=True)
    #transformer = Transformer.from_crs(origen, "epsg:4326", always_xy=True)
    # Lo pasás directamente a WGS84
    
    # Crear un diccionario con el nombre del barrio y las coordenadas
    barrios_dict = {}
    
    for _, row in gdf.iterrows():
        #print(row)
        nombre_barrio = row[campo_barrio]  # Ajusta según el campo del nombre del barrio en tu archivo
        geometria = row['geometry']
        print(nombre_barrio)
        # Verificar si la geometría es un polígono o multipolígono
        if geometria.geom_type == 'Polygon':
            coords = list(geometria.exterior.coords)
            # Convertir las coordenadas a latitud y longitud
            #coords = [transformer.transform(x, y) for x, y in coords]
            barrios_dict[nombre_barrio] = coords
        elif geometria.geom_type == 'MultiPolygon':
            coords = []
            #print(geometria)
            for polygon in geometria.geoms:
                polygon_coords = list(polygon.exterior.coords)
                # Convertir las coordenadas a latitud y longitud
                #polygon_coords = [transformer.transform(x, y) for x, y in polygon_coords]
                coords.extend(polygon_coords)
            barrios_dict[nombre_barrio] = coords

    extents = {}
    
    for barrio in barrios_dict.keys():
        coords = barrios_dict[barrio]
        min_lon = np.min([coord[0] for coord in coords])
        max_lon = np.max([coord[0] for coord in coords])
        min_lat = np.min([coord[1] for coord in coords])
        max_lat = np.max([coord[1] for coord in coords])
        extents[barrio] = {
            "min_lon": min_lon,
            "min_lat": min_lat,
            "max_lon": max_lon,
            "max_lat": max_lat,
        }
        print(extents[barrio])
    
    from shapely.geometry import Point, Polygon
    
    polygons = {}
    
    for barrio in barrios_dict.keys():
        coords = barrios_dict[barrio]
        polygons[barrio] = Polygon(coords)
    
    
    #df = df_aled

    def fix_barrio_name(name: str) -> str | None:
        """
        Corrige nombres de barrios con caracteres 'rotos' tipo 'GÃ²tic' -> 'Gòtic'.
    
        Intenta reinterpretar el string como latin-1 y decodificarlo como utf-8.
        Si algo falla o el nombre es None, lo deja como está.
        """
        if name is None:
            return None
        try:
            # Paso clave: reinterpretar bytes latin-1 como utf-8
            fixed = name.encode("latin1").decode("utf-8")
            return fixed
        except (UnicodeEncodeError, UnicodeDecodeError):
            # Si no funciona, devuelve el original
            return name

    """
Elementos de barrios fuera de ingresos {'Sants - Badal', 'el Poble-sec', 'la Sagrada Família', 'Pedralbes', 'el Turó de la Peira', 'Provençals del Poblenou', 'la Trinitat Vella', 'Vilapicina i la Torre Llobeta', "la Vall d'Hebron", 'Vallvidrera, el Tibidabo i les Planes', 'la Font de la Guatlla'}

Elementos de ingresos fuera de barrios {'el Bon Pastor i Baró de Viver', "Vall d'Hebron", 'Turó de la Peira', 'el Poble Sec', 'Sagrada Família', 'Verdun (Verdún)', 'el Congrés', 'Eixample', 'Trinitat Nova', 'Trinitat Vella', 'Bonanova', 'Prosperitat', 'les Planes'}    
"""

    
    df["barrio"] = df[["latitude", "longitude"]].apply(lambda x: fix_barrio_name(_get_neighborhood(x[0], x[1], barrios_dict, extents, polygons)),axis=1)
    
    df.to_pickle(output_path)




## STEP 4 - CLUSTERING

def construct_clusters(input_file="./data/paper/data_madrid_google.pkl", demo_file='./data/Madrid/ConsolidadoDemográfico.xlsx', output_file="./data/paper/data_madrid_cluster.pkl", ingresos=None):
    import pandas as pd
    df = pd.read_pickle(input_file)
    df_global = df.copy()
    print("Registros",len(df))
    
    if ingresos is None:
        # Ruta al archivo Excel
        file_path = demo_file
        
        # Cargar el archivo Excel
        xls = pd.ExcelFile(file_path)
        
        # Mostrar las hojas del archivo Excel
        print(xls.sheet_names)
        
        # Supongamos que los datos están en la primera hoja
        df_raw = xls.parse(xls.sheet_names[0])
        
        # Inicializar una lista para almacenar los datos procesados
        data = []
        
        # Iterar sobre las columnas del DataFrame para extraer los datos necesarios
        for i in range(0, len(df_raw.columns), 2):
            barrio = df_raw.iloc[0, i]
            ingreso_medio_hogar = df_raw.iloc[53, i+1]
            ingreso_medio_persona = df_raw.iloc[54, i+1]
            
            data.append({
                'Barrio': barrio,
                'Ingreso Medio por Hogar': ingreso_medio_hogar,
                'Ingreso Medio por Persona': ingreso_medio_persona
            })

    else:
        data = []
        for barrio in ingresos.keys():
            data.append({
                "Barrio": barrio,
                "Ingreso Medio por Persona": ingresos[barrio],
            })
    
    # Crear un DataFrame con los datos procesados
    df_econ = pd.DataFrame(data)
    
    # Mostrar el DataFrame resultante
    # Le agregamos los barrios aledaños a Madrid
    import copy
    ingresos = copy.copy(ingresos_aledanos) 
    
    for i in range(len(df_econ)):
        barrio = df_econ.iloc[i]
        ingresos[barrio["Barrio"]] = barrio["Ingreso Medio por Persona"]    
    barrios = []
    ingresos_persona = []    

    for barrio in ingresos.keys():
        barrios.append(barrio)
        ingresos_persona.append(ingresos[barrio])
    df_econ = pd.DataFrame({
        "Barrio": barrios,
        "Ingreso Medio por Persona": ingresos_persona
    })
    df = df.merge(df_econ, left_on="barrio", right_on="Barrio", how="left")

    features = [
     'size',
     'bedrooms',
     'bathrooms',
     'lift',
     'garage',
     'storage',
     'terrace',
     'air_conditioning',
     'swimming_pool',
     'garden',
     'sports',
     #'status',
     #'new_construction',
     'rating_leads',
     'rating_visits',
     'floor_int',
    ]

    def encode_floor(floor_value):
        floor_mapping = {
            '1º': 1,
            '2º': 2,
            '3º': 3,
            '4º': 4,
            '5º': 5,
            '6º': 6,
            '7º': 7,
            '8º': 8,
            '9º': 9,
            '10º': 10,
            '11º': 11,
            '12º': 12,
            '13º': 13,
            '14º': 14,
            '15º': 15,
            'A partir del 15º': 16,  # Puedes ajustar si necesitas un valor específico
            'Bajos': 0,              # Planta baja
            'Entresuelo': 0.5,       # Intermedio entre baja y primer piso
            'Principal': 0.5,        # Similar a entresuelo, pero podrías ajustar el valor si lo prefieres
            'Sótano': -1,            # Sótano
            'Subsótano': -2,         # Subsótano
            'Otro': 0,            # Puedes usar None o cualquier otro valor para categorías desconocidas
            '': 0                 # Para valores faltantes
        }
        
        return floor_mapping.get(floor_value, None)  # Devuelve None si no está en el mapeo

    df_1 = df.drop_duplicates(subset="id", keep="last")
    df_1["floor_int"] = df_1.floor.apply(encode_floor)
    dg = df_1[features]

    from sklearn.preprocessing import OneHotEncoder

    # Identifica las columnas categóricas
    categorical_features = [feature for feature in features if "object" in str(dg[feature].dtype) or "cate" in str(dg[feature].dtype)]
    
    print(categorical_features)
    
    # Aplica OneHotEncoding
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    X_categorical = encoder.fit_transform(dg[categorical_features])
    
    print(X_categorical.shape)
    
    # Convierte a DataFrame para combinar con los datos numéricos
    X_categorical_df = pd.DataFrame(X_categorical, columns=encoder.get_feature_names_out(categorical_features))
    
    print(X_categorical_df.shape)
    
    
    # Combina los datos categóricos codificados con los datos numéricos restantes
    X_numeric = dg.drop(columns=categorical_features)
    print(X_numeric.shape)
    #X_combined = pd.concat([X_numeric, X_categorical_df], axis=1)
    X_combined = X_numeric

    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    
    # Imputación de valores faltantes
    imputer = SimpleImputer(strategy='median')  # Puedes cambiar la estrategia a 'median', 'most_frequent', etc.
    X_imputed = imputer.fit_transform(X_combined)
    
    # Estandarización de los datos imputados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Reducción de dimensionalidad con PCA
    pca = PCA(n_components=0.95)  # Mantén el 95% de la varianza
    X_pca = pca.fit_transform(X_scaled)
    
    print(len(X_pca))
    
    # Aplicación de KMeans
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(X_pca)
    
    # Añadir el resultado de los clusters al DataFrame original
    df_1['Cluster'] = clusters

    # Calcular la mediana del precio para cada cluster
    cluster_median_price = df_1.groupby("Cluster")["price"].median()
    
    # Ordenar los clusters por el valor mediano y reasignar los números de 0 a 4
    sorted_clusters = cluster_median_price.sort_values().index
    new_cluster_labels = {old_label: new_label for new_label, old_label in enumerate(sorted_clusters)}
    
    # Crear una copia del dataframe con los clusters re-etiquetados
    df_1["Cluster"] = df_1["Cluster"].map(new_cluster_labels)

    clusters = sorted(list(df_1.Cluster.unique()))
    prices = [df_1[df_1.Cluster == c].price for c in clusters]
    plt.boxplot(prices, labels=clusters, showfliers=False)
    plt.xlabel("Cluster")
    plt.ylabel("Price (euros)")
    plt.show()

    df = df.merge(df_1[["id", "Cluster"]], on="id", how="left")

    df.to_pickle(output_file)


## STEP 5 - SUPPLY SIDE CALIBRATION


def supply_side_calib(input_file="./data/paper/data_madrid_cluster.pkl", output_file="./data/paper/data_Madrid_probs_propietario.pkl", cluster_stats_file="./data/paper/clusters_stats.pkl"):


    
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from importlib import reload
    import powerlaw
    from scipy.stats import lognorm
    from tqdm import tqdm

    target = "price_m2"
    #target = "price"
    df = pd.read_pickle(input_file)
    # Eliminamos duplicados y pasamos a propiedades en venga.
    if False: #True:
        df = df.drop_duplicates(subset="id", keep="last")
    
    
    df = df.dropna(subset="Cluster")
    clusters = df.Cluster.unique().tolist()
    clusters.sort()
    data = df[target]

    dg = df.groupby("id")["fecha"].count().reset_index()

    dh = df.groupby("id")[target].std().reset_index()
    dh = dg.merge(dh, on="id", how="left")
    dh = dh.rename(columns = {
        "fecha": "conteo",
        "price_m2": "std_price_m2",
    })
    
    total_datos = len(dh)
    
    conteos = sorted(list(dh["conteo"].unique()))
    frecuencia = {}
    for conteo in conteos:
        frecuencia[conteo] = len(dh[dh.conteo == conteo])/total_datos


    text = """
    \\begin{table}[h!] 
    \\label{spells}
    \\begin{tabular}{|c|c|c|c|c|c|}
    \\hline
    \\textbf{Spell (in months) / Cluster} & \\textbf{0} & \\textbf{1} & \\textbf{2} & \\textbf{3} & \\textbf{4} \\\\ \\hline
    
    """
    
    for freq in range(1,6):
        if freq < 5:
            text += "\\textbf{" + str(freq) + "} &"
        else:
            text += "\\textbf{5 or more} &"
    
        for cluster in clusters:
            dh = df[df.Cluster == cluster].groupby("id")[target].std().reset_index()
            ids = set(df[df.Cluster == cluster]["id"].unique())
            dh = dg[dg["id"].isin(ids)].merge(dh, on="id", how="left")
            dh = dh.rename(columns = {
                "fecha": "conteo",
                "price_m2": "std_price_m2",
            })
            
            total_datos = len(dh)
    
            
            
            conteos = sorted(list(dh["conteo"].unique()))
            frecuencia = {}
            for conteo in conteos:
                if conteo < 5:
                    frecuencia[conteo] = len(dh[dh.conteo == conteo])/total_datos
                else:
                    frecuencia[conteo] = len(dh[dh.conteo >= conteo])/total_datos
            if cluster != 4:
                text += str(round(frecuencia[freq],2)) + " &"
            else:
                text += str(round(frecuencia[freq],2)) + "\\\\ \\hline \n"
    text += """
    \\end{tabular}
    \\caption{Empirical distribution of spells across clusters.}
    \\label{table:cal_1}
    \\end{table}
    """
    print(text)

    ## SUPPLY

    target = "price_m2"
    #target = "price"
    df = pd.read_pickle(input_file)
    # Eliminamos duplicados y pasamos a propiedades en venga.
    if True:
        df = df.drop_duplicates(subset="id", keep="last")
    
    
    df = df.dropna(subset="Cluster")
    clusters = df.Cluster.unique().tolist()
    clusters.sort()
    data = df[target]

    datos = []
    for cluster in clusters:
        precios = df[df.Cluster == cluster].price_m2
        datos.append(precios)
        print(f"cluster-", np.percentile(precios,95.0))
        
    plt.boxplot(datos, showfliers=False)
    plt.show()

    datos = []
    for cluster in clusters:
        precios = df[df.Cluster == cluster].price
        datos.append(precios)
        print(f"cluster-", np.percentile(precios,95.0))
        
    plt.boxplot(datos, showfliers=False)
    plt.show()



    def generate_power_law_data(x_min, alpha, size, seed=42):
        """
        Genera datos aleatorios de una distribución de ley de potencias.
        
        Parámetros:
        - x_min: el valor mínimo (x_min) de la distribución.
        - alpha: el exponente de la ley de potencias.
        - size: la cantidad de datos a generar.
        
        Retorna:
        - Una lista o array de datos generados.
        """
        np.random.seed(seed)
        u = np.random.uniform(0, 1, size)  # Generar datos uniformemente distribuidos
        data = x_min * (1 - u) ** (-1 / (alpha - 1))  # Transformar con la inversa de la CDF
        return data
    
    
    def power_law_mean_std(x_min, alpha):
        """
        Calcula la media y la desviación estándar de una distribución de ley de potencias.
    
        Parámetros:
        - x_min: valor mínimo de la distribución (x_min).
        - alpha: exponente de la ley de potencias.
    
        Retorna:
        - media: valor de la media (o np.inf si no está definida).
        - desviación estándar: valor de la desviación estándar (o np.inf si no está definida).
        """
        # Calcular la media
        if alpha <= 2:
            media = np.inf  # La media no está definida
        else:
            media = (alpha * x_min) / (alpha - 1)
        
        # Calcular la varianza
        if alpha <= 3:
            varianza = np.inf  # La varianza no está definida
        else:
            varianza = (alpha * x_min**2) / ((alpha - 1)**2 * (alpha - 2))
        
        # Desviación estándar
        desviacion_estandar = np.sqrt(varianza) if varianza != np.inf else np.inf
        
        return media, desviacion_estandar
    
    
    def calibrar(df, disc_kappa=10, disc_x_min=10, min_kappa=2.0, max_kappa=4.0, min_x_min=1000.0, max_x_min=5000.0):
        # CALCULO DE FRECUENCIA Y TODO LO QUE VIENE DESPUES.
        df = df.dropna(subset="Cluster")
        clusters = df.Cluster.unique().tolist()
        clusters.sort()
        data = df[target]
        dg = df.groupby("id")["fecha"].count().reset_index()
        
        dh = df.groupby("id")[target].std().reset_index()
        dh = dg.merge(dh, on="id", how="left")
        dh = dh.rename(columns = {
            "fecha": "conteo",
            "price_m2": "std_price_m2",
        })
        
        total_datos = len(dh)
        
        conteos = sorted(list(dh["conteo"].unique()))
        frecuencia = {}
        for conteo in conteos:
            frecuencia[conteo] = len(dh[dh.conteo == conteo])/total_datos
            
        
        #print(frecuencia)
        kappas = np.linspace(start=min_kappa, stop=max_kappa, num=disc_kappa)
        x_mins = np.linspace(start=min_x_min, stop=max_x_min, num=disc_x_min)
        
        tabla = np.zeros((len(kappas), len(x_mins)))
        frecs = list(frecuencia.keys())
        probabilities = list(frecuencia.values())
        probabilities = np.array(probabilities) / sum(probabilities)
    
        mean_precios = df["price_m2"].mean()
        std_precios = df["price_m2"].std()
    
        import tqdm
        
        best_kappa = None
        best_x_min = None
        best_error = None
        best_mean = None
        best_std = None
    
        MAX_EUROS = np.percentile(df["price_m2"].values,95.0)
        print(f"MAX EUROS {MAX_EUROS}")
        
        for i in tqdm.tqdm(range(len(kappas))):
            for j in range(len(x_mins)):
                kappa = kappas[i]
                x_min = x_mins[j]
                # 1) Sampleo 100 precios con la power-law con kappa y x_min.
                # 2) Distribuyo esos 100 precios en la tabla de frecuencia.
                # 3) Los que tienen frecuencia 1 ya está, nada que hacer.
                # 4) Los que tienen más frecuencia que 1, requiere el pasaje por supply side.
                #    - Grillar entre un x_min "bajo" y un x_max "alto".
                # 5) Agarro cada uno de los precios simulados por la power-law, que no coincidirán con la grilla.
                # 6) Alternativa "fácil", le asigno el precio de la grilla más cercano.
        
                precios = generate_power_law_data(x_min, kappa, 100, seed=42)
                periodos = np.random.choice(frecs, size=100, p=probabilities)
    
                MIN_VALUE = 0.01
    
        
                num_points= 100
                capital_min_value = MIN_VALUE
                capital_max_value = 1.0
        
                kappa_param = kappa
                beta_param = 0.99
                ere_param = 0.05
                eta_param = 1 # Fixed. # VER SI SACARLO.
                sigma_param = 5.0
                fixed_cost = 0.0
                iteraciones = 100
                _, k_arr, k_prima = supply_side_V6(NO=num_points, SC=0.1, betaO=beta_param, 
                                           sigmaO=sigma_param, x_bar=capital_min_value, 
                                           x_max=capital_max_value, ere=ere_param, kappa=kappa_param, eta=eta_param,
                                           T=iteraciones)
    
                
                def supply(x):
                    if x == np.nan:
                        return np.nan
                    if x < x_min:
                        return np.nan
                    if x > MAX_EUROS:
                        x_norm = 1.0
                    x_norm = MIN_VALUE + (x - x_min)/(MAX_EUROS - x_min)        
                    indice_cercano = np.abs(k_arr - x_norm).argmin()
                    # Usar ese índice para encontrar el valor correspondiente en k_prima
                    precio_normalizado = k_prima[indice_cercano]
            
                    precio = (precio_normalizado - MIN_VALUE)*(12000 - x_min) + x_min
                    return precio
                
                # Convierto cada precio simulado en un precio iterado por k --> k' dada por supply_side, asignándolo el más cercano.
                new_data = []
                for k in range(len(precios)):
                    per = periodos[k]
                    if per == 1:
                        new_data.append(precios[k])
                    else:
                        precio = precios[k]
                        for l in range(per-1):
                            precio = supply(precio)
                            new_data.append(precio)
                media = np.mean(new_data)
                std = np.std(new_data)
                #media_teorica, std_teorico = power_law_mean_std(alpha, kappa)
                #error = (media - media_teorica)** 2 + (std - std_teorico)**2
                error = (media - mean_precios)** 2 + (std - std_precios)**2
                tabla[i,j] = error
                if best_error is None or error < best_error:
                    best_error = error
                    best_kappa = kappa
                    best_x_min = x_min
                    best_mean = media
                    best_std = std
                #print("Kappa", kappa, "x_min", x_min, "Media", media, "STD", std, "ERROR", error)
                            
        return best_error, best_kappa, best_x_min, mean_precios, best_mean, std_precios, best_std

    clusters_stats = []
    for cluster in clusters:
        print(f"CLUSTER {cluster}")
        dg = df[df.Cluster == cluster]
        MAX_EUROS = np.percentile(dg["price_m2"].values,95.0)
        error, kappa, x_min, mean_precios, best_mean, std_precios, best_std = calibrar(dg)
        cluster_stats = {
            "kappa": kappa,
            "x_min": x_min,
            "x_max": MAX_EUROS,
            "true_mean": mean_precios,
            "best_mean": best_mean,
            "true_std": std_precios,
            "best_std": best_std
        }
        print(cluster_stats)
        clusters_stats.append(cluster_stats)

    import pickle
    f = open(cluster_stats_file, "wb")
    pickle.dump(clusters_stats, f)
    f.close()    

    ## Tabla LaTeX
    
    
    
    text = """
    \\begin{table}[h!]
    \\begin{tabular}{|c|c|c|c|c|c|}
    \\hline
    \\textbf{Parameter / Cluster} & \\textbf{0} & \\textbf{1} & \\textbf{2} & \\textbf{3} & \\textbf{4} \\\\ \\hline
    
    """
    
    text += "\\textbf{$\\kappa$} & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(clusters_stats[cluster]["kappa"],2)) + " & "
        else:
            text += str(round(clusters_stats[cluster]["kappa"],2)) + " \\\\ \\hline \n"
    
    text += "\\textbf{$\\underline{p}$} & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(clusters_stats[cluster]["x_min"],2)) + " & "
        else:
            text += str(round(clusters_stats[cluster]["x_min"],2)) + " \\\\ \\hline \n"
    
    text += "\\textbf{$\\overline{p}$} & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(clusters_stats[cluster]["x_max"],2)) + " & "
        else:
            text += str(round(clusters_stats[cluster]["x_max"],2)) + " \\\\ \\hline \n"
    
    text += """
    \\end{tabular}
    \\caption{Calibrated parameters. Supply side}
    \\label{table:cal_2}
    \\end{table}
    """
    print(text)

    text = """
    \\begin{table}[h!]
    \\begin{tabular}{|c|c|c|c|c|c|}
    \\hline
    \\textbf{Parameter / Cluster} & \\textbf{0} & \\textbf{1} & \\textbf{2} & \\textbf{3} & \\textbf{4} \\\\ \\hline
    
    """
    
    text += "\\textbf{Empirical Mean} & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(clusters_stats[cluster]["true_mean"],2)) + " & "
        else:
            text += str(round(clusters_stats[cluster]["true_mean"],2)) + " \\\\ \\hline \n"
    
    text += "\\textbf{Simulated Mean} & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(clusters_stats[cluster]["best_mean"],2)) + " & "
        else:
            text += str(round(clusters_stats[cluster]["best_mean"],2)) + " \\\\ \\hline \n"
    
    text += "\\textbf{Empirical SD} & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(clusters_stats[cluster]["true_std"],2)) + " & "
        else:
            text += str(round(clusters_stats[cluster]["true_std"],2)) + " \\\\ \\hline \n"
    
    text += "\\textbf{Simulated SD} & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(clusters_stats[cluster]["best_std"],2)) + " & "
        else:
            text += str(round(clusters_stats[cluster]["best_std"],2)) + " \\\\ \\hline \n"
    
    text += """
    \\end{tabular}
    \\caption{Calibrated parameters. Supply side}
    \\label{table:cal_2}
    \\end{table}
    """
    print(text)

    ## SUPPLY SIDE -- PERCEIVED PROBABILITY
    
    
    def compute_probability_simulated(x_min, x_max, alpha, x1, x2, iterations=10000, seed=42 ):
        if x_max is None:
            replicaciones = generate_power_law_data(x_min, alpha, iterations, seed)
            exitos = len([data for data in replicaciones if data >= x1 and data <= x2])
            return exitos/iterations
        else:
            replicaciones = generate_power_law_data(x_min, alpha, iterations, seed)
            denominator = len([data for data in replicaciones if data >= x_min and data <= x_max])
            numerator = len([data for data in replicaciones if data >= x_min and data <= x_max and data >= x1 and data <= x2])
            return numerator/denominator    


    target = "price_m2"
    #target = "price"
    df = pd.read_pickle(input_file)
    # Eliminamos duplicados y pasamos a propiedades en venga.
    #if False: #True:
    df = df.drop_duplicates(subset="id", keep="first")
    
    
    df = df.dropna(subset="Cluster")
    clusters = df.Cluster.unique().tolist()
    clusters.sort()
    data = df[target]

    
    columnas = 10
    
    from tqdm import tqdm
    
    probs_totales = []
    ids = []
    dhs = []
    
    #clusters = [4]
    
    for cluster in clusters:
        dg = df[df.Cluster == cluster]
        supply = armar_supply_normalizado(cluster, clusters_stats, sigma=5.0, points=100, iteraciones=100)
        x_min = clusters_stats[cluster]["x_min"]
        x_max = clusters_stats[cluster]["x_max"]
        kappa = clusters_stats[cluster]["kappa"]
        assert x_min <= x_max
        ids_cluster = []
        ids.append(ids_cluster)
        probs_cluster = []
        probs_totales.append(probs_cluster)
        for i in tqdm(range(len(dg))):
            #print(i)
            inmueble = dg.iloc[i]
            ids_cluster.append(inmueble["id"])
            price_m2 = inmueble["price_m2"]
            probs_inmuebles = []
            probs_cluster.append(probs_inmuebles)
            techo = price_m2
            piso = supply(techo)
            prob = 1.0
            for j in range(columnas):
                #assert piso <= techo
                #assert techo >= x_min
                #print(f"PRE x_min: {x_min}, techo: {techo}, kappa: {kappa}, piso: {piso}")
                if piso == np.nan or techo < x_min:
                    probs_inmuebles.append(None)
                    #print("PROBLEMA")
                    continue
                if prob == 0.0 or piso >= techo:
                    #print(f"x_min: {x_min}, techo: {techo}, kappa: {kappa}, piso: {piso}, prob: {prob}")
                    #1/0
                    probs_inmuebles.append(0.0)
                    continue
                #print(f"x_min: {x_min}, techo: {techo}, kappa: {kappa}, piso: {piso}")
                prob = compute_probability_simulated(x_min = x_min, x_max=techo, alpha=kappa, x1=piso, x2=techo)
                #print(f"Prob {prob}")
                probs_inmuebles.append(prob)
                techo = piso
                piso = supply(piso)
        dh = pd.DataFrame({
            "id": ids_cluster,
        })    
        for j in range(columnas):
            probs_columna = [prob[j] for prob in probs_cluster]
            dh[f"prob_propietario_{j}"] = probs_columna
        dhs.append(dh)
    
    dh = pd.concat(dhs)
    df = df.merge(dh, on="id", how="left")

    for cluster in clusters:
        dh = df[df.Cluster == cluster]
        plt.boxplot(dh["prob_propietario_0"].dropna())
        plt.title(str(cluster))
        plt.show()
        
        plt.hist(dh["prob_propietario_0"].values)
        plt.title(str(cluster))
        plt.show()
    
        plt.hist(dh["price_m2"].values)
        plt.title(str(cluster))
        plt.show()
    
        supply = armar_supply_normalizado(cluster, clusters_stats, sigma=5.0, points=100, iteraciones=100)
        X = np.linspace(start=clusters_stats[cluster]["x_min"], stop=10000, num=1000)
        Y = [supply(x) for x in X]
        plt.plot(X,Y)
        plt.plot([0,10000], [0,10000], color="r")
        plt.title(str(cluster))
        plt.show()    
        
        print(f"Cluster {cluster} media: {dh["prob_propietario_0"].mean()}")
        print(f"Cluster {cluster} Prop NAN: {dh["prob_propietario_0"].isna().sum()/len(dh)}")    
                
    df.to_pickle(output_file)


## STEP PRE-6 - INCOME HANDLING


def income_handling(input_file="./data/paper/data_madrid_cluster.pkl", demo_file='./data/Madrid/ConsolidadoDemográfico.xlsx', income_file="./data/paper/ingresos.pkl", thetas_file="./data/paper/thetas.pkl", ingresos=None):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from importlib import reload
    import powerlaw
    from scipy.stats import lognorm
    from tqdm import tqdm


    df = pd.read_pickle(input_file)
    # Eliminamos duplicados y pasamos a propiedades en venga.
    if True:
        df = df.drop_duplicates(subset="id", keep="last")

    import pandas as pd

    if ingresos is None:
        # Ruta al archivo Excel
        file_path = demo_file
        
        # Cargar el archivo Excel
        xls = pd.ExcelFile(file_path)
        
        # Mostrar las hojas del archivo Excel
        print(xls.sheet_names)
        
        # Supongamos que los datos están en la primera hoja
        df_raw = xls.parse(xls.sheet_names[0])
        
        # Inicializar una lista para almacenar los datos procesados
        data = []
        
        # Iterar sobre las columnas del DataFrame para extraer los datos necesarios
        for i in range(0, len(df_raw.columns), 2):
            barrio = df_raw.iloc[0, i]
            ingreso_medio_hogar = df_raw.iloc[53, i+1]
            ingreso_medio_persona = df_raw.iloc[54, i+1]
            
            data.append({
                'Barrio': barrio,
                'Ingreso Medio por Hogar': ingreso_medio_hogar,
                'Ingreso Medio por Persona': ingreso_medio_persona
            })
        
        # Crear un DataFrame con los datos procesados
        df_econ = pd.DataFrame(data)
    else:
        data = []
        for barrio in ingresos.keys():
            data.append({
                "Barrio": barrio,
                "Ingreso Medio por Persona": ingresos[barrio],
            })
    
        # Crear un DataFrame con los datos procesados
        df_econ = pd.DataFrame(data)
        
    
    # Le agregamos los barrios aledaños a Madrid
    import copy
    
    """
    ingresos = copy.copy(ingresos_aledanos)
    
    for i in range(len(df_econ)):
        barrio = df_econ.iloc[i]
        ingresos[barrio["Barrio"]] = barrio["Ingreso Medio por Persona"]    
    """
        
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Supongamos que tu DataFrame 'df' tiene al menos las columnas 'municipio' y 'price'
    # Y que ya tienes definido el diccionario 'ingresos' con el ingreso medio por persona por municipio
    
    # 1. Calcular el precio medio por municipio (o barrio)
    df_grouped = df.groupby('barrio')['price_m2'].mean().reset_index()
    df_grouped.rename(columns={'price_m2': 'mean_price'}, inplace=True)

    print(df_grouped)
    
    # 2. Añadir la información de ingresos usando el diccionario 'ingresos'
    df_grouped['income'] = df_grouped['barrio'].map(ingresos)

    print(df_grouped)
    
    # Opcional: eliminar los municipios que no tengan dato de ingresos
    df_grouped = df_grouped.dropna(subset=['income'])
    
    # 3. Crear el scatterplot
    plt.figure(figsize=(8, 6))
    plt.scatter(df_grouped['income'], df_grouped['mean_price'], color='blue', alpha=0.7)
    plt.xlabel("Ingreso medio por persona (€)")
    plt.ylabel("Precio medio de M2 (€)")
    plt.title("Relación entre Ingreso medio y Precio medio de M2")
    plt.grid(True)
    
    # 4. Ajuste lineal (regresión lineal) utilizando np.polyfit
    slope, intercept = np.polyfit(df_grouped['income'], df_grouped['mean_price'], 1)
    
    # Generar la línea de ajuste para graficar
    x_vals = np.linspace(df_grouped['income'].min(), df_grouped['income'].max(), 100)
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, color='red', label=f"y = {slope:.2f}x + {intercept:.2f}")
    plt.legend()
    
    plt.show()
    
    # 5. Imprimir los coeficientes del ajuste
    print("Pendiente (slope):", slope)
    print("Ordenada al origen (intercept):", intercept)

    thetas = []
    for cluster in sorted(df.Cluster.unique()):
        dg = df[df.Cluster == cluster]
        # 1. Calcular el precio medio por municipio (o barrio)
        df_grouped = dg.groupby('barrio')['price_m2'].mean().reset_index()
        df_grouped.rename(columns={'price_m2': 'mean_price'}, inplace=True)
        
        # 2. Añadir la información de ingresos usando el diccionario 'ingresos'
        df_grouped['income'] = df_grouped['barrio'].map(ingresos)
        
        # Opcional: eliminar los municipios que no tengan dato de ingresos
        df_grouped = df_grouped.dropna(subset=['income'])
        
        # 3. Crear el scatterplot
        #plt.figure(figsize=(8, 6))
        #plt.scatter(df_grouped['income'], df_grouped['mean_price'], color='blue', alpha=0.7)
        #plt.xlabel("Ingreso medio por persona (€)")
        #plt.ylabel("Precio medio de M2 (€)")
        #plt.title("Relación entre Ingreso medio y Precio medio de M2")
        #plt.grid(True)
        
        # 4. Ajuste lineal (regresión lineal) utilizando np.polyfit
        slope, intercept = np.polyfit(df_grouped['income'], df_grouped['mean_price'], 1)
        thetas.append(slope)
    print(thetas)    

    import pickle
    f = open(income_file, "wb")
    pickle.dump(ingresos, f)
    f.close()
    
    f = open(thetas_file, "wb")
    pickle.dump(thetas, f)
    f.close()    



## STEP 6 - DEMAND SIDE


def demand_side_calib(input_file="./data/paper/data_Madrid_probs_propietario.pkl", demo_file='./data/Madrid/ConsolidadoDemográfico.xlsx', input_file_cluster="./data/paper/data_madrid_cluster.pkl", income_file="./data/paper/ingresos.pkl", thetas_file="./data/paper/thetas.pkl", cluster_stats_file="./data/paper/clusters_stats.pkl", output_file="./data/paper/data_madrid_probs_v2.pkl",
                     report_dir="./data/paper/"):

    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from importlib import reload    

        
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from importlib import reload
    import powerlaw
    from scipy.stats import lognorm
    from tqdm import tqdm    

    df = pd.read_pickle(input_file)

    df_probs = df[["id", "prob_propietario_0", "Cluster"]]

    clusters = sorted(list(df.Cluster.unique()))

    import pickle
    f = open(cluster_stats_file, "rb")
    clusters_stats = pickle.load(f)
    f.close()
    supply_functions = [armar_supply_normalizado(cluster, clusters_stats) for cluster in clusters]    
    

    df = pd.read_pickle(input_file_cluster)
    df = df.sort_values(by="fecha")
    #df = df.drop_duplicates(subset="id", keep="first")
    df = df.drop_duplicates(subset="id", keep="last")
    
    fecha_max = df.fecha.max()
    df = df[df.fecha < fecha_max]

    for cluster in clusters:
        print("Cluster", cluster, "Precio M2 10%", df[df.Cluster == cluster]["price_m2"].quantile(q=0.1))
        print("Cluster", cluster, "Precio M2 50%", df[df.Cluster == cluster]["price_m2"].quantile(q=0.5))
        print("Cluster", cluster, "Precio 50%", df[df.Cluster == cluster]["price"].quantile(q=0.5))
        print("Cluster", cluster, "Media", df[df.Cluster == cluster]["price_m2"].mean())    

    df = df.merge(df_probs[["id", "prob_propietario_0"]], on="id", how="left")

    import pickle
    f = open(income_file, "rb")
    ingresos = pickle.load(f)
    f.close()
    
    f = open(thetas_file, "rb")
    thetas = pickle.load(f)
    f.close()    
    
    clusters = sorted(list(df.Cluster.unique()))
    barrios = list(df.barrio.unique())
    
    #barrios = [barrio for barrio in barrios if not barrio is None or barrio != 'Villaverde Alto - Casco Histórico de Villaverde']
    #barrios.append('Villaverde Alto, C.H. Villaverde')
    if None in barrios:
        barrios.remove(None)
    
    min_ingreso = 0.0
    max_ingreso = max([ingresos[barrio] for barrio in ingresos.keys()])
    
    ingresos_normalizados = {}
    for barrio in ingresos.keys():
        ingresos_normalizados[barrio] = normalizar(ingresos[barrio], min_ingreso, max_ingreso)

    print("INGRESOS NORMALIZADOS", ingresos_normalizados)
    
    print(None in barrios)
    
    barrios_l = []
    ingresos_l = []
    for barrio in ingresos.keys():
        barrios_l.append(barrio)
        ingresos_l.append(ingresos[barrio])
    df_ingresos = pd.DataFrame({
        "barrio": barrios_l,
        "ingreso": ingresos_l,
    })
    
    #df_ingresos
    df = df.merge(df_ingresos, on="barrio", how="left")
    
    ingresos_cluster = []
    for cluster in clusters:
        ingreso = df[df.Cluster == cluster].ingreso.mean()
        ingresos_cluster.append(ingreso)
    sizes_cluster = []
    for cluster in clusters:
        print("Cluster", cluster)
        dh = df[df.Cluster == cluster]["size"]
        print("Media", dh.mean())
        print("Mediana", dh.median())
        #size = dh.mean()
        size = dh.median()
        plt.hist(dh)
        plt.show()
        plt.boxplot(dh)
        plt.show()
        sizes_cluster.append(size)    
        
    # 1. Calcular el precio medio por municipio (o barrio)
    df_grouped = df.groupby('barrio')['price_m2'].mean().reset_index()
    df_grouped.rename(columns={'price_m2': 'mean_price'}, inplace=True)
    
    # 2. Añadir la información de ingresos usando el diccionario 'ingresos'
    df_grouped['income'] = df_grouped['barrio'].map(ingresos)
    
    # Opcional: eliminar los municipios que no tengan dato de ingresos
    df_grouped = df_grouped.dropna(subset=['income'])
    
    # 4. Ajuste lineal (regresión lineal) utilizando np.polyfit
    slope, intercept = np.polyfit(df_grouped['income'], df_grouped['mean_price'], 1)
    
    
    # Factores de clusters.
    dicc_cluster_barrio = {}
    for cluster in clusters:
        suma = 0.0
        N = len(df[df.Cluster == cluster])
        for barrio in barrios:
            n = len(df[(df.Cluster == cluster) & (df.barrio == barrio)])
            dicc_cluster_barrio[(cluster, barrio)] = n/N

    
    # El Theta no debe trabajar con valores normalizados...
    thetas_normalizado = []
    for cluster in sorted(df.Cluster.unique()):
        dg = df[df.Cluster == cluster]
    
        min_price = 0.0 #df["price_m2"].quantile(q=0.1)
        max_price = min( max(list(dg["price_m2"].values)), 10000)
        #print(dg["price_m2"])
        #dg["price_norm"] = dg["price_m2"].apply(lambda x: normalizar(x, min_price, max_price))
        dg["price_norm"] = dg["price_m2"]
        
        # 1. Calcular el precio medio por municipio (o barrio)
        df_grouped = dg.groupby('barrio')['price_norm'].mean().reset_index()
        df_grouped.rename(columns={'price_norm': 'mean_price'}, inplace=True)
        
        # 2. Añadir la información de ingresos usando el diccionario 'ingresos'
        df_grouped['income'] = df_grouped['barrio'].map(ingresos)
        
        # Opcional: eliminar los municipios que no tengan dato de ingresos
        df_grouped = df_grouped.dropna(subset=['income'])
        
        # 3. Crear el scatterplot
        #plt.figure(figsize=(8, 6))
        #plt.scatter(df_grouped['income'], df_grouped['mean_price'], color='blue', alpha=0.7)
        #plt.xlabel("Ingreso medio por persona (€)")
        #plt.ylabel("Precio medio de M2 (€)")
        #plt.title("Relación entre Ingreso medio y Precio medio de M2")
        #plt.grid(True)
        
        # 4. Ajuste lineal (regresión lineal) utilizando np.polyfit
        slope, intercept = np.polyfit(df_grouped['income'], df_grouped['mean_price'], 1)
        thetas_normalizado.append(slope)




    ## MODE 7
    
    import scipy
    
    
    gammas = np.linspace(start=0.2, stop=0.45, num=100)
    factor_demanda = np.power(10,np.linspace(start=-20, stop=20, num=200))
    
    #factor_multiplicativo_ingreso = 4.0
    #factor_multiplicativo_ingreso = 20.0
    #factor_multiplicativo_ingreso = 1.0
    factor_multiplicativo_ingreso = None # En ese caso se topa el espacio de búsqueda en 1.0 (ingreso está normalizado)
    #disc = 1000
    disc = 100
    
    
    dgs = []
    import scipy.stats as stats
    
    np.seterr(over='raise')
    gammas_cluster = []
    hs_cluster = []
    factor_ingresos_cluster = []
    x_mins_demand_cluster = []
    factores_demanda_cluster = []
    percentiles_ingreso_cluster = []
    errores_cluster = []
    errores_1_cluster = []
    errores_2_cluster = []
    medianas_inversor_cluster = []
    ingresos_cluster = []
    min_price_cluster = []
    proporciones_ventas_cluster = []
    
    min_ingreso = 0.0 #min([ingresos[barrio] for barrio in ingresos.keys()])
    max_ingreso = max([ingresos[barrio] for barrio in ingresos.keys()])
    
    #thetas_normalizados = thetas #thetas_normalizados = [normalizar(theta, min_theta, max_theta) for theta in thetas]
    thetas_normalizados = thetas_normalizado
    #sizes_normalizados = [normalizar(size, min_size, max_size) for size in sizes_cluster]
    sizes_normalizados = sizes_cluster
    ingresos_normalizados = {}
    
    for barrio in ingresos.keys():
        ingresos_normalizados[barrio] = normalizar(ingresos[barrio], min_ingreso, max_ingreso)
    
    #ingresos_normalizados = ingresos
    print("INGRESOS NORMALIZADOS", ingresos_normalizados)

    barrios_set = set(barrios)
    barrios_ing_set = set(ingresos.keys())
    print("Elementos de barrios fuera de ingresos", barrios_set - barrios_ing_set)
    print("Elementos de ingresos fuera de barrios", barrios_ing_set - barrios_set)

    
    limite_k = 10
    
    for cluster in clusters:
        print("Cluster", cluster)
    
        ingreso_cluster = 0.0
        for barrio in barrios:
            ingreso_cluster += dicc_cluster_barrio[(cluster, barrio)]*((ingresos_normalizados[barrio]))
        print(f"Ingreso Cluster {ingreso_cluster}")
    
        momento_cuadrado = 0.0
        for barrio in barrios:
            momento_cuadrado += dicc_cluster_barrio[(cluster, barrio)]*((ingresos_normalizados[barrio])**2)
        print(f"Momento cuadrado {momento_cuadrado}")
    
        varianza_cluster = momento_cuadrado - ingreso_cluster**2
    
    
        #sigma = np.sqrt(np.log(0.5 * (1+np.sqrt(1 + 4*varianza_cluster/ingreso_cluster**2))))
        
        
        #sigma = np.sqrt(np.log(1 + varianza_cluster/ingreso_cluster**2))
        #mu = np.log(ingreso_cluster) - (sigma**2)/2
    
        mu = np.log(ingreso_cluster)
        sigma = 1.0
        
        
        
        #sigma = 1.0
        #sigma = np.sqrt(varianza_cluster)
        print(f"mu ingresos cluster {mu}")
        print(f"Sigma ingresos cluster {sigma}")
        
        
        kappa = clusters_stats[cluster]["kappa"]
        dg = df[(df.Cluster == cluster)]
        print("Percentil 10 del cluster", dg.price_m2.quantile(q=0.1))
        print("Activación del power law", clusters_stats[cluster]["x_min"])
        #x_min = max(dg.price_m2.quantile(q=0.1), clusters_stats[cluster]["x_min"]) 
        x_min = dg.price_m2.quantile(q=0.1) # CAMBIO PARA PERMITIR EVALUAR DEBAJO DEL POWER LAW DE SUPPLY.
        dg = dg[dg.price_m2 >= x_min]
        print("Min price:", x_min)
        x_mins_demand_cluster.append(x_min)
        x_max = dg.price_m2.quantile(q=0.95) #min(10000,dg.price_m2.max())
        #X = np.linspace(start=x_min + 0.01, stop=x_max, num=100)
        #X = np.linspace(start=x_min/2.0, stop=x_max, num=100)
        X = np.linspace(start=0.01, stop=x_max, num=100)
    
        best_h = None
        best_gamma = None
        best_ingreso = None
        best_cruce = None
        best_error = None
        best_error_1 = None
        best_error_2 = None
        best_supply = None
        best_demand_factor = None
        best_ingreso_factor = None
        best_percentil_ingreso = None
        bext_x = None
        best_surplus = None
        best_demand = None
        best_supply = None
        best_indice = None
        best_inversor = None
        best_ingreso = None
        min_price = None
        best_proporciones_ventas = None
        theta = thetas_normalizado[cluster]
        soluciones = []
        momento_propietario = dg["prob_propietario_0"].median()
        print("Mediana Propietario", momento_propietario)
        for gamma in tqdm(gammas):
            ahorcado = (gamma**((1.0/gamma)*(1.0 - 1.0/gamma)))*theta**(1.0/gamma + 1.0) 
            
            #ahorcado = (gamma**((1.0/gamma)*(1.0 - 1.0/gamma)))*slope**(1.0/gamma + 1.0)
            epsilon_1 = (1.0/gamma)*(gamma - 2.0 + 1.0/gamma)
            epsilon_2 = (1.0/gamma)*(1.0/gamma - 1.0)
            epsilon_3 = 1.0/gamma
            #print(f"Theta {theta} Epsilon 1: {epsilon_1} Epsilon 2: {epsilon_2} Epsilon 3: {epsilon_3} Ahorcado {ahorcado}")
            ahorcado = 1.0
            
            
            for factor in factor_demanda:
    
                factor_ingreso = factor**(1.0/epsilon_1)
                        
                supply_f = supply_functions[cluster]        
                try:
                    demand = armar_funcion_demanda_normalizado(clusters, barrios, dicc_cluster_barrio, ingresos,  thetas_normalizado, sizes_cluster, x_min, x_max,gamma, debug=False)
                    #demand = armar_funcion_demanda(gamma,size_factor_h=h, size_factor_ingreso=factor_ingreso_h, debug=True)
                except Exception as e:
                    print(e)
                    raise e
                    print(f"Gamma: {gamma}, H {h}, FI {factor_ingreso}")
                    continue
                # Punto de cruce.
                Y_surplus = []
                Y_demand = []
                Y_supply = []
                new_X = []
                completado = True
                for x in X:
                    y_supply = supply_f(x)
                    if np.isnan(y_supply):
                        #continue
                        #y_supply = x
                        y_supply = clusters_stats[cluster]["x_min"]
                    try:
                        demanda = demand(x,cluster)
                        y_demand = factor*demanda
                        #print("Demanda", y_demand)
                        y_surplus = y_demand - y_supply
                        #print("Surplus", y_surplus)
                        new_X.append(x)
                        Y_demand.append(y_demand)
                        Y_surplus.append(y_surplus)
                        Y_supply.append(y_supply)        
                    except Exception as e:
                        print("***********OVERFLOW*******")
                        print(f"Gamma: {gamma}, H {h}, Factor Demanda {factor} X {x} Demanda {demanda}")
                        #raise(e)
                        #continue
                        completado = False
                        break
                if not completado:
                    continue
                
                if len(Y_surplus) == 0:
                    #print("NO hay excedente para evaluar")
                    continue
                                 
                # Cruce.
                fui_negativo = False
                conteo_negativo = 0
                fui_positivo = False
                conteo_positivo = 0
                found = False
                
                for indice in range(len(new_X)):
                    y = Y_surplus[indice]
                    if y < 0.0:
                        fui_negativo = True
                        conteo_negativo += 1
                        continue
                    if y >= 0.0:
                        fui_positivo = True
                        conteo_positivo += 1
                        if fui_positivo and fui_negativo:
                            found = True
                            break
                if not found:  #or conteo_negativo < 0.1*len(X): # Recorrió poca curva...
                    #print("No hay cruce")
                    #print(f"Gamma: {gamma}, H {h}, Factor Demanda {factor} X {x} Demanda {demanda}")
                    continue
                indice = np.abs(np.array(Y_surplus) - 0.0).argmin()
                cruce = new_X[indice]
                error = np.abs(cruce - x_min) # Esto lo fuerza a estar cerca del origen de la curva...
                #print(f"Corte en indice {indice}, X {cruce}, X_min {x_min}, error relativo {error/x_min}")
                if error/x_min < 0.02:
                    soluciones.append({
                        "Gamma": gamma, 
                        "Factor": factor,
                        "Error": error,
                        "Error_rel": error/x_min,
                        "Cruce": cruce,
                    }
                )
                error_1 = error/x_min
                if error_1 > 0.05:
                    continue
    
                
                ## SECTOR PROBABILIDADES
                centro_ln_ingresos = 0.0
                for barrio in barrios:
                    # TODO: revisar si debiera ser ingreso o ingreso_normalizados.
                    centro_ln_ingresos += dicc_cluster_barrio[(cluster, barrio)]*((ingresos_normalizados[barrio]*factor_ingreso)**epsilon_1)
                ingreso_inmueble = centro_ln_ingresos
    
                
                
                h = sizes_cluster[cluster]**epsilon_2
                probs = []
                precios = dg["price_m2"].values
                #barrios = dg["barrio"].values
                ventas = 0
                valores_corte = []
                for i in range(len(dg)):
                #for i in tqdm(range(len(dg))):
                    price_m2 = precios[i]
                    price_norm = normalizar(price_m2, 0.0, x_max)
                    supply = supply_functions[cluster](price_m2) # Supply normaliza por su cuenta...                
                    # TODO: validar!!
                    #if np.isnan(supply):
                        #supply = price_m2
                        #supply = clusters_stats[cluster]["x_min"]
                        #continue
                    
                    def excedente(ingreso):
                        #demanda_loc = factor*ingreso*h*(price_norm**epsilon_3)*ahorcado
                        #demanda_loc = ingreso*h*(price_norm**epsilon_3)*ahorcado
                        demanda_loc = ((ingreso*factor_ingreso)**epsilon_1)*h*(price_norm**epsilon_3)*ahorcado
                        return demanda_loc - supply
                    if excedente(ingreso_cluster) < 0.0 or np.isnan(supply):
                        probs.append(None)
                        continue                
                    ventas += 1
                    if min_price is None or price_m2 < min_price:
                        min_price = price_m2
                    if factor_multiplicativo_ingreso is None:
                        grilla = np.linspace(start=0.01, stop=0.99, num=disc)
                        #mu = np.log(ingreso_cluster)
                        percentiles = scipy.stats.norm.ppf(grilla, loc=mu, scale=sigma)
                        ingresos_grilla = np.exp(percentiles)
                        values = [excedente(ingreso) for ingreso in ingresos_grilla]
                        indice = np.abs(np.array(values) - 0.0).argmin()
                        corte_ingreso = grilla[indice]
                        threshold = np.log(corte_ingreso)
                        probabilidad = 1 - stats.norm.cdf(threshold, loc=mu, scale=sigma) 
                        probs.append(probabilidad)
                        valor = np.log(corte_ingreso)
                        valores_corte.append(valor)
                    else:
                        tope = factor_multiplicativo_ingreso*ingreso_cluster
                        grilla = np.linspace(start=1e-6, stop=tope, num=disc)
                        values = [excedente(ingreso) for ingreso in grilla]
                        indice = np.abs(np.array(values) - 0.0).argmin()
                        corte_ingreso = grilla[indice]
                        #mu = np.log(ingreso_inmueble)
                        #mu = np.log(ingreso_cluster)
                        #sigma = 1.0
                        threshold = np.log(corte_ingreso)
                        probabilidad = 1 - stats.norm.cdf(threshold, loc=mu, scale=sigma) 
                        probs.append(probabilidad)
        
                        #valor = np.log(ingreso_cluster*factor_ingreso)
                        valor = np.log(corte_ingreso)
                        valores_corte.append(valor)
                        #percentil_ingreso = stats.norm.cdf(valor, loc=mu, scale=sigma) 
                    
                    """
                    print("Ingreso Inmueble", ingreso_inmueble)
                    print("Mu", mu)
                    print("Corte Ingreso", corte_ingreso)
                    print("Probabilidad", probabilidad)
                    print("Demanda", demanda)
                    print("Supply", supply)
                    print("Factor Demanda", factor)
                    print("Gamma", gamma)
                    print("Valor de Demanda en corte", corte_ingreso*h*(price_norm**epsilon_3)*ahorcado)
                    1/0
                    """
                probs = [prob for prob in probs if not prob is None]
                momento_inversor = np.median(probs)
                
                mu = np.log(ingreso_cluster)
                #sigma = 1.0
                salario_mediano = np.median([valor for valor in valores_corte if not valor is None])
                corte_ingreso_mediano = salario_mediano
                percentil_ingreso = stats.norm.cdf(salario_mediano, loc=mu, scale=sigma) 
            
                error_2 = np.abs(momento_inversor - momento_propietario)
    
                error = error_1 + error_2
                
                if best_error is None or error < best_error:
                    best_gamma = gamma
                    #best_h = h
                    #best_ingreso = factor_ingreso
                    best_error = error
                    best_cruce = cruce
                    best_x = new_X
                    best_surplus = Y_surplus
                    best_demand_factor = factor
                    best_ingreso_factor = factor_ingreso
                    best_percentil_ingreso = percentil_ingreso
                    best_supply = Y_supply
                    best_demand = Y_demand
                    best_min = cruce
                    best_error_1 = error_1
                    best_error_2 = error_2
                    best_inversor = momento_inversor
                    best_ingreso = corte_ingreso_mediano
                    best_proporciones_ventas = ventas/len(dg)
        plt.plot(best_x, best_surplus, color="blue", label="Surplus")
        plt.plot(best_x, best_supply, color="green", label="Supply")
        plt.plot(best_x, best_demand, color="red", label="Demand")
        plt.title(f"Cluster {cluster} Gamma {best_gamma}")
        plt.legend()
        plt.show()
        print("Cruce",best_min)
        print("Error", best_error)
        print("Error 1", best_error_1)
        print("Error 2", best_error_2) # Está muy alto esta diferencia...
        print("Momento Inversor:", best_inversor)
        print("Precio Mínimo de adquisición:", min_price)
        print("Proporciones de ventas:", best_proporciones_ventas)
        print("Percentil Ingresos", best_percentil_ingreso)
        #print("Ingresos", best_ingreso)
        gammas_cluster.append(best_gamma)
        hs_cluster.append(best_h)
        #factor_ingresos_cluster.append(best_ingreso)
        factores_demanda_cluster.append(best_demand_factor)
        factor_ingresos_cluster.append(best_ingreso_factor)
        percentiles_ingreso_cluster.append(best_percentil_ingreso)
        errores_cluster.append(best_error)
        errores_1_cluster.append(best_error_1)
        errores_2_cluster.append(best_error_2)
        ingresos_cluster.append(corte_ingreso)
        min_price_cluster.append(min_price)
        proporciones_ventas_cluster.append(best_proporciones_ventas)
        #print(soluciones)
        ventas = 0
        gamma = best_gamma
        epsilon_1 = (1.0/gamma)*(gamma - 2.0 + 1.0/gamma)
        epsilon_2 = (1.0/gamma)*(1.0/gamma - 1.0)
        epsilon_3 = 1.0/gamma
        #print(f"Theta {theta} Epsilon 1: {epsilon_1} Epsilon 2: {epsilon_2} Epsilon 3: {epsilon_3} Ahorcado {ahorcado}")
        ahorcado = 1.0
        h = sizes_cluster[cluster]**epsilon_2
        probs_k_cluster = []
        for i in tqdm(range(len(dg))):
            propiedad = dg.iloc[i]
            probs_k = []
            prices_k = []
            price_m2 = propiedad["price_m2"]
            #barrio = propiedad["barrio"]
            #if barrio == 'Villaverde Alto - Casco Histórico de Villaverde':
            #    barrio = "Villaverde Alto, C.H. Villaverde"
            #if barrio is None:
            #    continue
            price_norm = normalizar(price_m2, 0.0, x_max)
            supply_f = supply_functions[cluster]
            #supply = price_m2
            supply = supply_f(price_m2)
            #if np.isnan(supply): # AGREGADO.
                #supply = clusters_stats[cluster]["x_min"]#price_m2#x_min #VER!!
            
            #accum_prob_failure = 1.0
    
            #print("Price M2", price_m2)
            #print("Price Norm", price_norm)
            #1/0
            probs_k = []
            probs_k_cluster.append(probs_k)
            for t in range(limite_k):
                #prices_k.append(supply)
                def excedente(ingreso):
                    demanda_loc = ((ingreso*best_ingreso_factor)**epsilon_1)*h*(price_norm**epsilon_3)*ahorcado
                    return demanda_loc - supply
    
                exced = excedente(ingreso_cluster)
                #print("Excedente", exced)
                #1/0
                
                if exced < 0.0 or np.isnan(supply):
                    probs_k.append(None)
                    #prices_k.append(None)
                    continue
                elif t == 0 and exced >= 0.0:
                    #print("Vendido!")
                    ventas += 1
                
                if factor_multiplicativo_ingreso is None:
                    grilla = np.linspace(start=0.01, stop=0.99, num=disc)
                    #mu = np.log(ingreso_cluster)
                    percentiles = scipy.stats.norm.ppf(grilla, loc=mu, scale=sigma)
                    ingresos_grilla = np.exp(percentiles)
                    values = [excedente(ingreso) for ingreso in ingresos_grilla]
                    indice = np.abs(np.array(values) - 0.0).argmin()
                    corte_ingreso = grilla[indice]
                    threshold = np.log(corte_ingreso)
                    probabilidad = 1 - stats.norm.cdf(threshold, loc=mu, scale=sigma) 
                    probs.append(probabilidad)
                    valor = np.log(corte_ingreso)
                    valores_corte.append(valor)
                else:
                    tope = factor_multiplicativo_ingreso*ingreso_cluster
                    grilla = np.linspace(start=1e-6, stop=tope, num=disc)
                    values = [excedente(ingreso) for ingreso in grilla]
                    indice = np.abs(np.array(values) - 0.0).argmin()
                    corte_ingreso = grilla[indice]
                    #mu = np.log(ingreso_inmueble)
                    #mu = np.log(ingreso_cluster)
                    #sigma = 1.0
                    threshold = np.log(corte_ingreso)
                    probabilidad = 1 - stats.norm.cdf(threshold, loc=mu, scale=sigma) 
                    probs.append(probabilidad)
    
                    #valor = np.log(ingreso_cluster*factor_ingreso)
                    valor = np.log(corte_ingreso)
                    valores_corte.append(valor)
                    #percentil_ingreso = stats.norm.cdf(valor, loc=mu, scale=sigma) 
    
                probs_k.append(probabilidad) # Probabilidad de éxito, de vender en la instancia k.
                
                #price_m2 = supply # VER!!
                #price_norm = normalizar(price_m2, 0.0, x_max)
                supply = supply_f(supply)
                #if np.isnan(supply):
                #    supply = clusters_stats[cluster]["x_min"]
    
            #probs_k_cluster.append(probs_k)
            #prices_k_cluster.append(prices_k)
        print("Proporción de Ventas", float(ventas)/len(dg)) 
        for t in range(limite_k):
            dg[f"prob_investor_{t}"] = [probs_k_cluster[propiedad][t] for propiedad in range(len(probs_k_cluster))]    
            #dg[f"prices_{t}"] = [prices_k_cluster[propiedad][t] for propiedad in range(len(prices_k_cluster))]    
    
        
        # Asumiendo que df ya está cargado como DataFrame
        med_owner = dg['prob_propietario_0'].median()
        med_inv = dg['prob_investor_0'].median()
        
        plt.figure()
        plt.scatter(dg['prob_propietario_0'], dg['prob_investor_0'])
        plt.axvline(med_owner, linestyle='--')
        plt.axhline(med_inv, linestyle='--')
        plt.xlabel('Probability Owner')
        plt.ylabel('Probability Investor')
        plt.title(f'Scatter plot {cluster} of Probability Owner vs Probability Investor, with lines for the median values')
        plt.show()
        print("Mínimo Investor prob", dg['prob_investor_0'].min())
    
    
        #print("Cuadrante NO", len(df[(df.prob_investor_0 > med_inv) & ((df.prob_propietario_0 <= med_owner))]) / len(df) )
        #print("Cuadrante NE", len(df[(df.prob_investor_0 > med_inv) & ((df.prob_propietario_0 > med_owner))]) / len(df) )
        #print("Cuadrante SO", len(df[(df.prob_investor_0 <= med_inv) & ((df.prob_propietario_0 <= med_owner))]) / len(df) )
        #print("Cuadrante SE", len(df[(df.prob_investor_0 <= med_inv) & ((df.prob_propietario_0 > med_owner))]) / len(df) )    
        
        dgs.append(dg)            
    dh = pd.concat(dgs)
        
    dh.to_pickle(output_file)            

    for t in range(10):
        print(f"NaN en columna {t}: {dh[f"prob_investor_{t}"].isna().mean()}")


    plt.boxplot(dh.prob_investor_0.dropna())

    
    ingresos_cluster_barriales = []
    for cluster in clusters:
        print("Cluster", cluster)
    
        ingreso_cluster = 0.0
        for barrio in barrios:
            ingreso_cluster += dicc_cluster_barrio[(cluster, barrio)]*((ingresos[barrio]))
        print(f"Ingreso Cluster {ingreso_cluster}")
        ingresos_cluster_barriales.append(ingreso_cluster)

    
    text = """
    \\begin{table}[h!]
    \\begin{tabular}{|c|c|c|c|c|c|}
    \\hline
    \\textbf{Parameter / Cluster} & \\textbf{0} & \\textbf{1} & \\textbf{2} & \\textbf{3} & \\textbf{4} \\\\ \\hline
    
    """

    obj = {}
    obj["ingresos_cluster_barriales"] = ingresos_cluster_barriales
    obj["percentiles_ingreso_cluster"] = percentiles_ingreso_cluster
    obj["sizes_cluster"] = sizes_cluster
    obj["min_price_cluster"] = min_price_cluster
    obj["thetas_cluster"] = thetas
    obj["errores_1_cluster"] = errores_1_cluster
    obj["errores_2_cluster"] = errores_2_cluster
    obj["percentiles_ingreso_cluster"] = percentiles_ingreso_cluster
    obj["gammas_cluster"] = gammas_cluster
    obj["thetas"] = thetas

    import pickle
    f = open(report_dir + "/obj_demanda.pkl", "wb")
    pickle.dump(obj, f)
    f.close()
    

    
    text += "\\textbf{$\\omega(j,b)|_l$} & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(ingresos_cluster_barriales[cluster],2)) + " & "
        else:
            text += str(round(ingresos_cluster_barriales[cluster],2)) + " \\\\ \\hline \n"
    
    
    text += "Percentil Ingreso & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(percentiles_ingreso_cluster[cluster],2)) + " & "
        else:
            text += str(round(percentiles_ingreso_cluster[cluster],2)) + " \\\\ \\hline \n"
    
    
    text += "\\textbf{$h(l)$} & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(sizes_cluster[cluster],2)) + " & "
        else:
            text += str(round(sizes_cluster[cluster],2)) + " \\\\ \\hline \n"
    
    text += "$\\underline{p}_l^{se}$ & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(min_price_cluster[cluster],2)) + " & "
        else:
            text += str(round(min_price_cluster[cluster],2)) + " \\\\ \\hline \n"
    
    
    text += "\\textbf{$\\theta^l$} & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(thetas[cluster],2)) + " & "
        else:
            text += str(round(thetas[cluster],2)) + " \\\\ \\hline \n"
    
    text += "\\textbf{$\\gamma^l$} & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(gammas_cluster[cluster],2)) + " & "
        else:
            text += str(round(gammas_cluster[cluster],2)) + " \\\\ \\hline \n"
    
    
    text += """
    \\end{tabular}
    \\caption{Calibrated parameters. Demand side and Stationary Equilibrium}
    \\label{table:cal_3}
    \\end{table}
    
    """
    
    print(text)    


    
    text = """
    \\begin{table}[h!]
    \\begin{tabular}{|c|c|c|c|c|c|}
    \\hline
    \\textbf{Parameter / Cluster} & \\textbf{0} & \\textbf{1} & \\textbf{2} & \\textbf{3} & \\textbf{4} \\\\ \\hline
    
    """
    
    
    text += "ERROR 1 & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(errores_1_cluster[cluster],2)) + " & "
        else:
            text += str(round(errores_1_cluster[cluster],2)) + " \\\\ \\hline \n"
    
    
    
    text += "$|m(f^l)-m(\bar{F}^l)|$ & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(errores_2_cluster[cluster],2)) + " & "
        else:
            text += str(round(errores_2_cluster[cluster],2)) + " \\\\ \\hline \n"
    
    
    text += "Percentil Ingreso & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(percentiles_ingreso_cluster[cluster],2)) + " & "
        else:
            text += str(round(percentiles_ingreso_cluster[cluster],2)) + " \\\\ \\hline \n"
    
    
    text += "\\textbf{$h(l)$} & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(sizes_cluster[cluster],2)) + " & "
        else:
            text += str(round(sizes_cluster[cluster],2)) + " \\\\ \\hline \n"
    
    text += "$\\underline{p}_l^{se}$ & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(min_price_cluster[cluster],2)) + " & "
        else:
            text += str(round(min_price_cluster[cluster],2)) + " \\\\ \\hline \n"
    
    
    text += "\\textbf{$\\theta^l$} & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(thetas[cluster],2)) + " & "
        else:
            text += str(round(thetas[cluster],2)) + " \\\\ \\hline \n"
    
    text += "\\textbf{$\\gamma^l$} & "
    for cluster in clusters:
        if cluster < 4:
            text += str(round(gammas_cluster[cluster],2)) + " & "
        else:
            text += str(round(gammas_cluster[cluster],2)) + " \\\\ \\hline \n"
    
    
    text += """
    \\end{tabular}
    \\caption{Calibrated parameters. Demand side and Stationary Equilibrium}
    \\label{table:cal_3}
    \\end{table}
    
    """
    
    print(text)


## STEP 7 - INVESTOR PROBLEM

def investor_problem(input_file="./data/paper/data_madrid_probs_v2.pkl", prob_file="./data/paper/data_Madrid_probs_propietario.pkl", cluster_stats_file="./data/paper/clusters_stats.pkl", output_file="./data/paper/data_ROI.pkl"):



    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from importlib import reload
    import powerlaw
    from scipy.stats import lognorm
    from tqdm import tqdm



    
    dh = pd.read_pickle(input_file)
    clusters = sorted(list(dh.Cluster.unique()))
    
    import pickle
    f = open(cluster_stats_file, "rb")
    clusters_stats = pickle.load(f)
    f.close()
    supply_functions = [armar_supply_normalizado(cluster, clusters_stats) for cluster in clusters]
    #clusters_stats


    
    df = pd.read_pickle(prob_file)
    limite_k = 10
    columnas = [f"prob_propietario_{k}" for k in range(1,limite_k)] + ["id"]
    dh = dh.merge(df[columnas], how="left", on="id")

    dh.prob_investor_0.isna().mean()

    dh = dh[dh.prob_investor_0.isna() == False]
    len(dh)

    #r = 0.05
    #r = 0.01
    zeta = -4
    #zeta = 1.0
    beta = 0.99
    pS = 0.03
    #r = (1/beta)-1
    r = 0.0120
    
    print(r)
    
    
    limite_k = 10
    #limite_k = 1
    
    
    VI, VO, comprable = [], [], np.zeros(len(dh))
    
    
    for i in tqdm(range(len(dh))):
        propiedad = dh.iloc[i]
        cluster = propiedad["Cluster"]
        price_m2 = propiedad["price_m2"]  # ✅ PRECIO M2
        #size = propiedad["size"]
        size = 1.0
        supply = supply_functions[cluster]
        
        # ---- VI (inversor) ----
        suma_vi, prob_failure_vi, price_m2_t = 0.0, 1.0, price_m2
        price_t = price_m2_t*size
        if np.isnan(propiedad["prob_investor_0"]):
            VI.append(None); VO.append(None)
            continue
        for k in range(limite_k):
            f_k = propiedad[f"prob_investor_{k}"]
            if f_k is None:
                f_k = 0.0
            #ingreso_vi = ((1+r)/r) * f_k * prob_failure_vi * price_t * (1+r)**(-k)
            ingreso_vi =  f_k * prob_failure_vi * price_t * ((1+r)/r) * (1+r)**(-k)
            costo_vi = r * prob_failure_vi * price_t * (1+r)**(-k)
            suma_vi += ingreso_vi - costo_vi
            prob_failure_vi *= (1 - f_k)
            price_m2_t = supply(price_m2_t) if not np.isnan(supply(price_m2_t)) else price_m2_t
            price_t = price_m2_t*size
        VI.append(suma_vi)
    
        #print("VI", suma_vi)    
    
        
        # ---- VO (propietario) ----
        suma_vo, prob_failure_vo, price_m2_t = 0.0, 1.0, price_m2
        price_t = price_m2_t*size
        for k in range(limite_k):
            F_k = propiedad[f"prob_propietario_{k}"]
            if F_k is None:
                F_k = 0.0
            ingreso_vo = F_k * prob_failure_vo * price_t * (beta/(1-beta))* beta**k
    
            ceiling = clusters_stats[cluster]["x_min"]
            #floor = supply(price_m2_t)
            floor = price_m2_t
            holding_cost = (1/zeta)*(size*ceiling - size*floor)**zeta
            costo_vo = r*prob_failure_vo * holding_cost * beta**k
            """
            print("k", k)
            print("Ingreso vo", ingreso_vo)
            print("floor", floor)
            print("ceiling", ceiling)
            print("holding_cost", holding_cost)
            print("costo_vo", costo_vo)
            """
            #if np.isnan(costo_vo) or np.isinf(costo_vo):
            #    costo_vo = 0.0
    
            #print("vo", ingreso_vo - costo_vo)
    
    
            if np.isnan(costo_vo) or np.isinf(costo_vo):
                suma_vo += 0.0
            else:
                suma_vo += ingreso_vo - costo_vo   
                
    
            
            #suma_vo += ingreso_vo - costo_vo
            prob_failure_vo *= (1 - F_k)
            price_m2_t = supply(price_m2_t) if not np.isnan(supply(price_m2_t)) else price_m2_t
            price_t = price_m2_t*size
        VO.append(suma_vo)
    
        comprable[i] = 1.0 if VI[-1] > VO[-1] else 0.0
    
        #1/0
    
    dh["VI"] = VI
    dh["VO"] = VO
    dh["comprable"] = comprable
    dh.comprable.mean()
    


    def compute_ROI(x):
        if np.isnan(x["VI"]) or np.isnan(x["VO"]):
            return np.nan
        else:
            return (x["VI"] - x["VO"])/x["VO"]
    
    dh["ROI"] = dh[["VI", "VO"]].apply(compute_ROI, axis=1)
    dh[["VI", "VO", "ROI"]]

    dg = dh.sort_values(by="ROI", ascending=False)

    intrinsic = ['id',
     'fecha',
     'operation',
     'datasource_name',
     'property_type',
     'subtype',
     'municipality',
     'municipality_code5',
     'municipality_code5num',
     'comaut',
     'comaut_code',
     'province',
     'province_code',
     'district',
     'neighborhood',
     'title',
     'postal_code',
     'postal_codenum',
     'latitude_x',
     'longitude_x',
     'price',
     'lprice',
     'price_m2',
     'lprice_m2',
     'size',
     'lsize',
     'floor',
     'bedrooms',
     'bathrooms',
     'lift',
     'garage',
     'storage',
     'terrace',
     'air_conditioning',
     'swimming_pool',
     'garden',
     'sports',
     'status',
     'new_construction',
     'rating_leads',
     'rating_visits',]

    dg_1 = dg[dg.price_m2 <= 10000]


    dh.to_pickle(output_file)

## STEP 8 - PLOTS

def neighborhood_plots(input_file="./data/paper/data_ROI.pkl"):
    
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    df = pd.read_pickle(input_file)
    df.ROI.min(), df.ROI.max()
    df = df[df.comprable == 1.0]
    df.ROI.min(), df.ROI.max()
    
    import geopandas as gpd
    from shapely.geometry import mapping
    import json
    from pyproj import Transformer
    
    # Cargar el archivo SHP usando geopandas
    shp_path = './data/Madrid/Barrios.shp'  # Reemplaza con la ruta a tu archivo SHP
    gdf = gpd.read_file(shp_path)
    
    # Crear un transformador para convertir de UTM a WGS84
    # Supongamos que las coordenadas están en el sistema UTM zona 30N (EPSG:25830)
    transformer = Transformer.from_crs("epsg:25830", "epsg:4326", always_xy=True)
    
    # Crear un diccionario con el nombre del barrio y las coordenadas
    barrios_dict = {}
    
    for _, row in gdf.iterrows():
        nombre_barrio = row['NOMBRE']  # Ajusta según el campo del nombre del barrio en tu archivo
        geometria = row['geometry']
        
        # Verificar si la geometría es un polígono o multipolígono
        if geometria.geom_type == 'Polygon':
            coords = list(geometria.exterior.coords)
            # Convertir las coordenadas a latitud y longitud
            coords = [transformer.transform(x, y) for x, y in coords]
            barrios_dict[nombre_barrio] = coords
        elif geometria.geom_type == 'MultiPolygon':
            coords = []
            for polygon in geometria:
                polygon_coords = list(polygon.exterior.coords)
                # Convertir las coordenadas a latitud y longitud
                polygon_coords = [transformer.transform(x, y) for x, y in polygon_coords]
                coords.extend(polygon_coords)
            barrios_dict[nombre_barrio] = coords
    
    
    
    import numpy as np
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import plotly.graph_objects as go
    
    def rgba_to_plotly_string(rgba):
        """Convierte una tupla (r,g,b,a) con valores 0–1 en un string 'rgba(R,G,B,A)'."""
        r, g, b, a = rgba
        return f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a})"
    
    def plot_neighborhood_with_ranking(
            df, value_label, path, barrios_dict,
            municipality="Madrid", min_conteo=0,
            fill_opacity=0.5, font_size=12):
        """
        Dibuja polígonos con go.Scattermapbox usando fillcolor RGBA (para transparencia)
        y luego superpone el ranking como texto.
    
        Parámetros
        ----------
        df : pd.DataFrame
            Columnas mínimas: 'barrio', 'municipality' y value_label.
        value_label : str
            Nombre de la columna cuyos promedios se colorean.
        path : str
            Ruta de salida del HTML.
        barrios_dict : dict[str, list[tuple[lon,lat]]]
            Diccionario {barrio: [(lon,lat), …]} con los vértices.
        municipality : str
        min_conteo : int
        fill_opacity : float
            0.0 = totalmente transparente, 1.0 = totalmente opaco.
        font_size : int
            Tamaño del texto de ranking.
        """
        # 1) Filtrar, agrupar y calcular ranking
        #dg = (df.loc[(df[value_label].notna()) & (df["municipality"] == municipality)]
        dg = (df.loc[(df["municipality"] == municipality)]
                .groupby("barrio")
                .agg(
                    scoring=(value_label, "mean"),
                    conteo=(value_label, "size") #count
                )
                .reset_index())
        print(f"Cantidad de barrios para {value_label} ANTES DE FILTRO: {len(dg)}")
        print(dg.sort_values("conteo"))
        dg = dg[dg["conteo"] >= min_conteo].sort_values("scoring", ascending=False)
        dg["ranking"] = np.arange(1, len(dg) + 1)
    
        print(f"Cantidad de barrios para {value_label} LUEGO DE FILTRO: {len(dg)}")
        
        # 2) Preparar mapeo de colores con transparencia
        min_val, max_val = dg["scoring"].min(), dg["scoring"].max()
        norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
        cmap = cm.get_cmap("viridis")
        scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
        
        # 3) Crear figura y añadir polígonos
        fig = go.Figure()
        for _, row in dg.iterrows():
            barrio = row["barrio"]
            if barrio not in barrios_dict:
                continue
            value, conteo, rank = row["scoring"], row["conteo"], row["ranking"]
            coords = barrios_dict[barrio]
            lons, lats = zip(*coords)
            
            # color RGBA con el nivel de transparencia deseado
            r, g, b, _ = scalar_map.to_rgba(value)
            color = rgba_to_plotly_string((r, g, b, fill_opacity))
            
            fig.add_trace(go.Scattermapbox(
                fill="toself",
                lon=lons, lat=lats,
                marker={"size": 0},
                fillcolor=color,
                line={"color": "black", "width": 1},
                hoverinfo="text",
                hovertext=(
                    f"<b>{barrio}</b><br>"
                    f"{value_label}: {value:.2f}<br>"
                    f"N: {conteo}"
                ),
                showlegend=False
            ))
            
            # centroid para el ranking
            centroid_lon = np.mean(lons)
            centroid_lat = np.mean(lats)
            fig.add_trace(go.Scattermapbox(
                lon=[centroid_lon], lat=[centroid_lat],
                mode="text",
                text=[str(rank)],
                textfont={"size": font_size, "color": "black"},
                hoverinfo="none",
                showlegend=False
            ))
        
        # 4) (Opcional) Escala de colores como traza invisible para colorbar
        tick_vals = np.linspace(min_val, max_val, 5)
        tick_text = [f"{v:.2f}" for v in tick_vals]
        fig.add_trace(go.Scattermapbox(
            lon=[None], lat=[None], mode="markers",
            marker=dict(
                showscale=True,
                colorscale="Viridis",
                cmin=min_val, cmax=max_val,
                color=[min_val],
                colorbar=dict(
                    title=value_label,
                    tickvals=tick_vals,
                    ticktext=tick_text,
                    ticks="outside"
                )
            ),
            hoverinfo="none",
            showlegend=False
        ))
        
        # 5) Layout final
        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox=dict(center=dict(lat=40.4168, lon=-3.7038), zoom=11),
            margin=dict(t=40, r=0, l=0, b=0),
            title=value_label
        )
        
        # 6) Guardar y devolver
        fig.write_html(path, include_plotlyjs="cdn")
        return fig
    
    
    import numpy as np
    

    
    
    df_inv = pd.read_pickle("./data/paper/data_madrid_probs.pkl")
    print("Propiedades en DF_INV", len(df_inv))
    df_inv = df_inv[(df_inv.municipality == "Madrid")& (df_inv.barrio != None)]
    print("Propiedades en DF_INV luego de primer filtro", len(df_inv))
    corr_barrios = []
    new_barrios = []
    
    barrios = list(df_inv.barrio.unique())
    print("Cantidad de barrios con probs", len(df_inv.barrio.unique()))
    #print(barrios)
    for barrio in barrios:
        dg = df_inv[df_inv.barrio == barrio]
        dg = dg[["prob_investor_0", "price_m2"]].dropna()
        if len(dg) > 0:
            corr = np.corrcoef(dg["prob_investor_0"], dg["price_m2"])[0,1]
            #if np.isnan(corr):
            #    corr = 0.0
            if not np.isnan(corr):
                new_barrios.append(barrio)
                corr_barrios.append(corr)
    
    df_corr = pd.DataFrame({
        "barrio": new_barrios,
        "Corr_prob_investor_price": corr_barrios
    })
    
    print(df_corr)
    
    df_inv = df_inv.merge(df_corr, on="barrio", how="inner")
    print(df_inv[["id", "barrio", "Corr_prob_investor_price"]])
    
    print("******Cantidad de barrios********", len(df_inv.barrio.unique()))
    
    #1/0
    
    print(len(df_filt))
    
    plot_neighborhood_with_ranking(
        df_inv,
        value_label="prob_investor_0",
        path="./data/reportes/paper/mapas/madrid_B11.html",
        barrios_dict=barrios_dict,
        min_conteo=5,
        fill_opacity=0.5,
        font_size=12
    )
    
    
    plot_neighborhood_with_ranking(
        df_inv,
        value_label="Corr_prob_investor_price",
        path="./data/reportes/paper/mapas/madrid_B12.html",
        barrios_dict=barrios_dict,
        min_conteo=5,
        fill_opacity=0.5,
        font_size=12
    )
    
    
    plot_neighborhood_with_ranking(
        df_inv,
        value_label="price_m2",
        path="./data/reportes/paper/mapas/madrid_B13.html",
        barrios_dict=barrios_dict,
        min_conteo=5,
        fill_opacity=0.5,
        font_size=12
    )
    
    
    df_filt = df[df.ROI.isna() == False]
    df_filt = df_filt[(df_filt.municipality == "Madrid") & (df_filt.barrio != None)]
    
    plot_neighborhood_with_ranking(
        df_filt,
        value_label="ROI",
        path="./data/reportes/paper/mapas/madrid_B21.html",
        barrios_dict=barrios_dict,
        min_conteo=5,
        fill_opacity=0.5,
        font_size=12
    )
    
    
    
    """
    fig = plot_neighborhood_with_sidebar(
        df_filt,
        value_label="ROI",
        path="./data/reportes/paper/madrid_ranking_tabla.html",
        barrios_dict=barrios_dict,
        min_conteo=5,
        fill_opacity=0.5,
        font_size=12
    )
    """
    
    
    df_inv = pd.read_pickle("./data/paper/data_madrid_probs.pkl")
    df_inv = df_inv[df_inv.municipality == "Madrid"]
    corr_barrios = []
    barrios = list(df_inv.barrio.unique())
    print("Cantidad de barrios con probs", len(df_inv.barrio.unique()))
    #print(barrios)
    for barrio in barrios:
        dg = df_inv[df_inv.barrio == barrio]
        dg = dg[["prob_investor_0", "price_m2"]].dropna()
        corr = np.corrcoef(dg["prob_investor_0"], dg["price_m2"])[0,1]
        if np.isnan(corr):
            corr = 0.0
        corr_barrios.append(corr)
    
    df_corr = pd.DataFrame({
        "barrio": barrios,
        "Corr_prob_investor_price": corr_barrios
    })

    df = pd.read_pickle(input_file)

    import numpy as np
    import plotly.graph_objects as go
    
    import numpy as np
    import plotly.graph_objects as go
    
    def plot_topN_roi_map(
        df, lat_col, lon_col, roi_col, path,
        marker_symbol="cross",
        marker_color="red",
        marker_size=12,
        bubble_scale=2.5,
        bubble_opacity=0.25,
        map_zoom=13,
        N = 10,
    ):
        """
        Mapa con “globito” (círculo semitransparente) y cruz encima
        en las 10 propiedades de mayor ROI.
        """
        # 1) Top 10 por ROI
        df_top = df.nlargest(N, roi_col)
    
        # 2) Centro del mapa
        center_lat = df_top[lat_col].mean()
        center_lon = df_top[lon_col].mean()
    
        fig = go.Figure()
    
        # 3) Capa “globito” – un círculo semitransparente grande
        fig.add_trace(go.Scattermapbox(
            lat    = df_top[lat_col],
            lon    = df_top[lon_col],
            mode   = "markers",
            marker = dict(
                symbol   = "circle",
                size     = marker_size * bubble_scale,
                color    = marker_color,
                opacity  = bubble_opacity,
                allowoverlap=True
            ),
            hoverinfo="skip",
            showlegend=False
        ))
    
        # 4) Capa símbolo + texto
        fig.add_trace(go.Scattermapbox(
            lat           = df_top[lat_col],
            lon           = df_top[lon_col],
            mode          = "markers+text",
            marker        = dict(
                symbol = marker_symbol,
                size   = marker_size,
                color  = marker_color
            ),
            text          = df_top[roi_col].round(2).astype(str),
            textposition  = "top center",
            hovertemplate = (
                f"{roi_col}: "+"%{text}<br>"+
                f"lat: "+"%{lat:.5f}<br>"+
                f"lon: "+"%{lon:.5f}<extra></extra>"
            ),
            showlegend=False
        ))
    
        # 5) Layout y export
        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox=dict(center=dict(lat=center_lat, lon=center_lon),
                        zoom=map_zoom),
            margin=dict(t=0, b=0, l=0, r=0),
            title=f"Top {N} propiedades por {roi_col}"
        )
    
        fig.write_html(path, include_plotlyjs="cdn")
        return fig


    df_filt = df[df.ROI.isna() == False]
    df_filt = df_filt[(df_filt.municipality == "Madrid") & (df_filt.barrio != None)]
    
    fig = plot_topN_roi_map(
        df=df_filt,
        lat_col="latitude_x",
        lon_col="longitude_x",
        roi_col="ROI",
        N = 10,
        path="./data/reportes/paper/mapas/madrid_B22_10.html",
        marker_symbol="cross",    # o "marker", "circle", "star", etc.
        marker_color="blue",
        marker_size=16,
        map_zoom=13
    )
    
    fig = plot_topN_roi_map(
        df=df_filt,
        lat_col="latitude_x",
        lon_col="longitude_x",
        roi_col="ROI",
        N = 20,
        path="./data/reportes/paper/mapas/madrid_B22_20.html",
        marker_symbol="cross",    # o "marker", "circle", "star", etc.
        marker_color="blue",
        marker_size=16,
        map_zoom=13
    )
    
    fig = plot_topN_roi_map(
        df=df_filt,
        lat_col="latitude_x",
        lon_col="longitude_x",
        roi_col="ROI",
        N = 30,
        path="./data/reportes/paper/mapas/madrid_B22_30.html",
        marker_symbol="cross",    # o "marker", "circle", "star", etc.
        marker_color="blue",
        marker_size=16,
        map_zoom=13
    )
    


## STEP 9 - FINAL TABLES


def final_tables(input_file="./data/paper/data_ROI.pkl"):
    pass # REVISAR!


    