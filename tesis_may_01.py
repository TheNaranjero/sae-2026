import csv
import numpy as np
import time
import warnings


from astropy.stats import biweight_location
from functools import partial
from scipy import stats
from scipy.stats import rankdata,  chi2, t, lognorm
from statsmodels.tools.sm_exceptions import DomainWarning

# Ignore only DomainWarning
warnings.filterwarnings("ignore", category=DomainWarning)

def chisqprob(chisq, df):
    return stats.chi2.sf(chisq, df)
###################################
N_SIMULATIONS = 10_000  # CANTIDAD DE SIMULACIONES
N_PERMUTATIONS = 5_000  # CANTIDAD DE PERMUTACIONES EN LOS TESTS

RNG = np.random.default_rng(31_415_926_535)

#Controla los escenarios a evaluar, ignorará todos los escenarios seteados a False
ACTIVE_SCENARIOS ={
    "Central tendency alpha": True,
    "Central tendency power": True,
    "Sample size alpha": True,
    "Sample size power": True,
    "Group number alpha": True,
    "Group number power": True,
    "Outlier": True,
    "Power": True,
    }

##############
'''
PARAMETERS: tabla con los parámetros de cada distribución, utilizados para estandarizar los datos generados por cada función.
'''
PARAMETERS={
    "normal": {"m":0, "sd":1},
    "t5": {"m":0, "sd":np.sqrt(t.stats(df=5, moments='v'))},
    "exponential" : {"m":1, "sd":1},
    "lognormal": {"m": lognorm.stats(s=1, moments='m')   ,  "sd"  : np.sqrt(lognorm.stats(s=1, moments='v'))}
}



#############################################################
#### Tabla chicuadrado para no recalcularla constantemente###
# calcula los valores críticos de la chicuadrado una única vez

CHISQ_TABLE = {}
for k in range(100):
    CHISQ_TABLE[(k + 1)] = chi2.ppf(q=1 - 0.05, df=k + 1)



#############################################################
def get_copy(array_list):
    # Returns a deep copy of a list of NumPy arrays.
    return [arr.copy() for arr in array_list]


########################################################
######      Clase Permutación   ########################
# La idea de esta función es crear un array que me permita reorganizar las muestras
# Por ejemploe si le muestra consiste en grupo1=10,20,40 , grupo2=30,50,60
# Esta función tomará los valores 0,1,2,3,4,5  (porque hay 6 datos en total)
# Y los reorganizará varias veces de forma aleatoria en un único array, ejemplo:
# 5,4,3,2,1,0 , 0,5,1,4,2,3 , 1,0,3,2,5,4
# Debido a que son ordenes aleatorios, repetidos cientos de veces se pretende reutilizarlos
# almacenandolos en un Cache.

class Permutations:
    def __init__(self, n_permutations):
        self.cache = {}
        self.n_permutations = n_permutations

    def get_permutation(self, length):
        if length not in self.cache: #  chequea se la muestra está en el cache
            sample = []
            # va a repetir el proceso 'npermutation' veces
            for _ in range(self.n_permutations):
                # muestrea sin remplazo, resultando en una reorganización de los datos
                # (numeros entre 1 y N)
                sample.append(RNG.choice(range(length), length, replace = False))

             # añade la muestra al cache como un único vector.
            self.cache[length] = np.concatenate(sample)

        return self.cache[length]
    def get_N_permutation(self):
        return self.n_permutations

MY_PERMUTATIONS = Permutations(n_permutations=N_PERMUTATIONS)


###############################################################
######      Funciones de centralidad   ########################


def trim_mean_1(data):
    '''
    Calcula la media modificada / media olimpica. Consiste en remover la menor y mayor observación, y calcular la media aritmética de las restantes.
    ENTRADA: Una lista de valores numéricos
    SALIDA: float, la media modificada de la lista.
    '''
    data_array = np.array(data)
    sorted_data = np.sort(data_array)
    trimmed_data = sorted_data[1:-1]  # Remueve la menor y la mayor observacion

    return np.mean(trimmed_data)


def trim_mean_25(data):
    '''
    Calcula la media intercuartil /recortada al 25%. Consiste en remover las observaciones dentro del primer y cuarto cuartil, 
    y calcular la media aritmética de las restantes.
    ENTRADA: Una lista de valores numéricos
    SALIDA: float, la media intercuartil de la lista. 
    '''    
    data_array = np.array(data)
    sorted_data = np.sort(data_array)
    cut = (
        len(data) // 4
    )  # calcula el punto de corte para remover el 25% superior e inferior de los datos
    trimmed_data = sorted_data[cut:-cut]  # Remueve ambos extremos

    return np.mean(trimmed_data)


def TukeyM(data, c = 6):
    '''
    Calcula la media bicuadrada de tukey. 
    Da peso a cada observación en base a su distancia a la mediana.
    Valores extremos (con distancias de 6 MAD* o más de la mediana, son ignorados).
    ajusta los pesos por la función bicuadrada 
    *MAD = Desvíos abslutos medianos de la mediana
    Ver:
        https://docs.astropy.org/en/stable/api/astropy.stats.biweight_location.html

    ENTRADA: Una lista de valores numéricos
    SALIDA: float, la media  de la lista ponderada por la función bicuadrada
    '''  
    tukey_mean = biweight_location(data, c = c)
    return tukey_mean


##############################################################
######      Funciones de los Tests   ########################

# Advertencia: Originalmente este trabajo se planteo para considerar tamaños muestrales desiguales
# Las formulas podrían reescribirse de forma mucho más facil para tamaños iguales.


#####      ANOVA  tradicional      ######
def ANOVA_oneway(data):
    '''
    Aplica el test tradicional ANOVA.
    ENTRADA: Una lista de listas, de preferencia del mismo tamaño, todos valores numéricos
    SALIDA: Entero,  1 si el test rechaza, 0 si el test no rechaza
    '''
    _, pvalue = stats.f_oneway(*data)
    return 1 * (pvalue < 0.05)


#####ANOVA por permutaciones######
def ANOVA_permutation(data, permutation_factory = None):
    '''
    Aplica el test ANOVA exacto. Randomiza los datos y calcula el estadístico F en cada caso,
    este resultado es utilizado para aproximar la distribución del estadístico F y obtener el valor crítico.
    Compara este valor crítico con el estadístico F para la muestra.

    ENTRADA: 
        data = Una lista de listas, las listas deben ser dle mismo tamaño y compuestas de valores numéricos
        permutation_factory = objeto de la clase Permutation

    SALIDA: Entero,  1 si el test rechaza, 0 si el test no rechaza
    '''
    if permutation_factory is None:
        permutation_factory = Permutations(n_permutations=N_PERMUTATIONS)

    #concatena todas las listas en una único array:
    original_data = np.array(np.concatenate(data))

    N = len(original_data)  #cantidad de observaciones
    K = len(data)           #cantidad de grupos
    nk = N // K             #cantidad de observaciones por grupo
    nper = permutation_factory.get_N_permutation() #cantidad de permutaciones

    # Convierte todo los datos a un único vector para poder permutarlos
    squared_data = original_data**2
    # Permuta los datos
    permutations = permutation_factory.get_permutation(N)

    permuted_data = original_data[permutations]


    # Media general
    overall_mean = original_data.mean()
    # Suma de cuadrados sobre los datos originales
    SST = squared_data.sum() - overall_mean**2 * N

    # degres of freedom,  K= cantidad de grupos, N=cantidad de observaciones
    dfB = K - 1
    dfW = N - K

    # Se crea una matriz, donde cada fila es una permutación
    # Esto va a permitir aplicar los calculos del ANOVA a cada fila de forma eficiente
    matrix = permuted_data.reshape(nper, N)  # rows = permutaciones, columns = N

    # Initialize starting column index
    start_col = 0
    # Matriz de ceros
    SSB = np.zeros(matrix.shape[0])

    # divide las observaciones según su grupo ,
    # para cada permutacion y grupo obtiene la media. Luego calcula la SSB (Sum of Squares Between)
    for i in data:
        submatrix = matrix[
            :, start_col : start_col + nk
        ]  # matrix tiene los datos permutados, cada fila es una permutación distinta
        row_means = submatrix.mean(axis=1)

        # SSB:
        # Eleva al cuadrado la media de cada grupo, multiplica por el tamaño de cada grupo
        # sumalos
        SSB += (row_means**2) * nk
        start_col += nk  # actualiza el index para pasar al siguiente grupo

    # Resta la media general al cuadrado, mutiplicada por el tamaño total
    # obteniendo la SSB

    SSB = SSB - overall_mean**2 * N
    SSW = (
        SST - SSB
    )  # Sum of Squares Within (o también conocida como la suma de cuardados del error)

    # El resultado es un vecor conteniendo una gran cantidad de estadísticos F
    # producto de las permutaciones
    F = (SSB / dfB) / (SSW / dfW)

    # Repito el proceso, pero para los datos originales
    start_col = 0
    SSB = 0

    for i in data:
        submatrix = original_data[start_col : start_col + nk]  # datos originales
        row_means = submatrix.mean()

        SSB += (row_means**2) * nk
        start_col += nk
    SSB = SSB - overall_mean**2 * N
    SSW = SST - SSB
    F_statistic = (SSB / dfB) / (SSW / dfW)

    p_value = (F >= F_statistic).sum() / len(
        F
    )  # Proporción de veces que la F muestral fue mayor que la F permutada

    return 1 * (p_value < 0.05)  # devuelve un 1 si es significativo


######    KruskalWallis   ##########
def KruskalWallis(data):
    '''
    Aplica el test de KruskalWallis, aproximando su distribución por una chicuadrado. No corrige por empates. 

    ENTRADA:
        data = Una lista de listas, las listas deben ser del mismo tamaño y compuestas de valores numéricos.

    SALIDA: Entero,  1 si el test rechaza, 0 si el test no rechaza
    '''

    combined_data = np.concatenate(data)


    K = len(data)           # cantidad de grupos
    N = len(combined_data)  # cantidad de observaciones
    nk = N // K             # tamaño de los grupos (todos del mismo tamaño)

    rank_data = rankdata(combined_data)  # asigna rangos a los datos

    mean_rank_data = np.mean(np.split(rank_data, K), axis=1)

    d = N * (N + 1) / 12 / nk
    KW = np.sum(mean_rank_data**2) / d - 3 * (N + 1)    # Formula de trabajo de KruskalWallis

    return 1 * (KW > CHISQ_TABLE[(K - 1)])              # devuelve un 1 si es significativo



#################################################################
##########    DISTRIBUCIONES     ################################

# La siguiente tabla devuelve las funciones estandarizadas.
# Los valores fueron obtenidos por simulación o por la formula cuando fue posible.


def get_rng(random_seed):
    """
    Permite elegir entre usar un generador de números aleatorios predefinido o bien crear uno en el momento
    ENTRADA:un objeto generador de números aleatorios, un valor entero de semilla, o None.
    SALIDA:un objeto generador de números aleatorios
    """
    if random_seed is None:
        return np.random.default_rng()
    if isinstance(random_seed, int):
        return np.random.default_rng(random_seed)
    return random_seed  # Si random_seed ya era un generador de números aleatorios


def normal(size,  random_seed=None):
    '''
    Genera un np.array de números aleatorios provenientes de la distribución normal estandar
    ENTRADA:
        size= entero, el tamaño del array
        random_seed = un entero (la semilla) o un objeto del tipo RNG. 
    SALIDA: np.array del tipo float de tamaño "size"
        '''
    rng = get_rng(random_seed)
    return rng.normal(size=size)


def t5(size, random_seed=None):
    '''
    Genera un np.array de números aleatorios provenientes de la distribución t con 5 grados de libertad y los estandariza (con la media y desvío poblacional)  para tener media 0 y varianza 1
    ENTRADA: entero, el tamaño del array
    SALIDA: np.array del tipo float de tamaño "size"
    '''
    rng = get_rng(random_seed)
    return rng.standard_t(5, size=size) / PARAMETERS["t5"]["sd"]


    


def exponential(size, random_seed=None):
    '''
    Genera un np.array de números aleatorios provenientes de la distribución exponencial (lambda = 1)  y los estandariza (con la media y desvío poblacional)
    ENTRADA: entero, el tamaño del array
    SALIDA: np.array del tipo float de tamaño "size"
    '''
    rng = get_rng(random_seed)
    return rng.exponential(size=size) - PARAMETERS["exponential"]["m"]


def lognormal(size, random_seed=None):
    '''
    Genera un np.array de números aleatorios provenientes de la distribución lognormal (mu=0, sigma = 1)  y los estandariza (con la media y desvío poblacional)
    ENTRADA: entero, el tamaño del array
    SALIDA: np.array del tipo float de tamaño "size"
    '''
    rng = get_rng(random_seed)
    return rng.lognormal(size=size) / PARAMETERS["lognormal"]["sd"] - PARAMETERS["lognormal"]["m"] / PARAMETERS["lognormal"]["sd"]



## Para los outliers


def normal_mix(size, sd1=1, sd2=1, p1=0.5, random_seed=None):
    '''
    Genera un np.array de números aleatorios provenientes de una mezcla de distribuciones normales con desvío estándar sd1 y sd2 
    ENTRADA:
        size= entero, el tamaño del array
        sd1 = desvío estándar de la primera distribución normal
        sd2 = desvío estándar de la segunda distribución normal  (la que contendrá los outliers)
        p1 = proporción de la primera distribución normal en la mezcla
        random_seed = un entero (la semilla) o un objeto del tipo RNG. 
    SALIDA: np.array del tipo float de tamaño "size"
        '''
    rng = get_rng(random_seed)
    outlier = rng.uniform(size=size) > p1  # para que el generador de números aleatorios avance, evitando que los primeros valores sean siempre los mismos
    output = rng.normal(size=size) * (sd1 * (1 - outlier) + sd2 * outlier)
    return output



# Distribucines a estudiar, cada una con su función generadora de números aleatorios estandarizados (con media 0 y desvío 1)
DISTRIBUTIONS = {
    # symmetric
    "normal": partial(normal , random_seed = RNG),
    "t5": partial(t5 , random_seed = RNG),
    # skewed
    "exponential":partial(exponential , random_seed = RNG),
    "lognormal": partial(lognormal , random_seed = RNG),

}
# Funciones de tendencia central
CENTER_FUNCTIONS = {
    "mean": np.mean,
    "trim_mean_1": trim_mean_1,
    "trim_mean_25": trim_mean_25,
    "TukeyM": partial(TukeyM, c=6 ),
    "median": np.median,
}


CENTER_FUNCTIONS_SELECTED= {
    "median": np.median
}

SAMPLE_SIZES = [4, 8, 12,  16, 20, 24,  28, 30, 32, 36, 40,  100, 250, 500, 750, 1000  ]
SAMPLE_SIZES_SELECTED = [4, 16, 30]
GROUPS = [4]
GROUPS_SELECTED = [4]
EFFECTS = [1.25, 1.50, 2, 2.50, 3, 4, 5]


SCENARIOS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]


TESTS = {
    "ANOVA": ANOVA_oneway,
    "ANOVA_permutation":  partial(ANOVA_permutation, permutation_factory=MY_PERMUTATIONS),
    "KruskalWallis": KruskalWallis,

}

TESTS_SELECTED = {
    "ANOVA": ANOVA_oneway,
    "ANOVA_permutation":  partial(ANOVA_permutation, permutation_factory=MY_PERMUTATIONS),
    "KruskalWallis": KruskalWallis,

}






output_file = {}

# Crea una lista vacía
for sample_size in SAMPLE_SIZES:
    output_file[sample_size] = {}
    for group in GROUPS:
        output_file[sample_size][group] = {}
        for dist_idx, (dist_name, dist_func) in enumerate(DISTRIBUTIONS.items()):
            output_file[sample_size][group][dist_name] = {}
            for scenario in SCENARIOS:
                output_file[sample_size][group][dist_name][scenario] = {}
                for center_name, center_func in CENTER_FUNCTIONS.items():
                    output_file[sample_size][group][dist_name][scenario][center_name] = {}
                    for test_name, test_func in TESTS.items():
                        output_file[sample_size][group][dist_name][scenario][center_name][test_name] = 0


current_time_human = time.ctime()

# A continuación comienza la situación
# Para optimizar los tiempos se la dividió en escenarios
# En cada escenario solo unas combinaciones específica de atributos fueron estudiados
# (pruebas, medidas de centralidad, tamaño muestral, cantidad de grupos, etc.)
print("starting simulation")

if ACTIVE_SCENARIOS["Central tendency alpha"]:
    print("Escenario 1: Error tipo I para diferentes medidas de centralidad")
    print(f"Start time (human-readable): {current_time_human}")

    scenario = 1  # error tipo I para diversas medidas de centralidad




    for simulation in range(N_SIMULATIONS):
        if simulation % 10 == 0:
            print(simulation)
        for sample_size in SAMPLE_SIZES_SELECTED:
            for groups in GROUPS_SELECTED:
                for dist_idx, (dist_name, dist_func) in enumerate(DISTRIBUTIONS.items()):
                    original_data = np.array_split(
                        np.array(dist_func(size=groups * sample_size)), groups
                    )
                    # reemplaza data por la original
                    data = get_copy(original_data)

                    for center_name, center_func in CENTER_FUNCTIONS.items():
                        abs_dev = [np.abs(np.array(d) - center_func(d)) for d in data]
                        length = len(np.concatenate(abs_dev))
                        permutations = MY_PERMUTATIONS.get_permutation(length=length)
                        for test_name, test_func in TESTS.items():
                            output_file[sample_size][groups][dist_name][scenario][center_name][test_name] += test_func(abs_dev)

if ACTIVE_SCENARIOS["Central tendency power"]:

    print("Escenario 2: Potencia para diferentes medidas de centralidad")
    current_time_human = time.ctime()
    print(f"Start time (human-readable): {current_time_human}")

    scenario = 2  # potencia para diversas medidas de centralidad

    for simulation in range(N_SIMULATIONS):
        if simulation % 10 == 0:
            print(simulation)
        for sample_size in SAMPLE_SIZES_SELECTED:
            for groups in GROUPS_SELECTED:
                for dist_idx, (dist_name, dist_func) in enumerate(DISTRIBUTIONS.items()):
                    original_data = np.array_split(
                        np.array(dist_func(size=groups * sample_size)), groups
                    )
                    # reemplaza data por la original
                    data = get_copy(original_data)

                    data[0] *= 2

                    for center_name, center_func in CENTER_FUNCTIONS.items():
                        abs_dev = [np.abs(np.array(d) - center_func(d)) for d in data]
                        length = len(np.concatenate(abs_dev))
                        permutations = MY_PERMUTATIONS.get_permutation(length=length)

                        for test_name, test_func in TESTS.items():
                            output_file[sample_size][groups][dist_name][scenario][center_name][test_name] += test_func(abs_dev)

if ACTIVE_SCENARIOS["Sample size alpha"]:

    print("Escenario 3: Error tipo I para diferentes tamaños muestrales")
    current_time_human = time.ctime()
    print(f"Start time (human-readable): {current_time_human}")

    scenario = 3  # Error tipo 1





    for simulation in range(N_SIMULATIONS):
        if simulation % 10 == 0:
            print(simulation)
        for sample_size in SAMPLE_SIZES:
            for groups in GROUPS_SELECTED:
                for dist_idx, (dist_name, dist_func) in enumerate(DISTRIBUTIONS.items()):
                    original_data = np.array_split(
                        np.array(dist_func(size=groups * sample_size)), groups
                    )
                    # reemplaza data por la original
                    data = get_copy(original_data)

                    for center_name, center_func in CENTER_FUNCTIONS_SELECTED.items():
                        # print(center_name)
                        abs_dev = [np.abs(np.array(d) - center_func(d)) for d in data]
                        length = len(np.concatenate(abs_dev))
                        permutations = MY_PERMUTATIONS.get_permutation(length=length)
                        for test_name, test_func in TESTS.items():
                            output_file[sample_size][groups][dist_name][scenario][center_name][test_name] += test_func(abs_dev)


if ACTIVE_SCENARIOS["Group number alpha"]:

    print("Escenario 4: Error tipo I para diferente cantidad de grupos")
    current_time_human = time.ctime()
    print(f"Start time (human-readable): {current_time_human}")

    scenario = 4  # Error tipo 1





    for simulation in range(N_SIMULATIONS):
        if simulation % 10 == 0:
            print(simulation)
        for sample_size in SAMPLE_SIZES_SELECTED:
            for groups in GROUPS:
                for dist_idx, (dist_name, dist_func) in enumerate(DISTRIBUTIONS.items()):
                    original_data = np.array_split(
                        np.array(dist_func(size=groups * sample_size)), groups
                    )
                    # reemplaza data por la original
                    data = get_copy(original_data)

                    for center_name, center_func in CENTER_FUNCTIONS_SELECTED.items():
                        # print(center_name)
                        abs_dev = [np.abs(np.array(d) - center_func(d)) for d in data]
                        length = len(np.concatenate(abs_dev))
                        permutations = MY_PERMUTATIONS.get_permutation(length=length)
                        for test_name, test_func in TESTS.items():
                            output_file[sample_size][groups][dist_name][scenario][center_name][test_name] += test_func(abs_dev)


if ACTIVE_SCENARIOS["Sample size power"]:

    print("Escenario 5: Potencia para diferentes tamaños muestrales, solo tests seleccionados")
    current_time_human = time.ctime()
    print(f"Start time (human-readable): {current_time_human}")

    scenario = 5





    for simulation in range(N_SIMULATIONS):
        if simulation % 10 == 0:
            print(simulation)
        for sample_size in SAMPLE_SIZES:
            for groups in GROUPS_SELECTED:
                for dist_idx, (dist_name, dist_func) in enumerate(DISTRIBUTIONS.items()):
                    original_data = np.array_split(
                        np.array(dist_func(size=groups * sample_size)), groups
                    )
                    # reemplaza data por la original
                    data = get_copy(original_data)
                    data[0] *= 2

                    for center_name, center_func in CENTER_FUNCTIONS_SELECTED.items():
                        # print(center_name)
                        abs_dev = [np.abs(np.array(d) - center_func(d)) for d in data]
                        length = len(np.concatenate(abs_dev))
                        permutations = MY_PERMUTATIONS.get_permutation(length=length)
                        for test_name, test_func in TESTS_SELECTED.items():
                            output_file[sample_size][groups][dist_name][scenario][center_name][test_name] += test_func(abs_dev)

if ACTIVE_SCENARIOS["Group number power"]:

    print("Escenario 6: Potencia para diferentes cantidad de grupos, solo tests seleccionados")
    current_time_human = time.ctime()
    print(f"Start time (human-readable): {current_time_human}")

    scenario = 6  # Error tipo 1





    for simulation in range(N_SIMULATIONS):
        if simulation % 10 == 0:
            print(simulation)
        for sample_size in SAMPLE_SIZES_SELECTED:
            for groups in GROUPS:
                for dist_idx, (dist_name, dist_func) in enumerate(DISTRIBUTIONS.items()):
                    original_data = np.array_split(
                        np.array(dist_func(size=groups * sample_size)), groups
                    )
                    # reemplaza data por la original
                    data = get_copy(original_data)
                    data[0] *= 2

                    for center_name, center_func in CENTER_FUNCTIONS_SELECTED.items():
                        # print(center_name)
                        abs_dev = [np.abs(np.array(d) - center_func(d)) for d in data]
                        length = len(np.concatenate(abs_dev))
                        permutations = MY_PERMUTATIONS.get_permutation(length=length)
                        for test_name, test_func in TESTS_SELECTED.items():
                            output_file[sample_size][groups][dist_name][scenario][center_name][test_name] += test_func(abs_dev)


if ACTIVE_SCENARIOS["Power"]:

    print("Escenario 7, 8, 9,10,11,12,13: Potencia, solo tests seleccionados")
    current_time_human = time.ctime()
    print(f"Start time (human-readable): {current_time_human}")

    scenario = [7, 8, 9, 10, 11, 12, 13]


    for simulation in range(N_SIMULATIONS):
        if simulation % 10 == 0:
            print(simulation)
        for sample_size in SAMPLE_SIZES_SELECTED:
            for groups in GROUPS_SELECTED:
                for dist_idx, (dist_name, dist_func) in enumerate(DISTRIBUTIONS.items()):
                    original_data = np.array_split(
                        np.array(dist_func(size=groups * sample_size)), groups
                    )
                    # reemplaza data por la original
                    for i in [0, 1, 2, 3, 4, 5, 6]:
                        data = get_copy(original_data)
                        data[0] *= EFFECTS[i] + 0

                        for center_name, center_func in CENTER_FUNCTIONS_SELECTED.items():
                            abs_dev = [np.abs(np.array(d) - center_func(d)) for d in data]
                            length = len(np.concatenate(abs_dev))
                            permutations = MY_PERMUTATIONS.get_permutation(length=length)
                            for test_name, test_func in TESTS_SELECTED.items():
                                output_file[sample_size][groups][dist_name][scenario[i]][center_name][test_name] += test_func(abs_dev)

if ACTIVE_SCENARIOS["Outlier"]:
    '''
    Escenario 14, 15, 16, 17 y 18: Outliers
    Se generan datos provenientes de una mezcla de distribuciones normales, con una proporción p de datos provenientes de una normal con desvío estándar sd1 y
    el resto de datos provenientes de una normal con desvío estándar sd2.
    Las proporciones p1 y los desvíos estándar sd1 y sd2 fueron seleccionados de forma tal que el desvío estándar de la mezcla sea igual a 1.
    sd2 es el desvío estándar de la distribución que contiene los outliers, por lo que se mantiene constante en 4, mientras que sd1 y p1 varían para mantener el desvío de la mezcla igual a 1.
    '''


    current_time_human = time.ctime()
    print("Escenario  14 , 15 , 16, 17 y 18 : Outliers")
    print(f"Start time (human-readable): {current_time_human}")

    p = [0.99, 0.98, 0.97, 0.96, 0.95]
    sd1 = [0.92, 0.83, 0.73, 0.61, 0.46]  #seleccionaddos para que el desvío de la mezcla sea igual a 1
    sd2 = 4

    scenario = [14, 15, 16, 17, 18]

    for simulation in range(N_SIMULATIONS):
        if simulation % 10 == 0:
            print(simulation)
        for sample_size in SAMPLE_SIZES_SELECTED:
            for groups in GROUPS_SELECTED:
                for i in [0, 1, 2, 3, 4]:
                    dist_name = "normal"  #para guardar los resultados hay que llenar todas las columnas, se le asigna el nombre de la distribución normal por defecto
                    data = np.concatenate([
                        normal_mix(size= sample_size, sd1=sd1[i], sd2=sd2, p1=p[i], random_seed = RNG), #sample_size observaciones de la mezcla de normales con outliers
                        normal(size=(groups - 1) * sample_size, random_seed = RNG)   # (groups-1) x sample_size: total de observaciones provenientes de la normal estandar
                    ])

                    data = np.array_split(data, groups)
               

                    for center_name, center_func in CENTER_FUNCTIONS_SELECTED.items():
                        abs_dev = [np.abs(np.array(d) - center_func(d)) for d in data]
                        length = len(np.concatenate(abs_dev))
                        permutations = MY_PERMUTATIONS.get_permutation(length=length)
                        for test_name, test_func in TESTS_SELECTED.items():
                            output_file[sample_size][groups][dist_name][scenario[i]][center_name][test_name] += test_func(abs_dev)





current_time_human = time.ctime()

print(f"Finish time (human-readable): {current_time_human}")

print("saving")


output_list = []

for sample_size in SAMPLE_SIZES:
    for groups in GROUPS:
        for dist_idx, (dist_name, dist_func) in enumerate(DISTRIBUTIONS.items()):
            for scenario in SCENARIOS:
                for center_name, center_func in CENTER_FUNCTIONS.items():
                    for test_name, test_func in TESTS.items():
                        reject = output_file[sample_size][groups][dist_name][scenario][center_name][test_name]
                        element = [
                            sample_size,
                            groups,
                            dist_name,
                            scenario,
                            center_name,
                            test_name,
                            reject,
                        ]
                        output_list.append(element)
# Save to CSV
csv_file = "tesis.csv"

print(output_list[2])

with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(
        ["sample_size", "group", "dist_name", "scenario", "center_name", "test", "rejections"]
    )
    # Write the data rows
    writer.writerows(output_list)
