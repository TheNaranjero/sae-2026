import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

N_SIMULATIONS = 1_000
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


def load_simulation_data(n_simulations, filename="simulacion.csv"):
    """
    Lee el csv de la simulación.
    Elimina las filas sin usar (donde los rechazos son cero).
    Realiza los chequeos:
        rechazos máximos menor a la cantidad de simulaciones
        rechazos mínimos menor al 5% de la cantidad de simulaciones
        cantidad de filas para cada combinación de fatores es la correcta
    Reasigna nombres
    Ordena las variables categóricas

    INPUT 
    n_simulations : int número de simulaciones
    filename : str, nombre del archivo csv
    
    OUTPUT
    df: pd.DataFrame.
    """

    # Read CSV
    df = pd.read_csv(filename)

    # Keep only rows with at least one rejection
    df = df[df["rejections"] != 0].copy()

    # Validaciones en base a los rechazos

    min_rej = df["rejections"].min()
    max_rej = df["rejections"].max()


    if min_rej < 0:
        raise ValueError("rechazos negativos")

    if min_rej < n_simulations * 0.05:
        print("n_simulations puede estar mal especificado, todos los escenarios tienen demasiados rechazos")
    
    if max_rej > n_simulations:
         raise ValueError("n_simulations mal especificado, algún escenario supera el máximo de rechazos")

    if max_rej < n_simulations *0.95:
        print("n_simulations puede estar mal especificado, ningún escenario alcanza el 95% del máximo de rechazos")

    # Renombra y ordena cada variable

    ## dist_name

    df["dist_name"] = df["dist_name"].replace({
        "normal": "Normal",
        "t5": "Exponential",
        "exponential": "Exponencial",
        "lognormal": "Lognormal"
    })

    df["dist_name"] = pd.Categorical(
        df["dist_name"],
        categories=["Normal", "t(gl = 5)", "Exponencial", "Lognormal"],
        ordered=True
    )

     ## center_name

    df["center_name"] = df["center_name"].replace({
        "mean": "M. Aritmética",
        "trim_mean_1": "M. Modificada",
        "trim_mean_25": "M. Intercuartil",
        "median": "Mediana",
        "TukeyM": "M. Tukey",


    })

    df["center_name"] = pd.Categorical(
        df["center_name"],
        categories=["M. Aritmética", "M. Modificada", "M. Intercuartil", "Mediana", "M. Tukey"],
        ordered=True
    )

    ## Test

    df["test"] = df["test"].replace({
        "ANOVA": "ANOVA",
        "ANOVA_permutation": "Permutación",
        "ANOVA_Welch": "Welch",
        "ANOVA_Log": "ANOVA Log",
        "ANOVA_Raiz": "ANOVA Raiz",
        "Kruskal_Wallis": "Kruskal-Wallis",
        "Van_Der_Waerden": "Van Der Waerden",
        "ANOVA_winsorizado": "Winsorizado",
        "MLG_gamma" : "MLG Gamma",
        "Cucconi": "Cucconi",
        "Lepage": "Lepage",
        "Mood" : "Mood",
    })

    df["test"] = pd.Categorical(
        df["test"], categories=[ 
            "ANOVA",
            "Permutación",
            "Welch",

            "ANOVA Log",
            "ANOVA Raiz",
            "Winsorizado",

            "Kruskal-Wallis",
            "Van Der Waerden",

            "MLG Gamma",

            "Cucconi",
            "Lepage",
            "Mood",],
            ordered=True
        )
    # Cociente de los desvíos estandar

    STD_RATIO = {
        7: 1.25,
        8: 1.50,
        9: 2.00,
        10: 2.50,
        11: 3.00,
        12: 4.00,
        14: 5.00,
    }

    df["Cociente SD"] = (
        df["scenario"]
        .map(STD_RATIO)
        .fillna(0)
    )

    #Outliers / Porcentaje de contaminación

    CONTAMINATION = {
        14: 1,
        15: 2,
        16: 3,
        17: 4,
        18: 5,
  }

    df["Contaminación"] = (
        df["scenario"]
        .map(CONTAMINATION)
        .fillna(0)
    )

    # Patrón de los desvíos estandar

    SD_PATTERN = {
    ## Escenario , Cantidad de grupos
        (19, 4): "Uno (4)",
        (20, 4): "Mitad (4)",
        (21, 4): "Todos (4)",

        (19, 8): "Uno (8)",
        (20, 8): "Mitad (8)",
        (21, 8): "Todos (8)",
    }

    df["Patrón SD"] = (
        df[["scenario", "group"]]
        .apply(tuple, axis=1)
        .map(SD_PATTERN)
        .fillna("")
    )

    return df


def save_figure(fig, name):
    fig.savefig(
        FIGURES_DIR / f"{name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

df = load_simulation_data(N_SIMULATIONS)
print( df.head() )

### Convierte a lista
def ensure_list(x):
    '''
    Convierte cualquier elemento , tupla o set en una lista.
    INPUT: value, list, tuple, set
    OUTPUT: list
    '''
    if isinstance(x, (list, tuple, set)):
        return list(x)
    else:
        return [x]

# ----------------------------
# Okabe-Ito palette
# ----------------------------

OKABE_ITO = {
    "BLACK" : "#000000",
    "ORANGE": "#E69F00",
    "SKYBLUE" : "#56B4E9",
    "GREENBLUE" : "#009E73",
    "YELLOW" : "#F0E442",
    "BLUE" : "#0072B2",
    "REDORANGE" : "#D55E00",
    "MAGENTA" : "#CC79A7",
}


#### GRAPHS

def scatter_plot(
    scenario,
    group = 4,
    sample_size = 20,
    dist_name = ["Normal", "Exponencial"],

    dot_color = OKABE_ITO["SKYBLUE"],
    x_label = "Porcentaje de Rechazo (%)",
    y_label = "Medida de Centralidad",
    title_label = "Error Tipo I para las distintas medidas de centtralidad",

    x_ticks = [0, 20, 40, 60, 80, 100],
    x_lim = (0,100),

    lines = None,

    note = None,

    ):
    #convierte todo a lista
    scenario = ensure_list(scenario)
    group = ensure_list(group)
    sample_size = ensure_list(sample_size)
    dist_name = ensure_list(dist_name)
    lines = ensure_list(lines)




    #Aplica filtros
    plot_df = df.loc[
        (df["scenario"].isin(scenario))
        & (df["group"].isin(group) )
        & (df["sample_size"].isin(sample_size))
        & (df["dist_name"].isin(["Normal", "Exponencial"]))
    ].copy()

    plot_df["dist_name"] = (
        plot_df["dist_name"]
        .cat.remove_unused_categories()
    )

    #Porcentaje de rechazo

    plot_df["rejection_rate"] = (
        plot_df["rejections"] / N_SIMULATIONS * 100
    )

    # ----------------------------
    # Tema
    # ----------------------------

    sns.set_theme(
        style="whitegrid",
        context="talk",
    )

    plt.rcParams["font.family"] = "Atkinson Hyperlegible"

    plt.rcParams["font.size"] = 24
    plt.rcParams["axes.titlesize"] = 28
    plt.rcParams["axes.labelsize"] = 26
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20
    plt.rcParams["legend.fontsize"] = 20

    # ----------------------------
    # Plot
    # ----------------------------

    g = sns.FacetGrid(
        plot_df,
        col="dist_name",
        sharex=True,
        sharey=True,
        height=6,
        aspect=1.1,
        despine=False,
    )

    g.map_dataframe(
        sns.stripplot,
        x="rejection_rate",
        y="center_name",
        jitter=0.15,
        size=9,
        color=dot_color,
        alpha=0.80,
    )

    # ----------------------------
    # Labels
    # ----------------------------

    g.set_axis_labels(
        "",
        "",
    )

    # X label común

    g.figure.supxlabel(
        x_label,
        fontsize=28,
        y = 0.05
    )
    #Y label común

    g.figure.supylabel(
        y_label,
        fontsize=28,
        x=0.02
    )

    g.set_titles("{col_name}")

    # 2. Add the shared main title
    g.fig.suptitle(title_label, fontsize=30)
    # 3. Adjust spacing so the title doesn't overlap the subplots
    g.fig.subplots_adjust(top=0.85)


    # ----------------------------
    # Lines
    # ----------------------------
    if pd.isnull(lines).any():
        pass
    else:
        for ax in g.axes.flat:
            for line in lines:
                # Nominal 5% ± 2.5%
                ax.axvline(
                    line,
                    color="gray",
                    linestyle="--",
                    linewidth=2,
                    zorder=0,
                )

        

    ax.set_xlim(x_lim)
    ax.set_xticks(x_ticks)

    if pd.isnull(note):
        pass
    else:
        g.figure.text(
            0.01,
            0.02,
            note,
            ha="left",
            fontsize=16,
            color="dimgray",
        )

    g.figure.tight_layout()

    return g.figure

### barplot


def bar_plot(
    scenario,
    y_var,
    group = 4,
    sample_size = 20,
    dist_name = ["Normal", "Exponencial"],
    center_name = "Mediana",

    dot_color = OKABE_ITO["SKYBLUE"],
    x_label = "Porcentaje de Rechazo (%)",
    y_label = "Medida de Centralidad",
    title_label = "Error Tipo I para las distintas medidas de centtralidad",

    x_ticks = [0, 20, 40, 60, 80, 100],
    x_lim = (0,100),

    lines = None,

    note = None,

    ):
    #convierte todo a lista
    scenario = ensure_list(scenario)
    group = ensure_list(group)
    sample_size = ensure_list(sample_size)
    dist_name = ensure_list(dist_name)
    lines = ensure_list(lines)
    center_name = ensure_list(center_name)




    #Aplica filtros
    plot_df = df.loc[
        (df["scenario"].isin(scenario))
        & (df["group"].isin(group) )
        & (df["sample_size"].isin(sample_size))
        & (df["center_name"].isin(center_name))
        & (df["dist_name"].isin(["Normal", "Exponencial"]))
    ].copy()

    plot_df["dist_name"] = (
        plot_df["dist_name"]
        .cat.remove_unused_categories()
    )

    #Porcentaje de rechazo

    plot_df["rejection_rate"] = (
        plot_df["rejections"] / N_SIMULATIONS * 100
    )

    # ----------------------------
    # Tema
    # ----------------------------

    sns.set_theme(
        style="whitegrid",
        context="talk",
    )

    plt.rcParams["font.family"] = "Atkinson Hyperlegible"

    plt.rcParams["font.size"] = 24
    plt.rcParams["axes.titlesize"] = 28
    plt.rcParams["axes.labelsize"] = 26
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20
    plt.rcParams["legend.fontsize"] = 20

    # ----------------------------
    # Plot
    # ----------------------------

    g = sns.FacetGrid(
        plot_df,
        col="dist_name",
        sharex=True,
        sharey=True,
        height=6,
        aspect=1.1,
        despine=False,
    )

    g.map_dataframe(
        sns.barplot,
        x="rejection_rate",
        y= y_var,
        errorbar=None,
        color=dot_color,
        alpha=0.80,
    )

    # ----------------------------
    # Labels
    # ----------------------------

    g.set_axis_labels(
        "",
        "",
    )

    # X label común

    g.figure.supxlabel(
        x_label,
        fontsize=28,
        y = 0.05
    )
    #Y label común

    g.figure.supylabel(
        y_label,
        fontsize=28,
        x=0.02
    )

    g.set_titles("{col_name}")

    # 2. Add the shared main title
    g.fig.suptitle(title_label, fontsize=30)
    # 3. Adjust spacing so the title doesn't overlap the subplots
    g.fig.subplots_adjust(top=0.85)


    # ----------------------------
    # Lines
    # ----------------------------
    if pd.isnull(lines).any():
        pass
    else:
        for ax in g.axes.flat:
            for line in lines:
                # Nominal 5% ± 2.5%
                ax.axvline(
                    line,
                    color="gray",
                    linestyle="--",
                    linewidth=2,
                    zorder=0,
                )

        

    ax.set_xlim(x_lim)
    ax.set_xticks(x_ticks)

    if pd.isnull(note):
        pass
    else:
        g.figure.text(
            0.01,
            0.02,
            note,
            ha="left",
            fontsize=16,
            color="dimgray",
        )

    g.figure.tight_layout()

    return g.figure


def line_plot(
    scenario,
    x_var,
    hue_var,
    group = 4,
    sample_size = 20,
    dist_name = ["Normal", "Exponencial"],
    center_name = "Mediana",
    test = ["ANOVA", "Permutación","ANOVA Raiz", "Welch"],

    palette={
        "ANOVA" : OKABE_ITO["SKYBLUE"], 
        "Permutación": OKABE_ITO["ORANGE"] ,
        "ANOVA Raiz": OKABE_ITO["MAGENTA"], 
        "Welch": OKABE_ITO["GREENBLUE"]
    },

    dot_color = OKABE_ITO["SKYBLUE"],
    x_label = "Porcentaje de Rechazo (%)",
    y_label = "Medida de Centralidad",
    title_label = "Error Tipo I para las distintas medidas de centtralidad",

    y_ticks = [0, 20, 40, 60, 80, 100],
    y_lim = (0,100),

    lines = None,

    note = None,

    ):
    #convierte todo a lista
    scenario = ensure_list(scenario)
    group = ensure_list(group)
    sample_size = ensure_list(sample_size)
    dist_name = ensure_list(dist_name)
    lines = ensure_list(lines)
    center_name = ensure_list(center_name)
    test = ensure_list(test)





    #Aplica filtros
    plot_df = df.loc[
        (df["scenario"].isin(scenario))
        & (df["group"].isin(group) )
        & (df["sample_size"].isin(sample_size))
        & (df["center_name"].isin(center_name))
         & (df["test"].isin(test))
        & (df["dist_name"].isin(dist_name))
    ].copy()

    plot_df["dist_name"] = (
        plot_df["dist_name"]
        .cat.remove_unused_categories()
    )

    plot_df["test"] = (
        plot_df["test"]
        .cat.remove_unused_categories()
    )

    #Porcentaje de rechazo

    plot_df["rejection_rate"] = (
        plot_df["rejections"] / N_SIMULATIONS * 100
    )

    # ----------------------------
    # Tema
    # ----------------------------

    sns.set_theme(
        style="whitegrid",
        context="talk",
    )

    plt.rcParams["font.family"] = "Atkinson Hyperlegible"

    plt.rcParams["font.size"] = 24
    plt.rcParams["axes.titlesize"] = 28
    plt.rcParams["axes.labelsize"] = 26
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20
    plt.rcParams["legend.fontsize"] = 20

    # ----------------------------
    # Plot
    # ----------------------------

    g = sns.FacetGrid(
        plot_df,
        col="dist_name",
        sharex=True,
        sharey=True,
        height=6,
        aspect=1.1,
        despine=False,
    )

    g.map_dataframe(
        sns.lineplot,
        y="rejection_rate",
        x= x_var,
        hue= hue_var,
        style=hue_var,
        palette = palette,
        markers=True,
        dashes=True,
        linewidth=3,
    )

    g.add_legend(title="Pruebas")
    g.legend.set_loc("center left")
    g.legend.set_bbox_to_anchor((1.02, 0.5))

    
    # ----------------------------
    # Labels
    # ----------------------------

    g.set_axis_labels(
        "",
        "",
    )

    # X label común

    g.figure.supxlabel(
        x_label,
        fontsize=28,
        y = 0.05
    )
    #Y label común

    g.figure.supylabel(
        y_label,
        fontsize=28,
        x=0.02
    )

    g.set_titles("{col_name}")

    # 2. Add the shared main title
    g.fig.suptitle(title_label, fontsize=30)
    # 3. Adjust spacing so the title doesn't overlap the subplots
    g.fig.subplots_adjust(top=0.85, right = 0.82)


    # ----------------------------
    # Lines
    # ----------------------------
    if pd.isnull(lines).any():
        pass
    else:
        for ax in g.axes.flat:
            for line in lines:
                # Nominal 5% ± 2.5%
                ax.axhline(
                    line,
                    color="gray",
                    linestyle="--",
                    linewidth=2,
                    zorder=0,
                )

        

    ax.set_ylim(y_lim)
    ax.set_yticks(y_ticks)

    if pd.isnull(note):
        pass
    else:
        g.figure.text(
            0.01,
            0.02,
            note,
            ha="left",
            fontsize=16,
            color="dimgray",
        )

    g.figure.tight_layout()

    return g.figure





## Medidas de Centralidad 


fig = scatter_plot(
    scenario = 1,

    x_label = "Porcentaje de Rechazo (%)",
    y_label = "Medida de Centralidad",
    title_label = "Gráfico 1: Error Tipo I para las Distintas Medidas de Centralidad",

    x_ticks = [0, 10, 20,30, 40],
    x_lim = (0,40),

    lines = [2.5, 7.5],

    note = "Para distribuciones asimétricas, el ANOVA winsorizado junto a la M. Artimética o M. Modificada excede el 40% de rechazo."
)

save_figure(fig = fig, name = "Grafico 1 Error Tipo I para las Distintas Medidas de Centralidad")



fig = scatter_plot(
    scenario = 2,

    x_label = "Porcentaje de Rechazo (%)",
    y_label = "Medida de Centralidad",
    title_label = "Gráfico 2: Potencia para las Distintas Medidas de Centralidad",

    x_ticks = [0, 20, 40,60, 80, 100],
    x_lim = (0,100),

    lines = [80],

)

save_figure(fig = fig, name = "Grafico 2 Potencia para las Distintas Medidas de Centralidad")


## Pruebas


fig = bar_plot(
    scenario = 1,
    y_var = "test",
    center_name = "Mediana",

    x_label = "Porcentaje de Rechazo (%)",
    y_label = "Prueba de Localización",
    title_label = "Gráfico 3: Error Tipo I para las Distintas Pruebas",

    x_ticks = [0, 10, 20, 30, 40],
    x_lim = (0,40),

    lines = [2.5, 7.5],

)
save_figure(fig = fig, name = "Grafico 3 Error Tipo I para las Distintas Pruebas")


fig = bar_plot(
    scenario = 2,
    y_var = "test",
    center_name = "Mediana",

    x_label = "Porcentaje de Rechazo (%)",
    y_label = "Prueba de Localización",
    title_label = "Gráfico 4: Potencia para las Distintas Pruebas",

    x_ticks = [0,20 ,40, 60, 80, 100],
    x_lim = (0,100),

    lines = [80],

)
save_figure(fig = fig, name = "Grafico 4 Potencia para las Distintas Pruebas")

## Tamaño muestral

fig = line_plot(
    scenario=3,
    x_var = "sample_size",
    hue_var = "test",
    center_name = "Mediana",
    sample_size = [4,8,12,16,20,24,28,32,36,40],


    x_label = "Tamaño Muestral",
    y_label = "Porcentaje de Rechazo (%)",
    title_label = "Gráfico 5: Error Tipo I según Tamaño Muestral",

    y_ticks = [0, 5, 10, 15, 20],
    y_lim = (0,20),

    lines = [2.5,7.5],
)

save_figure(fig = fig, name = "Grafico 5 Error Tipo I segun Tamano Muestral")


fig = line_plot(
    scenario=5,
    x_var = "sample_size",
    hue_var = "test",
    center_name = "Mediana",
    sample_size = [4,8,12,16,20,24,28,32,36,40],


    x_label = "Tamaño Muestral",
    y_label = "Porcentaje de Rechazo (%)",
    title_label = "Gráfico 5: Potencia según Tamaño Muestral",

    y_ticks = [0, 20, 40, 60, 80, 100],
    y_lim = (0,100),

    lines = [80],
)

save_figure(fig = fig, name = "Grafico 6 Potencia segun Tamano Muestral")



## Cantidad de grupos

fig = line_plot(
    scenario=4,
    x_var = "group",
    hue_var = "test",
    center_name = "Mediana",
    group = [2,4,6,8,10,12,14,16,18,20],


    x_label = "Cantidad de Grupos",
    y_label = "Porcentaje de Rechazo (%)",
    title_label = "Gráfico 7: Error Tipo I según Cantidad de Grupos",

    y_ticks = [0, 5, 10, 15, 20],
    y_lim = (0,20),

    lines = [2.5,7.5],
)

save_figure(fig = fig, name = "Grafico 7 Error Tipo I segun Cantidad de Grupos")



fig = line_plot(
    scenario=6,
    x_var = "group",
    hue_var = "test",
    center_name = "Mediana",
    group = [2,4,6,8,10,12,14,16,18,20],


    x_label = "Cantidad de Grupos",
    y_label = "Porcentaje de Rechazo (%)",
    title_label = "Gráfico 8: Potencia según Cantidad de Grupos",

    y_ticks = [0, 20, 40, 60, 80, 100],
    y_lim = (0,100),

    lines = [2.5,7.5],
)

save_figure(fig = fig, name = "Grafico 8 Poteencia segun Cantidad de Grupos")


## 






print("-----------------------------")
print("-----------------------------")
print("done")
print("-----------------------------")
print("-----------------------------")
