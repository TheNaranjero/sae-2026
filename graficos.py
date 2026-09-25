import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

N_SIMULATIONS = 1_000
FIGURES_DIR = Path("figures")
CM_PER_INCH = 2.54
DEFAULT_FIGURE_SIZE_CM = (30, 15)

# ------------------------------------------------------------------------------------------------ #
#                                      Procesamiento de datos                                      #
# ------------------------------------------------------------------------------------------------ #


def load_simulation_data(n_simulations, filename="simulacion.csv"):
    """Lee y prepara los resultados de una simulación.

    Elimina las filas sin rechazos, valida sus valores y convierte las variables
    correspondientes en categorías ordenadas con nombres descriptivos.

    Parameters
    ----------
    n_simulations : int
        Número de simulaciones realizadas en cada escenario.
    filename : str or path-like, default="simulacion.csv"
        Ruta del archivo CSV que contiene los resultados.

    Returns
    -------
    pandas.DataFrame
        Datos filtrados, validados y recodificados.

    Raises
    ------
    ValueError
        Si existen rechazos negativos o si alguno supera el número de simulaciones.

    Warns
    -----
    UserWarning
        Si la cantidad de simulaciones parece incompatible con los rechazos observados.
    """

    DIST_NAME_MAPPING = {
        "normal": "Normal",
        "t5": "t(gl = 5)",
        "exponential": "Exponencial",
        "lognormal": "Lognormal",
    }
    DIST_NAME_LEVELS = list(DIST_NAME_MAPPING.values())

    CENTER_NAME_MAPPING = {
        "mean": "M. Aritmética",
        "trim_mean_1": "M. Modificada",
        "trim_mean_25": "M. Intercuartil",
        "median": "Mediana",
        "TukeyM": "M. Tukey",
    }
    CENTER_NAME_LEVELS = list(CENTER_NAME_MAPPING.values())

    TEST_MAPPING = {
        "ANOVA": "ANOVA",
        "ANOVA_permutation": "Permutación",
        "ANOVA_Welch": "Welch",
        "ANOVA_Log": "ANOVA Log",
        "ANOVA_Raiz": "ANOVA Raiz",
        "Kruskal_Wallis": "Kruskal-Wallis",
        "Van_Der_Waerden": "Van Der Waerden",
        "ANOVA_winsorizado": "Winsorizado",
        "MLG_gamma": "MLG Gamma",
        "Cucconi": "Cucconi",
        "Lepage": "Lepage",
        "Mood": "Mood",
    }
    TEST_LEVELS = list(TEST_MAPPING.values())

    STD_RATIO = {
        7: 1.25,
        8: 1.50,
        9: 2.00,
        10: 2.50,
        11: 3.00,
        12: 4.00,
        13: 5.00,
    }
    CONTAMINATION = {
        14: 1,
        15: 2,
        16: 3,
        17: 4,
        18: 5,
    }
    SD_PATTERN = {
        (19, 4): "Uno (4)",
        (20, 4): "Mitad (4)",
        (21, 4): "Todos (4)",
        (19, 8): "Uno (8)",
        (20, 8): "Mitad (8)",
        (21, 8): "Todos (8)",
    }

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
        warnings.warn(
            "n_simulations puede estar mal especificado, todos los escenarios "
            "tienen demasiados rechazos",
            UserWarning,
            stacklevel=2,
        )

    if max_rej > n_simulations:
        raise ValueError(
            "n_simulations mal especificado, algún escenario supera el máximo de rechazos"
        )

    if max_rej < n_simulations * 0.95:
        warnings.warn(
            "n_simulations puede estar mal especificado, ningún escenario alcanza "
            "el 95% del máximo de rechazos",
            UserWarning,
            stacklevel=2,
        )

    # Renombra y ordena cada variable

    ## dist_name
    df["dist_name"] = df["dist_name"].replace(DIST_NAME_MAPPING)
    df["dist_name"] = pd.Categorical(df["dist_name"], categories=DIST_NAME_LEVELS, ordered=True)

    ## center_name
    df["center_name"] = df["center_name"].replace(CENTER_NAME_MAPPING)
    df["center_name"] = pd.Categorical(
        df["center_name"], categories=CENTER_NAME_LEVELS, ordered=True
    )

    ## Test
    df["test"] = df["test"].replace(TEST_MAPPING)
    df["test"] = pd.Categorical(df["test"], categories=TEST_LEVELS, ordered=True)

    # Cociente de los desvíos estandar
    df["Cociente SD"] = df["scenario"].map(STD_RATIO).fillna(0)

    # Outliers / Porcentaje de contaminación
    df["Contaminación"] = df["scenario"].map(CONTAMINATION).fillna(0)

    # Patrón de los desvíos estandar
    df["Patrón SD"] = df[["scenario", "group"]].apply(tuple, axis=1).map(SD_PATTERN).fillna("")

    return df


def ensure_list(x):
    """Convierte un valor o una colección compatible en una lista.

    Parameters
    ----------
    x : object
        Valor individual o instancia de ``list``, ``tuple`` o ``set``.

    Returns
    -------
    list
        El valor convertido en lista. Si ya era una colección compatible,
        contiene sus elementos; en caso contrario, contiene únicamente ``x``.
    """
    if isinstance(x, (list, tuple, set)):
        return list(x)
    else:
        return [x]


# ------------------------------------------------------------------------------------------------ #
#                                          Visualización                                           #
# ------------------------------------------------------------------------------------------------ #

OKABE_ITO = {
    "BLACK": "#000000",
    "ORANGE": "#E69F00",
    "SKYBLUE": "#56B4E9",
    "GREENBLUE": "#009E73",
    "YELLOW": "#F0E442",
    "BLUE": "#0072B2",
    "REDORANGE": "#D55E00",
    "MAGENTA": "#CC79A7",
}


def configure_plot_theme():
    """Configura el tema predeterminado para todos los gráficos del módulo."""
    sns.set_theme(
        style="whitegrid",
        context="talk",
        rc={
            "font.family": "Atkinson Hyperlegible",
            "font.size": 18,
            "axes.titlesize": 26,  # Títulos de los paneles
            "axes.labelsize": 18,  # Etiquetas de los ejes
            "figure.titlesize": 22,  # Título a nivel figuras
            "figure.labelsize": 18,  # Etiquetas a nivel de figuras
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "grid.linewidth": 0.7,
            "grid.color": "#8D8D8D",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.9,
            "axes.axisbelow": True,
        },
    )


def scatter_plot(
    data,
    scenario,
    group=4,
    sample_size=20,
    dist_name=None,
    dot_color=OKABE_ITO["SKYBLUE"],
    x_label="Porcentaje de Rechazo (%)",
    y_label="Medida de Centralidad",
    x_ticks=None,
    x_lim=(0, 100),
    lines=None,
    band=None,
    band_color=OKABE_ITO["YELLOW"],
    band_alpha=0.50,
    band_linewidth=1.0,
    band_line_alpha=1.0,
    figure_size_cm=DEFAULT_FIGURE_SIZE_CM,
    test=None,
    note=None,
):

    if dist_name is None:
        dist_name = ["Normal", "Exponencial"]

    if x_ticks is None:
        x_ticks = [0, 20, 40, 60, 80, 100]

    if band is not None:
        try:
            band_start, band_end = band
        except TypeError, ValueError:
            raise ValueError("band debe contener exactamente dos límites") from None

    if test is None:
        test = [
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
            "Mood",
        ]

    # convierte todo a lista
    scenario = ensure_list(scenario)
    group = ensure_list(group)
    sample_size = ensure_list(sample_size)
    dist_name = ensure_list(dist_name)
    test = ensure_list(test)
    lines = ensure_list(lines)

    # Aplica filtros
    plot_df = data.loc[
        (data["scenario"].isin(scenario))
        & (data["group"].isin(group))
        & (data["sample_size"].isin(sample_size))
        & (data["dist_name"].isin(["Normal", "Exponencial"]))
        & (data["test"].isin(test))
    ].copy()

    plot_df["dist_name"] = plot_df["dist_name"].cat.remove_unused_categories()

    plot_df["test"] = plot_df["test"].cat.remove_unused_categories()

    # Porcentaje de rechazo
    plot_df["rejection_rate"] = plot_df["rejections"] / N_SIMULATIONS * 100

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
        edgecolor="black",
        linewidth=0.5,
        alpha=0.80,
    )

    # ----------------------------
    # Labels
    # ----------------------------
    g.set_axis_labels(x_label, y_label)
    g.set_titles("{col_name}", size=plt.rcParams["axes.titlesize"])

    # ----------------------------
    # Lines
    # ----------------------------
    if band is not None:
        for ax in g.axes.flat:
            ax.axvspan(
                band_start,
                band_end,
                color=band_color,
                alpha=band_alpha,
                zorder=0,
            )
            for boundary in (band_start, band_end):
                ax.axvline(
                    boundary,
                    color=band_color,
                    alpha=band_line_alpha,
                    linewidth=band_linewidth,
                    zorder=1,
                )

    if pd.isnull(lines).any():
        pass
    else:
        for ax in g.axes.flat:
            for line in lines:
                # Nominal 5% ± 2.5%
                ax.axvline(
                    line,
                    color=OKABE_ITO["REDORANGE"],
                    linestyle="-",
                    linewidth=2,
                    zorder=0.75,
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
            color="dimgray",
        )

    figure_width_cm, figure_height_cm = figure_size_cm
    g.figure.set_size_inches(
        figure_width_cm / CM_PER_INCH,
        figure_height_cm / CM_PER_INCH,
    )
    g.figure.tight_layout()

    return g.figure


def bar_plot(
    data,
    scenario,
    y_var,
    group=4,
    sample_size=20,
    dist_name=None,
    center_name="Mediana",
    dot_color=OKABE_ITO["SKYBLUE"],
    x_label="Porcentaje de Rechazo (%)",
    y_label="Medida de Centralidad",
    x_ticks=None,
    x_lim=(0, 100),
    lines=None,
    band=None,
    band_color=OKABE_ITO["YELLOW"],
    band_alpha=0.20,
    band_linewidth=1.0,
    band_line_alpha=1.0,
    figure_size_cm=DEFAULT_FIGURE_SIZE_CM,
    note=None,
    hue=None,
    palette=None,
    test=None,
    legend = False,
):

    if dist_name is None:
        dist_name = ["Normal", "Exponencial"]
    if x_ticks is None:
        x_ticks = [0, 20, 40, 60, 80, 100]

    if band is not None:
        try:
            band_start, band_end = band
        except TypeError, ValueError:
            raise ValueError("band debe contener exactamente dos límites") from None
    if test is None:
        test = [
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
            "Mood",
        ]

    # convierte todo a lista
    scenario = ensure_list(scenario)
    group = ensure_list(group)
    sample_size = ensure_list(sample_size)
    dist_name = ensure_list(dist_name)
    lines = ensure_list(lines)
    center_name = ensure_list(center_name)
    test = ensure_list(test)

    # Aplica filtros
    plot_df = data.loc[
        (data["scenario"].isin(scenario))
        & (data["group"].isin(group))
        & (data["sample_size"].isin(sample_size))
        & (data["center_name"].isin(center_name))
        & (data["dist_name"].isin(["Normal", "Exponencial"]))
        & (data["test"].isin(test))
    ].copy()

    plot_df["dist_name"] = plot_df["dist_name"].cat.remove_unused_categories()

    plot_df["test"] = plot_df["test"].cat.remove_unused_categories()

    # Porcentaje de rechazo

    plot_df["rejection_rate"] = plot_df["rejections"] / N_SIMULATIONS * 100

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
        y=y_var,
        errorbar=None,
        color=dot_color,
        alpha=1.0,
        hue=hue,
        palette=palette,
        edgecolor="none",
        linewidth=0,
        zorder=2,
        saturation=1,

    )

    #----------------------------
    # Legend
    #----------------------------
    if legend:
        g.add_legend(title="Pruebas")
        g.legend.set_loc("center right")
        g.legend.set_bbox_to_anchor((0.99, 0.5))

    # ----------------------------
    # Labels
    # ----------------------------
    g.set_axis_labels(x_label, y_label)
    g.set_titles("{col_name}", size=plt.rcParams["axes.titlesize"])

    # ----------------------------
    # Lines
    # ----------------------------
    if band is not None:
        for ax in g.axes.flat:
            ax.axvspan(
                band_start,
                band_end,
                color=band_color,
                alpha=band_alpha,
                zorder=0,
            )
            for boundary in (band_start, band_end):
                ax.axvline(
                    boundary,
                    color=band_color,
                    alpha=band_line_alpha,
                    linewidth=band_linewidth,
                    zorder=1,
                )

    if pd.isnull(lines).any():
        pass
    else:
        for ax in g.axes.flat:
            for line in lines:
                # Nominal 5% ± 2.5%
                ax.axvline(
                    line,
                    color=OKABE_ITO["REDORANGE"],
                    linestyle="-",
                    linewidth=2,
                    zorder=0.75,
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
            color="dimgray",
        )

    figure_width_cm, figure_height_cm = figure_size_cm
    g.figure.set_size_inches(
        figure_width_cm / CM_PER_INCH,
        figure_height_cm / CM_PER_INCH,
    )
    
    if legend:
        g.figure.canvas.draw()
        renderer = g.figure.canvas.get_renderer()
        legend_bbox = g.legend.get_window_extent(renderer).transformed(g.figure.transFigure.inverted())
        plot_right = legend_bbox.x0 - 0.02
        g.figure.tight_layout(rect=(0, 0, plot_right, 1))
    else:
        g.figure.tight_layout()
    

    return g.figure


def line_plot(
    data,
    scenario,
    x_var,
    hue_var,
    group=4,
    sample_size=20,
    dist_name=None,
    center_name="Mediana",
    test=None,
    palette=None,
    dot_color=OKABE_ITO["SKYBLUE"],
    x_label="Porcentaje de Rechazo (%)",
    y_label="Medida de Centralidad",
    y_ticks=None,
    y_lim=(0, 100),
    x_ticks=None,
    lines=None,
    band=None,
    band_color=OKABE_ITO["YELLOW"],
    band_alpha=0.20,
    band_linewidth=1.0,
    band_line_alpha=1.0,
    figure_size_cm=DEFAULT_FIGURE_SIZE_CM,
    note=None,
):

    if dist_name is None:
        dist_name = ["Normal", "Exponencial"]
    if y_ticks is None:
        y_ticks = [0, 20, 40, 60, 80, 100]

    if band is not None:
        try:
            band_start, band_end = band
        except TypeError, ValueError:
            raise ValueError("band debe contener exactamente dos límites") from None
    if test is None:
        test = ["ANOVA", "Permutación", "ANOVA Raiz", "Welch"]

    if palette is None:
        palette = {
            "ANOVA": OKABE_ITO["SKYBLUE"],
            "Permutación": OKABE_ITO["ORANGE"],
            "ANOVA Raiz": OKABE_ITO["MAGENTA"],
            "Welch": OKABE_ITO["GREENBLUE"],
        }

    # convierte todo a lista
    scenario = ensure_list(scenario)
    group = ensure_list(group)
    sample_size = ensure_list(sample_size)
    dist_name = ensure_list(dist_name)
    lines = ensure_list(lines)
    center_name = ensure_list(center_name)
    test = ensure_list(test)

    # Aplica filtros
    plot_df = data.loc[
        (data["scenario"].isin(scenario))
        & (data["group"].isin(group))
        & (data["sample_size"].isin(sample_size))
        & (data["center_name"].isin(center_name))
        & (data["test"].isin(test))
        & (data["dist_name"].isin(dist_name))
    ].copy()

    plot_df["dist_name"] = plot_df["dist_name"].cat.remove_unused_categories()

    plot_df["test"] = plot_df["test"].cat.remove_unused_categories()

    # Porcentaje de rechazo

    plot_df["rejection_rate"] = plot_df["rejections"] / N_SIMULATIONS * 100

    # ----------------------------
    # Plot
    # ----------------------------

    g = sns.FacetGrid(
        plot_df,
        col="dist_name",
        sharex=True,
        sharey=True,
        height=9,
        aspect=0.9,
        despine=False,
    )

    g.map_dataframe(
        sns.lineplot,
        y="rejection_rate",
        x=x_var,
        hue=hue_var,
        style=hue_var,
        palette=palette,
        markers=True,
        dashes=True,
        linewidth=5,
        markersize=12,
    )

    g.add_legend(title="Pruebas")
    g.legend.set_loc("center right")
    g.legend.set_bbox_to_anchor((0.99, 0.5))

    # ----------------------------
    # Labels
    # ----------------------------
    g.set_axis_labels(x_label, y_label)
    g.set_titles("{col_name}", size=plt.rcParams["axes.titlesize"])

    # ----------------------------
    # Lines
    # ----------------------------
    if band is not None:
        for ax in g.axes.flat:
            ax.axhspan(
                band_start,
                band_end,
                color=band_color,
                alpha=band_alpha,
                zorder=0,
            )
            for boundary in (band_start, band_end):
                ax.axhline(
                    boundary,
                    color=band_color,
                    alpha=band_line_alpha,
                    linewidth=band_linewidth,
                    zorder=1,
                )

    if pd.isnull(lines).any():
        pass
    else:
        for ax in g.axes.flat:
            for line in lines:
                # Nominal 5% ± 2.5%
                ax.axhline(
                    line,
                    color=OKABE_ITO["REDORANGE"],
                    linestyle="-",
                    linewidth=2,
                    zorder=0.75,
                )

    ax.set_ylim(y_lim)
    ax.set_yticks(y_ticks)

    if x_ticks is not None:
        ax.set_xticks(x_ticks)

    if pd.isnull(note):
        pass
    else:
        g.figure.text(
            0.01,
            0.02,
            note,
            ha="left",
            color="dimgray",
        )

    figure_width_cm, figure_height_cm = figure_size_cm
    g.figure.set_size_inches(
        figure_width_cm / CM_PER_INCH,
        figure_height_cm / CM_PER_INCH,
    )
    g.figure.canvas.draw()
    renderer = g.figure.canvas.get_renderer()
    legend_bbox = g.legend.get_window_extent(renderer).transformed(g.figure.transFigure.inverted())
    plot_right = legend_bbox.x0 - 0.02
    g.figure.tight_layout(rect=(0, 0, plot_right, 1))

    return g.figure


def save_figure(fig, name):
    fig.savefig(FIGURES_DIR / f"{name}.pdf")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ #
#                                      Inicialización                                              #
# ------------------------------------------------------------------------------------------------ #
FIGURES_DIR.mkdir(exist_ok=True)
configure_plot_theme()
simulation_data = load_simulation_data(N_SIMULATIONS)
print(simulation_data.head())

# ------------------------------------------------------------------------------------------------ #
#                                      Generación de gráficos                                      #
# ------------------------------------------------------------------------------------------------ #
fig = scatter_plot(
    data=simulation_data,
    scenario=1,
    x_label="Porcentaje de Rechazo (%)",
    y_label="Medida de Centralidad",
    x_ticks=[0, 10, 20, 30, 40],
    x_lim=(0, 40),
    band=(2.5, 7.5),
    band_color=OKABE_ITO["REDORANGE"],
    band_alpha=0.40,
    band_line_alpha=0.80,
    band_linewidth=1,
    test=["ANOVA", "Permutación", "Kruskal-Wallis"],
    note=None,
)

save_figure(fig=fig, name="Grafico 1 Error Tipo I para las Distintas Medidas de Centralidad")

fig = scatter_plot(
    data=simulation_data,
    scenario=2,
    x_label="Porcentaje de Rechazo (%)",
    y_label="Medida de Centralidad",
    x_ticks=[0, 20, 40, 60, 80, 100],
    x_lim=(0, 100),
    lines=[80],
    test=["ANOVA", "Permutación", "Kruskal-Wallis"],
)

save_figure(fig=fig, name="Grafico 2 Potencia para las Distintas Medidas de Centralidad")

fig = bar_plot(
    data=simulation_data,
    scenario=1,
    y_var="test",
    center_name="Mediana",
    x_label="Porcentaje de Rechazo (%)",
    y_label="Prueba de Localización",
    x_ticks=[0, 10, 20, 30, 40],
    x_lim=(0, 40),
    band=(2.5, 7.5),
    band_color=OKABE_ITO["REDORANGE"],
    band_alpha=0.40,
)
save_figure(fig=fig, name="Grafico 3 Error Tipo I para las Distintas Pruebas")


fig = bar_plot(
    data=simulation_data,
    scenario=2,
    y_var="test",
    center_name="Mediana",
    x_label="Porcentaje de Rechazo (%)",
    y_label="Prueba de Localización",
    x_ticks=[0, 20, 40, 60, 80, 100],
    x_lim=(0, 100),
    lines=[80],
)
save_figure(fig=fig, name="Grafico 4 Potencia para las Distintas Pruebas")

## Tamaño muestral

fig = line_plot(
    data=simulation_data,
    scenario=3,
    x_var="sample_size",
    hue_var="test",
    center_name="Mediana",
    sample_size=[4, 8, 12, 16, 20, 24, 28, 32, 36, 40],
    x_label="Tamaño Muestral",
    y_label="Porcentaje de Rechazo (%)",
    y_ticks=[0, 5, 10, 15, 20],
    y_lim=(0, 25),
    test=["ANOVA", "Permutación", "Kruskal-Wallis"],
    palette={
        "ANOVA": OKABE_ITO["SKYBLUE"],
        "Permutación": OKABE_ITO["ORANGE"],
        "Kruskal-Wallis": OKABE_ITO["MAGENTA"],
    },
    band=(2.5, 7.5),
    band_color=OKABE_ITO["REDORANGE"],
    band_alpha=0.40,
)

save_figure(fig=fig, name="Grafico 5 Error Tipo I segun Tamano Muestral")


fig = line_plot(
    data=simulation_data,
    scenario=5,
    x_var="sample_size",
    hue_var="test",
    center_name="Mediana",
    sample_size=[4, 8, 12, 16, 20, 24, 28, 32, 36, 40],
    x_label="Tamaño Muestral",
    y_label="Porcentaje de Rechazo (%)",
    y_ticks=[0, 20, 40, 60, 80, 100],
    y_lim=(0, 100),
    lines=[80],
)

save_figure(fig=fig, name="Grafico 6 Potencia segun Tamano Muestral")


## Cantidad de grupos

fig = line_plot(
    data=simulation_data,
    scenario=4,
    x_var="group",
    hue_var="test",
    center_name="Mediana",
    group=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    x_label="Cantidad de Grupos",
    y_label="Porcentaje de Rechazo (%)",
    y_ticks=[
        0,
        10,
        20,
        30,
    ],
    y_lim=(0, 35),
    x_ticks=[0, 4, 8, 12, 16, 20],
    band=(2.5, 7.5),
    band_color=OKABE_ITO["REDORANGE"],
    band_alpha=0.40,
    test=["ANOVA", "Permutación", "Kruskal-Wallis"],
    palette={
        "ANOVA": OKABE_ITO["SKYBLUE"],
        "Permutación": OKABE_ITO["ORANGE"],
        "Kruskal-Wallis": OKABE_ITO["MAGENTA"],
    },
)

save_figure(fig=fig, name="Grafico 7 Error Tipo I segun Cantidad de Grupos")


fig = line_plot(
    data=simulation_data,
    scenario=6,
    x_var="group",
    hue_var="test",
    center_name="Mediana",
    group=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    x_label="Cantidad de Grupos",
    y_label="Porcentaje de Rechazo (%)",
    y_ticks=[0, 20, 40, 60, 80, 100],
    y_lim=(0, 100),
    lines=[80],
)

save_figure(fig=fig, name="Grafico 8 Potencia segun Cantidad de Grupos")


## Tamaño del efecto


fig = line_plot(
    data=simulation_data,
    scenario=[7, 8, 9, 10, 11, 12, 13],
    x_var="Cociente SD",
    hue_var="test",
    center_name="Mediana",
    group=[4],
    x_label="Desvío Estándar",
    y_label="Porcentaje de Rechazo (%)",
    x_ticks=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
    y_ticks=[0, 20, 40, 60, 80, 100],
    y_lim=(0, 100),
    lines=[80],
    test=[
        "ANOVA",
        "Permutación",
    ],
    palette={
        "ANOVA": OKABE_ITO["SKYBLUE"],
        "Permutación": OKABE_ITO["ORANGE"],
    },
)

save_figure(fig=fig, name="Grafico 9 Potencia segun Tamano del Efecto")


## Diferentes patrones de desvíos

fig = bar_plot(
    data=simulation_data,
    scenario=[19, 20, 21],
    y_var="Patrón SD",
    center_name="Mediana",
    hue="test",
    group=[4, 8],
    x_label="Porcentaje de Rechazo (%)",
    y_label="Patrón de Desvíos Estándar",
    x_ticks=[0, 20, 40, 60, 80, 100],
    x_lim=(0, 100),
    lines=[80],
    palette={
        "ANOVA": OKABE_ITO["SKYBLUE"],
        "Permutación": OKABE_ITO["ORANGE"],
    },
    test=["ANOVA", "Permutación"],
    legend=True,
)
save_figure(fig=fig, name="Grafico 10 Potencia para los distintos patrones de desvio estandar")


## Outliers

fig = line_plot(
    data=simulation_data,
    scenario=[3, 14, 15, 16, 17, 18],
    x_var="Contaminación",
    hue_var="test",
    center_name="Mediana",
    group=4,
    test=["ANOVA", "Permutación"],
    dist_name=["Normal"],
    x_label="Porcentaje de Contaminación (%)",
    y_label="Porcentaje de Rechazo (%)",
    y_ticks=[0, 10, 20, 30, 40, 50, 60],
    y_lim=(0, 60),
    band=(2.5, 7.5),
    band_color=OKABE_ITO["REDORANGE"],
    band_alpha=0.40,
    figure_size_cm=(22, 15),
)

save_figure(fig=fig, name="Grafico 11 Error Tipo I segun Cantidad de Valores Atípicos")


print("# " + "-" * 96 + " #")
print("# " + "DONE".center(96) + " #")
print("# " + "-" * 96 + " #")
