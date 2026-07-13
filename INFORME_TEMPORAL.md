#
Usamos n=20 y 4 grupos

## Medidas de centralidad

![grafico 1](<figures/Grafico 1 Error Tipo I para las Distintas Medidas de Centralidad.png>)


 En disribuciones simétricas todas las medidas controlaron el error tipo I. En distribuciones asimétricas, solo algunas pruebas en combinación con la media intercuartil, media de tukey y mediana controlaron el error tipo I.
 **falta chequear para n=10, 40 y k_grupos=8 y distribuciones t5 y lognormal**

 ![grafico 2](<figures/Grafico 2 Potencia para las Distintas Medidas de Centralidad.png>)

  En distribuciones simétricas todas las medidas alcanzaron potencias similares. En distribuciones asimétricas, mediana, media intercuartil y media de tukey presetaron potencias parecidas.
**falta chequear para n=10, 40 y k_grupos=8 y distribuciones t5 y lognormal**

CONCLUSIÓN: Para distribuciones simétricas todas las medidas presentan el mismo rendimiento (control del error tipo I y potencia). Para distribuciones asimétricas solo las medidas más robustas (mediana, media intercuartil, media de tukey) controlan el error tipo I, aunque las tres presentan potencias parecidas. Se decide utilizar la mediana debido a que es la medida más usada en la literatura.

## Prueba 

![grafico 3](<figures/Grafico 3 Error Tipo I para las Distintas Pruebas.png>)

En distribuciones simétricas, salvo el ANOVA winsorizado, todas las pruebas controlan el error tipo I.

En distribuciones asimétricas, solo las pruebas ANOVA, ANOVA por Permutación, ANOVA con raiz cuadrada y ANOVA de Welch logran controlar el Error Tipo I.
**falta chequear para n=10, 40 y k_grupos=8 y distribuciones t5 y lognormal**

![grafico 4](<figures/Grafico 4 Potencia para las Distintas Pruebas.png>)

En distribuciones simétricas, ANOVA, ANOVA por Permutación alcanzan las potencias más alta, seguidos del ANOVA dela raiz cuadrada y por último el ANOVA de Welch.

En distribuciones asimétricas, ANOVA, ANOVA por Permutación y ANOVA dela raiz cuadrada alcanzan las potencias más altas.
**falta chequear para n=10, 40 y k_grupos=8 y distribuciones t5 y lognormal**

## Tamaño muestral

![grafico 5](<figures/Grafico 5 Error Tipo I segun Tamano Muestral.png>)

Para tamaño 4 todas las pruebas presentan error tipo I inflado. Para tamaño 8 y superiror se estabilizan.
Para distribuciones simétricas, todas las pruebas controlan el error tipo I
Para distribuciones asimétricas , el ANOVA de la Raiz Cuadrada y el ANOVA de Welch se encuentran en el borde superior de lo concideradao "error tipo I bajo control" (borde de las franjas).


![grafico 6](<figures/Grafico 6 Potencia segun Tamano Muestral.png>)

Observamos como las pruebas ANOVA y ANOVA por permutación presentan consistentement una potencia mayor a la del ANOVA de Welch.
Para la distribuciones simétricas alcanzan potencias aceptables para tamaño 20, mientras que para asimétricas requieren de tamaño mucho mayor40.
(aclarar que distribuciones).

## Cantidad de grupos

![grafico 7](<figures/Grafico 7 Error Tipo I segun Cantidad de Grupos.png>)

Para distribuciones simétricas, se controla el error tipo I en todos los casos.
Para distribuciones asimétricas solo ANOVA y ANOVA por permutación controlan el error tipo I para 6 más grupos

![grafico 8](<figures/Grafico 7 Poteencia segun Cantidad de Grupos.png>)


Para el ANOVA y ANOVA por permutación Se observa que la potencia es menor para 2 grupos, se estabiliza para 4 a 12 grupos, y disminuye para 14 a 20 grupos

Para el ANOVA de Welch, en distribuciones simétricas la potencia dismnuye progresivamente, para distribuciones asimétricas parece disminuir inicialmente (para k=4) y luego incrementa lentamente. EN todos los casos (salvo k=2) es menor que la potencia de los otros dos tests.

## Tamaño del Efecto

## Distribución de los desvíos

## Outliers

