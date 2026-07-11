# ESTUDIO SOBRE ALGUNAS VARIANTES DE LA PRUEBA DE LEVENE


## Etapa 1: Medidas de centralidad

Se comparan las medidas de centralidad:
Media, Media Modificada, Media Intercuartilica, Media de Tukey (c=6) y Mediana.
Opcional:
Se pueden utilizar distintos hiperparámetros para la media recortada (10%, 20%, 30%, 40%) y para la Media de Tukey( c=2, 3, 4, 5, 6)

## Etapa 2: Prueba

Se comparan las disintas pruebas:
ANOVA, ANOVA por permutación, Kruskar Wallis.

Opcional (listas para añadir):
- Van der Waerden.
- ANOVA de Welch.
- Mood (Prueba de la Mediana).
- Lepage. 
- Cucconi. 
- Modelos Lineales Generalizados (usando distribución gamma).
 
Opcional (aún sin programar):
- ANOVA de Yuen
- Prueba de Rachas
- Kolgomorov Smirnov

## Etapa 3: Tamaño muestral

Se estudia la potencia y error tipo I para:
n = 4, 8, 12, 16, 20, 24, 28, 32, 36, 40

## Etapa 4: Cantidad de grupos

Se estudia la potencia y error tipo I para:
k = 2, 4, 6, 8, 10, 12, 14, 16, 18, 20

## Etapa 5: Tamaño del efecto

Se estudia la potencia para:
sigma = 1.25, 1.5, 2 , 2.5, 3, 4, 5

## Etapa 6: Desvíos Estandar desiguales

Se estudia la potencia para los siguientes casos:

- 4 grupos:
| Metodo | sigma_1 | sigma_2  | sigma_3 | sigma_4 | 
| ---- | --- | ---  | --- | --- | 
| one | 1 | 1  | 1 | 2 | 
| half | 1 | 1  | 2 | 2 | 
| ladder | 1 | 1.33  | 1.67 | 2 | 

- 8 grupos:
| Metodo | sigma_1 | sigma_2  | sigma_3 | sigma_4 | sigma_5 | sigma_6  | sigma_7 | sigma_8 | 
| ---- | --- | ---  | --- | --- | --- | ---  | --- | --- | 
| one | 1 | 1  | 1 | 1 | 1 | 1  | 1 | 2 | 
| half | 1 | 1  | 1 | 1 |  2 | 2  | 2 | 2 | 
| ladder | 1 | 1.14  | 1.29 | 1.43 |  1.57 | 1.71  | 1.86 | 2 |

## Etapa 7: Valores atípicos
Se estudia el error tipo I en muestras donde uno o más observaciones provienen de una población de colas pesadas.
