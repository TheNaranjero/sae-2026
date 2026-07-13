# LISTA DE TAREAS

- Añadir Cantidad de grupos y Desvíos distintos a **main.py**. [HECHO]
- Añadir más pruebas a **main.py**. [HECHO]
- Ajustar los tamaños muestrales, cantidad de puebas, pruebas, etc según la esturctura del trabajo. [HECHO]
- Preparar gráficos en R (o el ggplot de python). [LISTOS LOS PRIMEROS 8 GRAFICOS]
- Armar informe de ejemplo en markdown. [ACTUALIZADO A LOS PRIMEROS 8 GRAFICOS]
- Decidir la extensión de la presentación y la exclusión de secciones.
- Decidir si se añadirá "tamaños muestrales diferentes" como etapa.



# ESTRUCTURA DEL TRABAJO

- Mostrar para n=20, 4 grupos, distribución normal y exponencial.
- Mencionar que se chequeó para n=10,20,40 , 4 y 8 grupos, distribución t y lognormal.

## MEDIDAS DE CENTRALIDAD

- Gráfico de puntos con el error tipo I, eje X=medidas de centralidad. 
- Idem potencia.
- Total 4 gráficos (error tipo I y Potencia, para normal y exponencial )
- Conclusión: *En disribuciones simétricas todas las medidas controlaron el error tipo I y obtuvieron potencias similares. En distribuciones asimétricas, solo la media intercuartil, media de tukey y mediana controlaron el error tipo I alcanzando potencias similares. Se procede a usar la mediana en el resto del trabajo por ser la usada tradicionalmente*
 
## PRUEBAS

- Gráfico de barras con el error tipo I, eje Y=Pruebas. 
- Idem potencia (filtrando solo las pruebas que controlaron el error tipo I ?).
- Total 4 gráficos (error tipo I y Potencia, para normal y exponencial )
- Conclusión: *En disribuciones simétricas todas las pruebas controlaron el error tipo I. En distribuciones asimétricas, solo la ANOVA, ANOVA por permutación y ANOVA de Welch controlaron el error tipo I, poseyendo los dos primeros mayor potencia. Se procede a usar ANOVA y ANOVA por permutación en el resto del trabajo por ser la usada tradicionalmente*

## TAMAÑO MUESTRAL

- Gráfico de linea con el error tipo I, eje X=tamaño muestral, de 4 a 40. 
- Idem potencia.
- Total 4 gráficos (error tipo I y Potencia, para normal y exponencial )
- Conclusión: *Se requiere de un tamaño de al menos 8 para controlar el error tipo I, en la normal un tamaño de 20 alcanza una potencia satisfactioria, para la exponencial se requiere tamaño 40. Nota: en la t(gl=5) se necesita tamaño 24, y la lognormal tamaño=100.*

## CANTIDAD DE GRUPOS

- Gráfico de linea con el error tipo I, eje X=cantidad de grupos, de 2 a 20. 
- Idem potencia.
- Total 4 gráficos (error tipo I y Potencia, para normal y exponencial )
- Conclusión: *Para cualquier cantidad de grupos el error tipo I queda contrlado. En tanto a la potencia, se observa que la potencia es máxima para entre 4 y 10 grupos*.

## DISTRIBUCIÓN DE LOS DESVÍOS ESTANDAR

- Gráfico de barras con la potencia mostrando las 3 configuaraciones para 4 y 8 grupos.
- Total 2 gráficos (normal y exponencial)
- Conclusión: *La potencia no varía  al tener 1 grupo con mayor desvío o la mitad de ellos. Cuando todos tienen desvíos diferentes, la potencia dismnuye.*"*

## CURVAS DE POTENCIA

- Gráfico de la línea con la potencia, eje X= desvío estandar del grupo distinto, sigma = 1.25 a 5.
- Total 2 gráficos (normal y exponencial).
- Conclusión: *Para la normal puede detectar desvíos estandar de 2, por otro lado la exponencial de 4. Nota: para la t(gl=5) fue de 2.5 y para la lognormal de 5. Es decir, para tamaños medianos (n=20) y distribuciones asimétricas solo podemos detectar desvíos estandar muy distintos del resto. Para distribuciones simétricas es razonable detectar desvíos del doble del resto*
