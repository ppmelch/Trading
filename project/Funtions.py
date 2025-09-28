def librerias():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt    

def primera_funcion():
    return print("Hola, esta es mi primera funcion")


def segunda_funcion():
    np.random.seed(0)
    data = np.random.randn(1000)
    plt.hist(data, bins=30, alpha=0.7, color='blue')
    plt.title('Histograma de datos aleatorios')
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia')
    plt.show()