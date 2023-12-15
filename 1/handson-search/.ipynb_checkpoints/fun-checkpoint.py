import matplotlib.pyplot as plt
import numpy as np

# Definisci la funzione che desideri visualizzare (ad esempio, una funzione quadratica)
def funzione_quadratica(x):
    return x ** 2

# Genera un insieme di punti x
x = np.linspace(-10, 10, 100)  # Intervallo da -10 a 10 con 100 punti

# Calcola i corrispondenti valori y utilizzando la funzione
y = funzione_quadratica(x)

# Crea il grafico
plt.plot(x, y, label='y = x^2')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Grafico di y = x^2')
plt.grid(True)

# Aggiungi una legenda
plt.legend()

# Mostra il grafico
plt.show()
