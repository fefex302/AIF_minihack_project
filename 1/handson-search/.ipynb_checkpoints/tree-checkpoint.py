def stampa_albero(n):
    for i in range(1, n + 1):
        spazi = " " * (n - i)
        asterischi = "*" * (2 * i - 1)
        print(spazi + asterischi)

n = int(input("Inserisci l'altezza dell'albero di Natale: "))
stampa_albero(n)
