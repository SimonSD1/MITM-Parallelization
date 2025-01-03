# Simon Sepiol-Duchemin, Joshua Setia

# Nom de l'exécutable
EXEC = dic

# Nom du fichier source
SRC = dictionary.c

# Options générales
CFLAGS = -Wall -O3
CC = mpicc

# Compilation par défaut (sans OpenMP)
all: no_openmp

# Compilation sans OpenMP
no_openmp: $(SRC)
	$(CC) $(CFLAGS) -o $(EXEC) $(SRC)

# Compilation avec OpenMP
openmp:
	$(CC) $(CFLAGS) -fopenmp -o $(EXEC) $(SRC)

# Force la recompilation pour openmp
.PHONY: openmp

# Nettoyage des fichiers générés
clean:
	rm -f $(EXEC)
