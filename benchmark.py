# Simon Sepiol-Duchemin, Joshua Setia

import matplotlib.pyplot as plt
import numpy as np

nbProc14 = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
          7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20, 
          30, 30, 30, 30, 30, 40, 40, 40, 40, 40, 50, 50, 50, 50, 50, 60, 60, 60, 60, 60, 70, 70, 
          70, 70, 70, 80, 80, 80, 80, 80, 90, 90, 90, 90, 90, 100, 100, 100, 100, 100, 110, 110, 
          110, 110, 110, 120, 120, 120, 120, 120, 130, 130, 130, 130, 130, 140, 140, 140, 140, 140, 
          150, 150, 150, 150, 150])

nbProc30 = np.array([40,40,40,40,40,
                        50,50,50,50,50,
                        60,60,60,60,60,
                        70,70,70,70,70,
                        80,80,80,80,80,
                        90,90,90,90,90,
                        100,100,100,100,100,
                        110,110,110,110,110,
                        120,120,120,120,120,
                        130,130,130,130,130,
                        140,140,140,140,140,
                        150,150,150,150,150,
                        160,160,160,160,160])

tempsSize14 = np.array([
    # p=1 (5 runs)
    4855, 4847, 4833, 4830, 4852,
    # p=2 (5 runs)
    2431, 2489, 2455, 2459, 2523,
    # p=3 (5 runs)
    2041, 1955, 1938, 1938, 2395,
    # p=4 (5 runs)
    1547, 2145, 1944, 1940, 1517,
    # p=5 (5 runs)
    1150, 1092, 1666, 1181, 1509,
    # p=6 (5 runs)
    1039, 1044, 1317, 1282, 1299,
    # p=7 (5 runs)
    1323, 1213, 1205, 1145, 1014,
    # p=8 (5 runs)
    1066, 1169, 1168, 1144, 904,
    # p=9 (5 runs)
    1033, 845, 1043, 1070, 847,
    # p=10 (5 runs)
    1112, 1055, 1068, 1053, 1135,
    # p=20 (5 runs)
    51440, 42187, 51520, 51215, 51445,
    # p=30 (5 runs)
    31650, 33930, 33541, 33112, 33388,
    # p=40 (5 runs)
    32904, 33128, 33345, 32915, 33328,
    # p=50 (5 runs)
    32380, 32766, 32952, 32986, 33903,
    # p=60 (5 runs)
    32939, 32808, 41579, 32319, 32758,
    # p=70 (5 runs)
    32428, 42228, 32498, 34092, 32296,
    # p=80 (5 runs)
    32945, 41906, 41785, 41771, 33497,
    # p=90 (5 runs)
    42377, 33535, 34405, 42042, 43055,
    # p=100 (5 runs)
    42487, 33597, 34395, 42901, 42727,
    # p=110 (5 runs)
    42999, 44278, 44590, 43904, 43629,
    # p=120 (5 runs)
    43478, 43520, 43380, 43029, 43687,
    # p=130 (5 runs)
    34689, 43476, 43179, 43281, 43312,
    # p=140 (5 runs)
    43292, 43836, 44802, 44364, 44118,
    # p=150 (5 runs)
    44353, 43978, 43448, 35378, 43479
])


memoireSize14 = np.array([
    # p=1 (5 runs)
    23268, 20956, 21048, 21008, 21160,
    # p=2 (5 runs)
    21092, 21256, 21064, 20960, 20984,
    # p=3 (5 runs)
    21076, 21132, 23164, 23080, 21100,
    # p=4 (5 runs)
    20888, 21168, 21204, 21060, 21100,
    # p=5 (5 runs)
    20992, 20872, 21124, 21084, 20996,
    # p=6 (5 runs)
    21116, 21020, 21284, 23104, 21016,
    # p=7 (5 runs)
    21068, 21036, 21108, 21096, 21128,
    # p=8 (5 runs)
    21108, 23324, 21196, 20976, 23216,
    # p=9 (5 runs)
    21184, 21204, 21152, 21056, 21132,
    # p=10 (5 runs)
    23372, 21296, 21096, 21148, 23120,
    # p=20 (5 runs)
    21544, 21736, 21652, 21624, 21564,
    # p=30 (5 runs)
    21820, 23392, 21872, 21712, 21880,
    # p=40 (5 runs)
    21896, 21816, 23412, 21940, 21968,
    # p=50 (5 runs)
    24376, 24356, 24328, 24344, 24360,
    # p=60 (5 runs)
    26636, 24444, 23520, 22600, 22564,
    # p=70 (5 runs)
    24660, 22516, 24680, 24444, 24444,
    # p=80 (5 runs)
    24572, 24704, 24720, 24724, 22568,
    # p=90 (5 runs)
    22948, 22760, 22728, 24832, 22560,
    # p=100 (5 runs)
    23020, 22852, 22852, 24640, 24768,
    # p=110 (5 runs)
    24748, 24936, 22920, 22848, 24772,
    # p=120 (5 runs)
    26908, 24992, 24744, 24704, 26796,
    # p=130 (5 runs)
    22756, 24836, 24996, 26848, 27020,
    # p=140 (5 runs)
    27084, 24912, 24932, 22756, 25052,
    # p=150 (5 runs)
    25064, 24996, 24112, 23024, 24904
])


tempsSize14OpenMP = np.array([
    # p=1
    9486, 9602, 9909, 9390, 7493,
    # p=2
    2259, 4233, 2323, 4204, 4923,
    # p=3
    1745, 2436, 1741, 1797, 1797,
    # p=4
    1658, 10041, 1530, 1293, 1552,
    # p=5
    2293, 1087, 1485, 1583, 1147,
    # p=6
    1235, 1322, 1036, 7491, 1411,
    # p=7
    1108, 8825, 2998, 2067, 8631,
    # p=8
    2513, 2374, 8121, 1367, 1210,
    # p=9
    1878, 17756, 1104, 7462, 1372,
    # p=10
    8215, 7714, 1333, 10676, 19527,
    # p=20
    58153, 58155, 58039, 58184, 57568,
    # p=30
    38209, 40472, 39176, 39627, 38117,
    # p=40
    39035, 39329, 38720, 39305, 39145,
    # p=50
    48508, 48090, 49102, 51942, 52306,
    # p=60
    49811, 48817, 48878, 38892, 48346,
    # p=70
    39168, 54295, 47476, 48829, 60802,
    # p=80
    48553, 59186, 58437, 60726, 58831,
    # p=90
    42066, 59887, 50967, 60611, 59399,
    # p=100
    44462, 52730, 60802, 63012, 44267,
    # p=110
    51194, 57011, 52424, 58017, 61688,
    # p=120
    53406, 53134, 60769, 53534, 61914,
    # p=130
    54155, 53413, 61587, 53892, 56438,
    # p=140
    62134, 54043, 60690, 59593, 50936,
    # p=150
    61490, 61722, 62089, 49719, 61009
])

memoireSize14OpenMP = np.array([
    # p=1
    21032, 23172, 23180, 21024, 20960,
    # p=2
    23024, 20956, 21040, 21088, 21088,
    # p=3
    22924, 21012, 20996, 21060, 21236,
    # p=4
    21288, 20956, 21284, 23132, 21240,
    # p=5
    21164, 23240, 23080, 20924, 21060,
    # p=6
    21048, 20980, 20776, 21052, 23120,
    # p=7
    23140, 21120, 21076, 21128, 20896,
    # p=8
    20972, 21244, 21064, 21204, 23008,
    # p=9
    21004, 21048, 21068, 21192, 21128,
    # p=10
    21376, 21192, 21196, 21256, 21212,
    # p=20
    21592, 21792, 23468, 21540, 21620,
    # p=30
    21896, 21812, 21780, 21812, 21948,
    # p=40
    21940, 21868, 22012, 21856, 21808,
    # p=50
    22304, 24500, 22344, 22288, 22444,
    # p=60
    24580, 22468, 26480, 22440, 24556,
    # p=70
    26504, 23744, 22628, 24488, 24684,
    # p=80
    23680, 22620, 22388, 24732, 24604,
    # p=90
    24780, 24612, 22592, 24652, 26812,
    # p=100
    22592, 24768, 24796, 24712, 26804,
    # p=110
    24632, 22668, 22648, 22904, 22820,
    # p=120
    22860, 26956, 24948, 22920, 24880,
    # p=130
    22832, 22736, 24768, 22820, 23192,
    # p=140
    22856, 23012, 25008, 25012, 25008,
    # p=150
    25068, 24992, 24924, 24844, 25048
])


tempsSize30 = np.array([
    # p = 40 (5 runs)
    50879379, 50483715, 50107011, 49299245, 50370617,
    # p = 50 (5 runs)
    25274896, 25223592, 25254130, 25204874, 25240297,
    # p = 60 (5 runs)
    22126392, 22069028, 22155375, 22445368, 22188707,
    # p = 70 (5 runs)
    20417338, 19629696, 20050987, 19653358, 19643251,
    # p = 80 (5 runs)
    18245191, 18960872, 18456765, 17820463, 18808785,
    # p = 90 (5 runs)
    15902005, 16415420, 16302492, 16149325, 16590817,
    # p = 100 (5 runs)
    16953066, 17234771, 16514710, 17516946, 17181856,
    # p = 110 (5 runs)
    16580054, 16063570, 15669144, 16053235, 16418752,
    # p = 120 (5 runs)
    19726769, 18547964, 19357868, 21384149, 21004358,
    # p = 130 (5 runs)
    20297400, 18808456, 21020733, 20746172, 21240839,
    # p = 140 (5 runs)
    36436326, 32748228, 33326585, 36682799, 34372317,
    # p = 150 (5 runs)
    30134105, 30088189, 28970777, 32172089, 31829852,
    # p = 160 (5 runs)
    36459066, 36952989, 37071550, 35828065, 34119649
])


memoireSize30 = np.array([
    # p = 40 (5 runs)
    384772, 385044, 384892, 385036, 384920,
    # p = 50 (5 runs)
    314288, 314216, 314556, 314288, 314720,
    # p = 60 (5 runs)
    269048, 270300, 273572, 273880, 273728,
    # p = 70 (5 runs)
    234252, 234448, 234320, 234192, 234336,
    # p = 80 (5 runs)
    208220, 207768, 208120, 208004, 208196,
    # p = 90 (5 runs)
    191576, 191584, 191552, 191976, 191696,
    # p = 100 (5 runs)
    173432, 173356, 173524, 173864, 173632,
    # p = 110 (5 runs)
    162032, 162204, 162796, 161716, 160724,
    # p = 120 (5 runs)
    149180, 149032, 149144, 151072, 151116,
    # p = 130 (5 runs)
    141336, 145068, 141600, 141584, 140164,
    # p = 140 (5 runs)
    138712, 130316, 129548, 130080, 130400,
    # p = 150 (5 runs)
    123012, 132248, 130776, 127260, 123176,
    # p = 160 (5 runs)
    117492, 117732, 128936, 131888, 127544
])


# Fonction pour calculer la moyenne
def calculate_mean(data, processes):
    unique_processes = np.unique(processes)
    means = [data[processes == p].mean() for p in unique_processes]
    return np.array(means)

# Calcul des accélérations
def calculate_speedup(serial_times, parallel_times):
    return serial_times / parallel_times

# Calcul des moyennes
mean_temps14 = calculate_mean(tempsSize14, nbProc14)
mean_temps14OpenMP = calculate_mean(tempsSize14OpenMP, nbProc14)
mean_memoire14 = calculate_mean(memoireSize14, nbProc14)
mean_memoire14OpenMP = calculate_mean(memoireSize14OpenMP, nbProc14)

mean_temps30 = calculate_mean(tempsSize30, nbProc30)
mean_memoire30 = calculate_mean(memoireSize30, nbProc30)

# Accélérations
speedup14 = calculate_speedup(mean_temps14[0], mean_temps14)
speedup14OpenMP = calculate_speedup(mean_temps14[0], mean_temps14OpenMP)
speedup30 = calculate_speedup(mean_temps30[0], mean_temps30)

# Graphiques
def plot_results():
    plt.figure(figsize=(12, 6))
    plt.plot(np.unique(nbProc14), mean_temps14, label='Temps Size 14', marker='o')
    plt.plot(np.unique(nbProc14), mean_temps14OpenMP, label='Temps Size 14 OpenMP', marker='o')
    plt.xlabel('Nombre de Processus')
    plt.ylabel('Temps d\'exécution (ms)')
    plt.title('Comparaison des temps d\'exécution (Input size 14)')
    plt.legend()
    plt.grid()

    plt.figure(figsize=(12, 6))
    plt.plot(np.unique(nbProc14), mean_memoire14, label='Mémoire Size 14', marker='o')
    plt.plot(np.unique(nbProc14), mean_memoire14OpenMP, label='Mémoire Size 14 OpenMP', marker='o')
    plt.xlabel('Nombre de Processus')
    plt.ylabel('Mémoire (Ko)')
    plt.title('Comparaison de la mémoire (Input size 14)')
    plt.legend()
    plt.grid()

    plt.figure(figsize=(12, 6))
    plt.plot(np.unique(nbProc14), speedup14, label='Speedup Size 14', marker='o')
    plt.plot(np.unique(nbProc14), speedup14OpenMP, label='Speedup Size 14 OpenMP', marker='o')
    plt.xlabel('Nombre de Processus')
    plt.ylabel('Accélération')
    plt.title('Accélération (Input size 14)')
    plt.legend()
    plt.grid()

    plt.figure(figsize=(12, 6))
    plt.plot(np.unique(nbProc30), mean_temps30, label='Temps Size 30', marker='o')
    plt.xlabel('Nombre de Processus')
    plt.ylabel('Temps d\'exécution (ms)')
    plt.title('Temps d\'exécution (Input size 30)')
    plt.legend()
    plt.grid()

    plt.figure(figsize=(12, 6))
    plt.plot(np.unique(nbProc30), mean_memoire30, label='Mémoire Size 30', marker='o')
    plt.xlabel('Nombre de Processus')
    plt.ylabel('Mémoire (Ko)')
    plt.title('Mémoire (Input size 30)')
    plt.legend()
    plt.grid()

    plt.figure(figsize=(12, 6))
    plt.plot(np.unique(nbProc30), speedup30, label='Speedup Size 30', marker='o')
    plt.xlabel('Nombre de Processus')
    plt.ylabel('Accélération')
    plt.title('Accélération (Input size 30)')
    plt.legend()
    plt.grid()

    plt.show()

# Exécution des graphes
plot_results()
