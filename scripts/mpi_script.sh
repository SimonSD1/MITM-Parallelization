# Simon Sepiol-Duchemin, Joshua Setia

#!/bin/bash

# Script modifié pour exécuter 5 fois par taille d'entrée n avec les paramètres et le nombre de processus adaptés.

# Fichiers pour les logs
output_file="output_execution.txt"
error_file="error_log.txt"

# Nettoyer les fichiers de sortie
> $output_file
> $error_file

echo "Contenu de \$OAR_NODEFILE:" >> $output_file
cat $OAR_NODEFILE >> $output_file

# Tableau des paramètres
declare -A params
params[10]="8bbf923330a45c86 29e0694a8bd0564d"
params[11]="ad9fad871dd77ea0 405c3ecfd84712d0"
params[12]="7e13a50f2cc4dfc2 e9f8aff25657c833"
params[13]="1ae5fbddba8de71b bc2009383b71775f"
params[14]="04a58a4838c8994e 88dec3e5408b5707"
params[15]="dcab0655e634c921 6c676feb94f1ac1a"
params[16]="ff9a939856e72e42 6dfe59d87f6b919b"
params[17]="573d0052736e96bf 743b37bc4d3d578b"
params[18]="6a1f6fcbbe5dc0f8 5d340d467818077b"
params[19]="9e5f746317155de1 bd913488bc34431a"
params[20]="855ac5f0e32f2ea3 27eba3c43117d51e"
params[21]="1a184a33c679e01d 87514ccf63cce937"
params[22]="07aaec1070da733e 8729c30c554299ec"
params[23]="3b9837c29d074c05 33432685471db252"
params[24]="0252278e867abfc6 723e1694de23d130"
params[25]="7dd4f3795586232d 6132c0fec4186daa"
params[26]="b723e1f723b217da 2181bf90e29f4ac0"
params[27]="1c56d4064e2c35f7 c0e9ca9ef4db6e8b"
params[28]="53b97d360fe75e34 96034c6a30d631e4"
params[29]="73908bf3a3b21408 1256efda7c5cf0ab"
params[30]="797b45fb4d863eca d50beaf32a0f27eb"
params[31]="846d93ea512054fd b7dfe7c5a87a3c4d"
params[32]="a2adde7c356aacd1 ca78bbea95f9ee60"
params[33]="21ec533ce0b1b190 7c7d8c6c188afc89"
params[34]="d9e2639ca72317a2 acf4ab49e6dfa20e"
params[35]="a3b39a92d4394ee3 037b371593f41eee"
params[36]="863c39ab9efd0204 ee6aa16cc8bdfc75"

# Tableau des nombres de processus
declare -A processes
processes[10]=2
processes[11]=5
processes[12]=5
processes[13]=5
processes[14]=5
processes[15]=10
processes[16]=10
processes[17]=10
processes[18]=10
processes[19]=10
processes[20]=20
processes[21]=20
processes[22]=20
processes[23]=20
processes[24]=20
processes[25]=20
processes[26]=20
processes[27]=20
processes[28]=20
processes[29]=20
processes[30]=90
processes[31]=110
processes[32]=120
processes[33]=130
processes[34]=144
processes[35]=144
processes[36]=144

echo "Starting test runs..." >> $output_file

# Fonction pour exécuter 5 fois pour une taille donnée
function run_mpi() {
    local n=$1
    local p=${processes[$n]}
    local c0_c1=${params[$n]}
    local run_index=1

    echo "Running mpiexec for --n $n with $p processes and parameters: $c0_c1" >> $output_file
    echo "Number of processes: $p" >> $output_file

    while [ $run_index -le 5 ]; do
        echo "Execution #$run_index for --n $n with $p processes..." >> $output_file
        mpiexec --n $p --mca pml ^ucx --hostfile $OAR_NODEFILE ./dic --n $n --C0 ${c0_c1%% *} --C1 ${c0_c1##* } >> $output_file 2>> $error_file

        if [ $? -eq 0 ]; then
            echo "Run #$run_index for --n $n completed successfully." >> $output_file
            run_index=$((run_index + 1))
        else
            echo "Error occurred during run #$run_index for --n $n. Retrying..." >> $output_file
            echo "Detailed error:" >> $error_file
            sleep 2  # Pause avant de réessayer
        fi
    done

    echo "Completed 5 successful runs for --n $n" >> $output_file
}

# Lancer les tests pour chaque taille n
for n in {10..36}; do
    if [[ -v params[$n] && -v processes[$n] ]]; then
        run_mpi $n
        echo "Completed runs for --n $n." >> $output_file
    else
        echo "Parameters or processes not defined for --n $n. Skipping." >> $output_file
    fi
done

echo "All test runs completed." >> $output_file
