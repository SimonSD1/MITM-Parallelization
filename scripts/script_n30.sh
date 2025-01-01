# Simon Sepiol-Duchemin, Joshua Setia
#!/bin/bash

# Output and error files to log results
output_file="output_n30.txt"
error_file="error_log.txt"

# Clear the output and error files
> $output_file
> $error_file

echo "Contenu de \$OAR_NODEFILE:" >> $output_file
cat $OAR_NODEFILE >> $output_file

# Function to run mpiexec with a specified number of processes
echo "Starting test runs..." >> $output_file

function run_mpi() {
    local n=$1
    local run_index=1
    local max_attempts=10  # Stop retries after this many failed attempts per run
    local attempt=0

    echo "Running mpiexec with --n $n" >> $output_file

    while [ $run_index -le 5 ]; do
        echo "Attempt #$((attempt + 1)) for run #$run_index with $n processes:" >> $output_file
        mpiexec --n $n --mca pml ^ucx --hostfile $OAR_NODEFILE ./dic --n 30 --C0 797b45fb4d863eca --C1 d50beaf32a0f27eb >> $output_file 2>> $error_file

        if [ $? -eq 0 ]; then
            echo "Run #$run_index for $n processes completed successfully." >> $output_file
            run_index=$((run_index + 1))
            attempt=0  # Reset attempt counter for the next run
        else
            echo "Error occurred during run #$run_index for $n processes. Retrying..." >> $output_file
            echo "Detailed error:" >> $error_file
            attempt=$((attempt + 1))
            sleep 2  # Pause before retrying

            # Abort retries if max attempts are reached
            if [ $attempt -ge $max_attempts ]; then
                echo "Max attempts reached for run #$run_index with $n processes. Skipping." >> $output_file
                break
            fi
        fi
    done

    echo "Completed $((run_index - 1)) successful runs for --n $n" >> $output_file
}


# Run for processes from 40 to 160 in steps of 10
for n in $(seq 40 10 160); do
    run_mpi $n
    echo "Completed runs for $n processes." >> $output_file
done

echo "All test runs completed." >> $output_file
