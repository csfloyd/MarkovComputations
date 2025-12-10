import os
import shutil
import numpy as np

n_params = 3
run_script = "/project/svaikunt/csfloyd/MarkovComputation/Python/ICL/run_icl.py"
#output_base = "/project/svaikunt/csfloyd/MarkovComputation/DirsICL/rho_all_seed_n_nodes5_2/"
#output_base = "/project/svaikunt/csfloyd/MarkovComputation/DirsICL/rho_edge_seed_n_nodes5_nlm0p05/"
output_base = "/project/svaikunt/csfloyd/MarkovComputation/DirsICL/rho_all_seed_n_nodes5_nlm_L_6/"
output_base = "/project/svaikunt/csfloyd/MarkovComputation/DirsICL/n_nodes_N_seed/"
#output_base = "/project/svaikunt/csfloyd/MarkovComputation/DirsICL/n_nodes_seed_3_nobase/"

# Define the range of values for param1 and labels for param2
param1_values = np.arange(0.1, 1.1, 0.1)
param1_values = np.arange(0.2, 1.2, 0.2)
param1_values = np.arange(0.0, 1.1, 0.1)
param1_values = np.arange(0.0, 0.22, 0.02)
param1_values = [round(s,2) for s in param1_values]
#param1_values = [2,3]
#param2_values = [1,2,3,4,5]
param1_values = np.arange(1,16,1)
param2_values = np.arange(1,16,1)
param3_values = [1,2,3,4,5]



# SLURM job template
job_template = """#!/bin/bash
#SBATCH --job-name=computation
#SBATCH --output={output}/training_batch.out   # Redirect stdout to the output directory
#SBATCH --error={output}/training_batch.err    # Redirect stderr to the output directory
#SBATCH --time=32:00:00
#SBATCH --partition=caslake
##SBATCH --partition=svaikunt 
#SBATCH --account=pi-svaikunt
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=32000

# module load python3

python3 {run_script} --param1 {param1} --output {output}
"""

if n_params == 1:
    # Loop over different parameter values
    for param1 in param1_values:
        output = output_base + f"{param1}"  # Define output folder name

        # Remove existing directory if it exists, then recreate it
        if os.path.exists(output):
            shutil.rmtree(output)  # Delete existing directory and contents
        os.makedirs(output)  # Create a new empty directory

        print(f"Created directory: {output}")

        # Generate job script content
        job_script_content = job_template.format(param1=param1, output=output, run_script=run_script)

        # Define a unique job filename
        job_filename = os.path.join(output, f"job_{param1}.sh")

        # Write the job script to a file
        with open(job_filename, "w") as job_file:
            job_file.write(job_script_content)

        # Submit the job using sbatch
        os.system(f"sbatch {job_filename}")

        print(f"Submitted job with param1={param1} and output={output}")



# SLURM job template
job_template_2 = """#!/bin/bash
#SBATCH --job-name=computation
#SBATCH --output={output}/training_batch.out   # Redirect stdout to the output directory
#SBATCH --error={output}/training_batch.err    # Redirect stderr to the output directory
#SBATCH --time=32:00:00
#SBATCH --partition=caslake
##SBATCH --partition=svaikunt 
#SBATCH --account=pi-svaikunt
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=32000

# module load python3

python3 {run_script} --param1 {param1} --param2 {param2} --output {output}
"""

if n_params == 2:
    # Loop over different parameter values
    for param1 in param1_values:
        for param2 in param2_values:
            output = os.path.join(output_base, f"{param1}_{param2}")  # Unique output folder for each param1, param2 combination

            # Remove existing directory if it exists, then recreate it
            if os.path.exists(output):
                shutil.rmtree(output)  # Delete existing directory and contents
            os.makedirs(output)  # Create a new empty directory

            print(f"Created directory: {output}")

            # Generate job script content
            job_script_content = job_template_2.format(param1=param1, param2=param2, output=output, run_script=run_script)

            # Define a unique job filename inside the output directory
            job_filename = os.path.join(output, f"job_{param1}_{param2}.sh")

            # Write the job script to a file
            with open(job_filename, "w") as job_file:
                job_file.write(job_script_content)

            # Submit the job using sbatch
            os.system(f"sbatch {job_filename}")

            print(f"Submitted job with param1={param1}, param2={param2}, and output={output}")


# SLURM job template for 3 parameters
job_template_3 = """#!/bin/bash
#SBATCH --job-name=computation
#SBATCH --output={output}/training_batch.out   # Redirect stdout to the output directory
#SBATCH --error={output}/training_batch.err    # Redirect stderr to the output directory
#SBATCH --time=32:00:00
#SBATCH --partition=caslake
##SBATCH --partition=svaikunt 
#SBATCH --account=pi-svaikunt
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=32000

# module load python3

python3 {run_script} --param1 {param1} --param2 {param2} --param3 {param3} --output {output}
"""

if n_params == 3:
    # Loop over different parameter values
    for param1 in param1_values:
        for param2 in param2_values:
            for param3 in param3_values:
                output = os.path.join(output_base, f"{param1}_{param2}_{param3}")  # Unique output folder

                # Skip if results.pkl already exists
                results_file = os.path.join(output, "results.pkl")
                if os.path.exists(results_file):
                    print(f"Skipping {output} - results.pkl already exists")
                    continue

                # Remove existing directory if it exists, then recreate it
                if os.path.exists(output):
                    shutil.rmtree(output)  # Delete existing directory and contents
                os.makedirs(output)  # Create a new empty directory

                print(f"Created directory: {output}")

                # Generate job script content
                job_script_content = job_template_3.format(param1=param1, param2=param2, param3=param3, output=output, run_script=run_script)

                # Define a unique job filename inside the output directory
                job_filename = os.path.join(output, f"job_{param1}_{param2}_{param3}.sh")

                # Write the job script to a file
                with open(job_filename, "w") as job_file:
                    job_file.write(job_script_content)

                # Submit the job using sbatch
                os.system(f"sbatch {job_filename}")

                print(f"Submitted job with param1={param1}, param2={param2}, param3={param3}, and output={output}")

