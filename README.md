# E-I-Balance
Codebase for the work Biologically-Informed Excitatory and Inhibitory Balance for Robust Spiking Neural Network Training

# Simulation workflow
Due to the massive workload of processing hundreds of training sessions (and then multiple layers of processing) this workflow is designed to enable massive batch computing on a SLURM system (which was used here at GWU).

The overview is the following:  
1. Generate the shell scripts with each script corresponding to a single SLURM job which may have multiple training sessions (`shellScripts/generateShellScripts.py`)  
2. Submit these scripts with a correspondingly generated `runAll.sh` script which just submits all of these jobs with a single run  
3. Once these trainings have finished, follow similar steps for generating reports (`shellScripts/generateShellScripts_Reports.py`) and submitting the jobs to generate this.  
4. We can aggregate the results into combined csv files using (`scripts/generateCSV_SR.py`)
5. These final csvs are what the final jupyter notebooks use for generating the corresponding figures


# For additional information
Running of each portion of the process is parallelized by making all configuration settings accessible through command line args. For additional information check the `--help` flag with the fundamental scripts of this repository to see what parameters are accessible.  

Key scripts to check:
- `scripts/excInhTraining.py`
- `scripts/generateReport.py`

# Available data
Included here are the final csv files used in the larger/key figures of the paper. Additional data (in a raw format) is available upon request due to the large nature of the data underneath (which would drastically bloat the repository)

# Additional Support
Please file an issue as needed for additional support, questions, or suggestions for code reorganization
