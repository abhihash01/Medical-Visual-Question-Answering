# Model Running Instructions

## Setup Instructions
### Download the dataset
 ** Download the Slake and VQA RAD directories from their websites**
 Slake: https://www.med-vqa.com/slake/
 VQA RAD: https://osf.io/89kps/
 
 The slake dataset has been split into test and train. But VQA rad has just one file out of which on the basis of label, test and train needs to be seperated. These have already been done and placed on the cluster
 
 **Download the filtered datasets of slake and vqa rad, which have been split into test and train **
 
*** path on the cluster- /home/abhilash/user_root/IMAGE_Lab/git/git_2/medvqa/README.md ***

### clone the repository
1. **For running LLaVa Med 1.5 (MistraL)**
   - Navigate to the LLaVa Med 1.5 directory. 
   ```bash
   cd LLaVA-Med-1.5
   ```
   - If creating a new environment from scratch
   ```bash
   conda create -n llava-med python=3.10 -y
   conda activate llava-med
   pip install --upgrade pip  # enable PEP 660 support
   pip install -e .
   ```
   - The python file for running slake dataset is running_files/model_vqa_med_mistral.py and the file for running vqa rad dataset is running_files/model_vqa_med_mistral_vqa_rad.py
   - We run each of these with the job files running_files/job_model_vqa_mistral
   - For running LLaVA Med 1.5 on filtered Slake Dataset, edit the running_files/job_model_vqa_med_mistral.job. Add the newly created python envrironment paths by adding it in the source and activate the created environment. If running on the cluster, the already created file should work.
   - Then add the python path, path to the model_vqa_med_mistral.py file. --model-path has to be the one fixed in the file
   - Add the path of slake test file in -question-file, slake images directory in the -image-folder
   - Add the path you want the answer to be stored in the answers-file.
   
   - Submit the job file
   ```bash
   sbatch running_files/job_model_vqa_med_mistral.job
   ```
   - The results will be written in the answers file 

   

2. **Med-Flamingo**
   - Navigate to the Med-Flamingo directory:
     ```bash
     cd /home/aabhilash/med_vlms/med-flamingo
     ```

   - Activate the conda environment:
     ```bash
     conda activate /home/aabhilash/miniconda3/envs/med-flamingo
     ```

   - To test an instance, execute:
     ```bash
     sbatch running_files/job_run_med_flamingo.job
     ```
