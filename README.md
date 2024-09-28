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

   

2. **For Running LLaVA Med 1.0**
   - For running llava med 1.0, we will need the base LLaVA 7B weights. Since Meta is not giving out llava med 1.0 weights anymore, it can be downloaded from, download the llama 7B weights from https://huggingface.co/huggyllama/llama-7b/tree/main
   - LLaVA 1.0, has published 4 different model weight deltas.
	1)LLaVA Med 1.0:  | LLaVA-Med | [llava_med_in_text_60k_ckpt2_delta.zip](https://hanoverprod.z21.web.core.windows.net/med_llava/models/llava_med_in_text_60k_ckpt2_delta.zip) | 11.06 GB |

	2)LLaVA Med 1.0 Finetuned on PathVQA: (https://hanoverprod.z21.web.core.windows.net/med_llava/models/pvqa-9epoch_delta.zip) | 11.06 GB |
     3) LLaVA Med 1.0 Finetuned on VQARAD: (https://hanoverprod.z21.web.core.windows.net/med_llava/models/data_RAD-9epoch_delta.zip) | 11.06 GB |
     4)LLaVA med 1.0 Finetuned on SLAKE: (https://hanoverprod.z21.web.core.windows.net/med_llava/models/Slake1.0-9epoch_delta.zip) | 11.06 GB |
* We will be using the LLaVA Med 1.0, LLaVA finetuned on VQA, LLaVA finetuned on SLAKE

Download the merged weights and store it in a directory to be used in the next step

- Create the environment or use the existent one at /home/aabhilash/miniconda3/etc/profile.d/conda.sh

For creating new, navigate to LLaVA Med 1.0 code directory
     ```bash
     cd llava-med-1/llava-med-1/LLaVAMed100
     ```
Then install the required packages with 
```bash
pip uninstall torch torchvision -y
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
pip install openai==0.27.8
pip uninstall transformers -y
pip install git+https://github.com/huggingface/transformers@cae78c46
pip install -e .
```

```
pip install einops ninja open-clip-torch
pip install flash-attn --no-build-isolation
```
   - To obtain the complete models, we have to merge these delta weights with the original LLaMA 7B weights. For this we apply the following merge jobs. 
     There are 3 job files in the running_files subdirectory, job_apply_delta_llava_med_in_text.job, job_apply_delta_vqa_rad.job, job_apply_delta_slake.job. In each of these files, set the required python path and activate the required conda environment according to the need. Set the path to the llama 7B weights file in the --base parameter. Set the path to the required models delta, downloaded in the previous step with the --delta parameter. Set the path to the directory in which the model has to be dumped with the target parameter.
     Now run each of the job files.
     ```bash
     sbatch job_apply_delta_llava_med_in_text.job
     sbatch job_apply_delta_vqa_rad.job
     sbatch job_apply_delta_slake.job
     ```
     The 3 completed model weights are written into the target directory specified. 
     
   - For the LLaVA med 1.0 model, we run it both on slake and vqa rad dataset. But for each of the finetuned model, we run it on the test sets of the dataset it was trained on. the files job_model_vqa_med_llava_med_in_vqa_rad.job, job_model_vqa_med_llava_med_in_slake.job, job_model_vqa_med_llava_med_slake_finetuned_test.job, job_model_vqa_med_llava_med_vqa_rad_finetuned_test.job, are for running llava med 1.0 on vqa rad, llava med 1.0 on slake, slake fine tuned llava med 1.0 on slake test set and vqa rad fine tuned llava med 1.0 on vqa rad test dataset. In each of these files, the path to the respective merged model files needs to be given along with the output file path in which the answer is expected. For images directory in slake, the path to Slake1.0/imgs needs to be given in the --images folder parameter and for vqa rad the path to VQA_RAD/VQA_RAD\ Image\ Folder needs to be given. For the question file parameter, when it is slake, we give the path to the test files in slake directory on the english language subset. If dataset downloaded from the path mentioned above, it is in the file english_slake_test.json. When it is VQA RAD, we need to give the test vqa set from VQA RAD. If downloaded from the path above, then it is in the file test_vqa_rad.json 
   
   ```
