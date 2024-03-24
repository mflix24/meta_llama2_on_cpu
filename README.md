# meta_llama2_on_cpu
This is a repository about llama2 model which downloaded from HuggingFace and loaded on local CPU. This model is a quantized model menas that it reduces the computation cost of 
neural network training, which can replace high-cost floating-point numbers 
(e.g., float32) with low-cost fixed-point numbers. This is how the model works.

we have 3 important .py files
(a). run_local.py ; this file for quantized model runs into our local CPU.
(b). main.py ; this file for RAG system meant Knowledge Based System.
(c). app.py ; this file for UI design ; here we use flask for UI design.

The model downloaded link is given below : Model Name: llama-2-7b-chat.ggmlv3.q4_0.bin
    https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main

### Steps to be followed :

#### Step-01 : clone the repository through commandPromptTerminal
'''
git clone https://github.com/mflix24/meta_llama2_on_cpu.git
'''

#### Step-02 : create a virtual environment and activate it by using these commands
'''
way-1 : conda create -p envname python==3.10 -y
way-2 : conda create -n envname python==3.10 -y
        conda activate envname
'''

#### Step-03 : installing all dependencies through pip
'''
pip install -r requirements.txt
'''


