## 1. Setup

1. If using OpenAI, set up environment variables:

```
export OPENAI_API_KEY={api_key}
```
In our experiments, we used ```gpt-turbo-3.5``` for all modules and ```gpt-turbo-3.5-instruct``` for the generative surrogate model (Note: these models might require separate set of credentials).

2. Confirm that environmental variables are set:
```
echo $OPENAI_API_KEY
```

3. Set up Conda environment:
```
git clone https://github.com/ChangLee0903/BO-SCULPT.git
conda create -n llambo python=3.8
conda install jupyter
conda activate llambo
## Note: {project_dir} is the path to where to your local directory
export PROJECT_DIR={project_dir}
conda env config vars set PYTHONPATH=${PYTHONPATH}:${PROJECT_DIR}
conda env config vars set PROJECT_DIR=${PROJECT_DIR}
conda deactivate
conda activate llambo
```

4. Install requirements:
```
pip install -r requirements.txt
```

---

## 2. Reproducing Results

To reproduce results, execute any of the shell experimental shell scripts:
- To run benchmark: ```run_benchmark.sh [METHOD]```
- To run Case I: ```run_case_1.sh [METHOD]```
- To run Case II: ```run_case_2.sh [METHOD]```
[METHOD] can be: LLAMBO-WOS, LLAMBO-WIS, GP, BO-SCULPT, BO-SCULPT-WOF