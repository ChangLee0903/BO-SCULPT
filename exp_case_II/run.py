import os
import pickle
import json
import argparse
import logging
import warnings
import random
import pandas as pd
import numpy as np
import torch
from llambo.llambo import LLAMBO
from llambo.bosculpt import BOSCULPT
from llambo.bogp import BOGP

logger = logging.getLogger(__name__)

def setup_logging(log_name):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(log_name, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

given_init_samples = [
[{'X0': 1.416734933791386, 'score': 0.4552452722544271},
{'X0': 2.4378699885026323, 'score': 0.8041825382302547},
{'X0': 0.5022440296250482, 'score': 1.2860462483817403},
{'X0': 0.4130083835255146, 'score': 0.7830335552667724},
{'X0': 1.17160960373761, 'score': 0.5716735267864494}],

[{'X0': 2.159133226579807, 'score': 0.8019273055824868},
{'X0': 1.8675248560123823, 'score': 0.33195856124662143},
{'X0': 0.7583221539818021, 'score': 0.9764577838576896},
{'X0': 1.2739306282581466, 'score': 0.7968293564878675},
{'X0': 2.791753581091008, 'score': 0.5418446377750519}],
    
[{'X0': 0.31746686723286477, 'score': 1.0872638064641031},
{'X0': 2.0950160013623838, 'score': 0.7671265966734585},
{'X0': 1.6835763763225442, 'score': 0.5276172743803107},
{'X0': 0.015259763185942354, 'score': 0.536100816786582},
{'X0': 1.5209301457091775, 'score': 0.29037949863732787}],

[{'X0': 2.519228664881051, 'score': 0.7857424824890535},
{'X0': 2.57966609027231, 'score': 0.7361235490945105},
{'X0': 2.023987310247034, 'score': 0.7942282613641514},
{'X0': 1.9638279804974021, 'score': 0.6738940698289664},
{'X0': 1.553751133278745, 'score': 0.26819385799431467}],

[{'X0': 0.43205754239034233, 'score': 0.9556279806877352},
{'X0': 2.964659578820563, 'score': -0.002395256119087863},
{'X0': 1.7918423027643533, 'score': 0.5225772976699563},
{'X0': 1.2993072217081107, 'score': 0.4015230967179581},
{'X0': 1.3215699712244957, 'score': 0.40154463651720756}]
]

task_context = {
    'model': 'model', 
    'task': 'a task', 
    'tot_feats': 0, 
    'cat_feats': 0, 
    'num_feats': 0, 
    'n_classes': 0, 
    'metric': 'a metric', 
    'lower_is_better': False, 
    'num_samples': 0, 
    'hyperparameter_constraints': {
    'X0': ['float', 'logit', [0.0, 3.0]]
    }
}

def f(X, noise=0.7):
    X = X - 1
    res = -np.sin(3*X) - X**2 + 0.7*X
    nres = res + noise * np.random.randn(*X.shape)
    res = (res + 2.5) / (0.6+2.5)
    nres = (nres + 2.5) / (0.6+2.5)
    return res, nres

class DummyExpRunner:
    def __init__(self, task_context, func, seed=5566):
        self.seed = seed
        self.model = task_context['model']
        self.task = task_context['task']
        self.metric = task_context['metric']
        self.func = func
        self.seed = seed

    def generate_initialization(self, n_samples):
        '''
        Generate initialization points for BO search
        Args: n_samples (int)
        Returns: init_configs (list of dictionaries, each dictionary is a point to be evaluated)
        '''
        ndim = 1
        init_configs = []
        for n in range(n_samples):
            features = {}
            features['X0'] = given_init_samples[self.seed - 1][n]['X0']
            init_configs.append(features)
        assert len(init_configs) == n_samples

        return init_configs
        
    def evaluate_point(self, candidate_config):
        '''
        Evaluate a single point on bbox
        Args: candidate_config (dict), dictionary containing point to be evaluated
        Returns: (dict, dict), first dictionary is candidate_config (the evaluated point), second dictionary is fvals (the evaluation results)
        '''
        seed = random.choice(list(range(5566)))
        np.random.seed(seed)
        random.seed(seed)
        generalization_score, score = self.func(np.array(list(candidate_config.values())))
        results = {'score': score.max(), 'generalization_score': generalization_score[score.argmax()]}
        return candidate_config, results


if __name__ == '__main__':
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_name', type=str)
    parser.add_argument('--method', type=str)
    parser.add_argument('--engine', type=str) # temporary fix to run multiple in parallel
    parser.add_argument('--sm_mode', type=str)

    args = parser.parse_args()
    out_name =args.out_name
    chat_engine = args.engine
    sm_mode = args.sm_mode

    assert sm_mode in ['discriminative', 'generative']
    if sm_mode == 'generative':
        top_pct = 0.25
    else:
        top_pct = None

    # define result save directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_res_dir = f'{script_dir}/results_{sm_mode}'
    if not os.path.exists(save_res_dir):
        os.makedirs(save_res_dir)
    # define logging directory
    logging_fpath = f'{script_dir}/logs_{sm_mode}.log'
    if not os.path.exists(os.path.dirname(logging_fpath)):
        os.makedirs(os.path.dirname(logging_fpath))
    setup_logging(logging_fpath)

    tot_llm_cost = 0
    logger.info('='*200)
    logger.info(f'Executing Bayesian Optimization ({sm_mode} | top_pct: {top_pct})...')
    logger.info(f'Task context: {task_context}')

    seed = int(args.out_name[-1])
    benchmark = DummyExpRunner(task_context, f, seed)
    
    # instantiate agent
    assert args.method in ['LLAMBO-WOS', 'GP', 'BO-SCULPT', 'BO-SCULPT-WOF']
    if 'LLAMBO' in args.method:
        bo_agent = LLAMBO(task_context, sm_mode, n_candidates=10, n_templates=2, n_gens=10, prompt_setting='partial_context',
                        alpha=0.1, n_initial_samples=5, n_trials=25, init_f=benchmark.generate_initialization,
                        bbox_eval_f=benchmark.evaluate_point, chat_engine=chat_engine, top_pct=top_pct)
    elif 'GP' in args.method:
        bo_agent = BOGP(task_context, n_initial_samples=5, n_trials=25, init_f=benchmark.generate_initialization,
                        bbox_eval_f=benchmark.evaluate_point)
    elif 'BO-SCULPT' in args.method:
        use_filtering = False if 'WOF' in args.method else True
        bo_agent = BOSCULPT(task_context, sm_mode, n_candidates=10, n_templates=2, n_gens=10, prompt_setting='partial_context',
                        alpha=0.1, n_initial_samples=5, n_trials=25, init_f=benchmark.generate_initialization,
                        bbox_eval_f=benchmark.evaluate_point, chat_engine=chat_engine, top_pct=top_pct, use_filtering=use_filtering)
    
    bo_agent.seed = seed
    configs, fvals = bo_agent.optimize()


    logger.info(f'[AGENT] Query cost: {sum(bo_agent.llm_query_cost):.4f}')
    logger.info(f'[AGENT] Query time: {sum(bo_agent.llm_query_time):.4f}')
    tot_llm_cost += sum(bo_agent.llm_query_cost)

    # save search history
    search_history = pd.concat([configs, fvals], axis=1)
    search_history.to_csv(f'{save_res_dir}/{out_name}.csv', index=False)

    
    logger.info(search_history)
    logger.info(f'[AGENT] RUN COMPLETE, saved results to {save_res_dir}...')

    # save search info
    search_info = {
        'llm_query_cost_breakdown': bo_agent.llm_query_cost,
        'llm_query_time_breakdown': agent.llm_query_time,
        'llm_query_cost': sum(bo_agent.llm_query_cost),
        'llm_query_time': sum(bo_agent.llm_query_time),
    }
    with open(f'{save_res_dir}/{seed}_search_info.json', 'w') as f:
        json.dump(search_info, f)

    logger.info('='*200)
    logger.info(f'[AGENT] {seed+1} evaluation runs complete! Total cost: ${tot_llm_cost:.4f}')