import os
import pickle
import json
import argparse
import logging
import warnings
import random
import pandas as pd
import numpy as np
from llambo.llambo import LLAMBO
from llambo.bosculpt import BOSCULPT
from llambo.bogp import BOGP
from bayesmark.bbox_utils import get_bayesmark_func
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)

def setup_logging(log_name):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(log_name, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

BAYESMARK_TASK_MAP = {
    'digits': ['classification', 'accuracy'],
    'wine': ['classification', 'accuracy'],
}

class BayesmarkExpRunner:
    def __init__(self, task_context, dataset, seed, use_unk_mode=False):
        self.seed = seed
        self.model = task_context['model']
        self.task = task_context['task']
        self.metric = task_context['metric']
        self.dataset = dataset
        self.hyperparameter_constraints = task_context['hyperparameter_constraints']
        self.bbox_func = get_bayesmark_func(self.model, self.task, dataset['test_y'])
        self.feat2name = {}
        self.name2feat = {}
        if use_unk_mode:
            for i, k in enumerate(self.hyperparameter_constraints):
                self.feat2name[k] = f'X{i}'
                self.name2feat[f'X{i}'] = k
        else:
            for k in self.hyperparameter_constraints:
                self.feat2name[k] = k
                self.name2feat[k] = k

    def generate_initialization(self, n_samples):
        '''
        Generate initialization points for BO search
        Args: n_samples (int)
        Returns: list of dictionaries, each dictionary is a point to be evaluated
        '''

        # Read from fixed initialization points (all baselines see same init points)
        init_configs = pd.read_json(f'bayesmark/configs/{self.model}/{self.seed}.json').head(n_samples)
        init_configs = init_configs.to_dict(orient='records')
        assert len(init_configs) == n_samples
        
        tmp = []
        for cfg in init_configs:
            tmp.append({self.feat2name[k]: v for k, v in cfg.items()})
        init_configs = tmp
        return init_configs
        
    def evaluate_point(self, candidate_config):
        '''
        Evaluate a single point on bbox
        Args: candidate_config (dict), dictionary containing point to be evaluated
        Returns: (dict, dict), first dictionary is candidate_config (the evaluated point), second dictionary is fvals (the evaluation results)
        fvals can contain an arbitrary number of items, but also must contain 'score' (which is what LLAMBO optimizer tries to optimize)
        fvals = {
            'score': float,                     -> 'score' is what the LLAMBO optimizer tries to optimize
            'generalization_score': float
        }
        '''
        np.random.seed(self.seed)
        random.seed(self.seed)

        X_train, X_test, y_train, y_test = self.dataset['train_x'], self.dataset['test_x'], self.dataset['train_y'], self.dataset['test_y']

        for hyperparam, value in candidate_config.items():
            if self.hyperparameter_constraints[self.name2feat[hyperparam]][0] == 'int':
                candidate_config[hyperparam] = round(value)

        if self.task == 'regression':
            mean_ = np.mean(y_train)
            std_ = np.std(y_train)
            y_train = (y_train - mean_) / std_
            y_test = (y_test - mean_) / std_
        
        scorer = get_scorer(self.metric)
        model = self.bbox_func(**{self.name2feat[k]: candidate_config[k] \
            for k in candidate_config})
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            S = cross_val_score(model, X_train, y_train, scoring=scorer, cv=5)
        cv_score = np.mean(S)
        
        model = self.bbox_func(**{self.name2feat[k]: candidate_config[k] \
            for k in candidate_config})
        model.fit(X_train, y_train)
        generalization_score = scorer(model, X_test, y_test)

        if self.metric == 'neg_mean_squared_error':
            cv_score = -cv_score
            generalization_score = -generalization_score

        return candidate_config, {'score': cv_score, 'generalization_score': generalization_score}



if __name__ == '__main__':
    # parse input arguments
    import random
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--out_name', type=str)
    parser.add_argument('--method', type=str)
    parser.add_argument('--engine', type=str) # temporary fix to run multiple in parallel
    parser.add_argument('--sm_mode', type=str)

    args = parser.parse_args()
    out_name =args.out_name
    dataset = args.dataset
    model = args.model
    chat_engine = args.engine
    sm_mode = args.sm_mode

    assert sm_mode in ['discriminative', 'generative']
    if sm_mode == 'generative':
        top_pct = 0.25
    else:
        top_pct = None

    # Load training and testing data
    if dataset in BAYESMARK_TASK_MAP:
        TASK_MAP = BAYESMARK_TASK_MAP
        pickle_fpath = f'bayesmark/data/{dataset}.pickle'
        with open(pickle_fpath, 'rb') as f:
            data = pickle.load(f)
        X_train = data['train_x']
        y_train = data['train_y']
        X_test = data['test_x']
        y_test = data['test_y']
    else:
        raise ValueError(f'Invalid dataset: {dataset}')


    # Describe task context
    task_context = {}
    task_context['model'] = model
    task_context['task'] = TASK_MAP[dataset][0]
    task_context['tot_feats'] = X_train.shape[1]
    task_context['cat_feats'] = 0       # bayesmark datasets only have numerical features
    task_context['num_feats'] = X_train.shape[1]
    task_context['n_classes'] = len(np.unique(y_train))
    task_context['metric'] = TASK_MAP[dataset][1]
    task_context['lower_is_better'] = True if 'neg' in task_context['metric'] else False
    task_context['num_samples'] = X_train.shape[0]
    with open('hp_configurations/bayesmark.json', 'r') as f:
        task_context['hyperparameter_constraints'] = json.load(f)[model]

    # define result save directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_res_dir = f'{script_dir}/results_{sm_mode}/{dataset}/{model}'
    if not os.path.exists(save_res_dir):
        os.makedirs(save_res_dir)
    # define logging directory
    logging_fpath = f'{script_dir}/logs_{sm_mode}/{dataset}/{model}.log'
    if not os.path.exists(os.path.dirname(logging_fpath)):
        os.makedirs(os.path.dirname(logging_fpath))
    setup_logging(logging_fpath)

    tot_llm_cost = 0
    logger.info('='*200)
    logger.info(f'Executing Bayesian Optimization ({sm_mode} | top_pct: {top_pct})...')
    logger.info(f'Task context: {task_context}')

    seed = int(args.out_name[-1])
    assert args.method in ['LLAMBO-WOS', 'LLAMBO-WIS', 'GP', 'BO-SCULPT', 'BO-SCULPT-WOF']
    prompt_setting = 'full_context'
    use_unk_mode = True if args.method != 'LLAMBO-WIS' else False
    benchmark = BayesmarkExpRunner(task_context, data, seed, use_unk_mode)

    if use_unk_mode:
        tmp = {}
        for k in task_context['hyperparameter_constraints']:
            tmp[benchmark.feat2name[k]] = task_context['hyperparameter_constraints'][k]
        task_context['hyperparameter_constraints'] = tmp
    
        prompt_setting = 'partial_context'
        task_context['model'] = 'model'
        task_context['task'] = 'a task'
        task_context['tot_feats'] = 0
        task_context['cat_feats'] = 0       # bayesmark datasets only have numerical features
        task_context['num_feats'] = 0
        task_context['n_classes'] = 0
        task_context['metric'] = 'a metric'
        task_context['lower_is_better'] = False
        task_context['num_samples'] = 0

    # instantiate LLAMBO
    if 'GP' in args.method:
        bo_agent = BOGP(task_context, n_initial_samples=5, n_trials=25, init_f=benchmark.generate_initialization,
                bbox_eval_f=benchmark.evaluate_point)
    elif 'LLAMBO' in args.method:
        bo_agent = LLAMBO(task_context, sm_mode, n_candidates=10, n_templates=2, n_gens=10, prompt_setting=prompt_setting,
                        alpha=0.1, n_initial_samples=5, n_trials=25, init_f=benchmark.generate_initialization,
                        bbox_eval_f=benchmark.evaluate_point, chat_engine=chat_engine, top_pct=top_pct)
    elif 'BO-SCULPT' in args.method:
        use_filtering = False if 'WOF' in args.method else True
        bo_agent = BOSCULPT(task_context, sm_mode, n_candidates=10, n_templates=2, n_gens=10, prompt_setting=prompt_setting,
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