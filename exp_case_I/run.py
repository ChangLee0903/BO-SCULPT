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

given_init_samples = [[{'X0': 0.16787022705230384, 'score': 1.0162152382231089}, {'X0': 0.20431989807401735, 'score': 0.8862907646758729}, {'X0': 0.22005555340249583, 'score': 0.9207149086079071}, {'X0': 0.1837807529377828, 'score': 0.9052508953855476}, {'X0': 0.23014051809446356, 'score': 0.917011820004952}, {'X0': 0.1941841724364413, 'score': 0.9948577466168986}, {'X0': 0.19623224119087126, 'score': 0.9612720564801679}, {'X0': 0.17620147118495827, 'score': 0.95952794454456}, {'X0': 0.2403474498627854, 'score': 0.9712001053820877}, {'X0': 0.1666975570013308, 'score': 0.9211708943387813}, {'X0': 0.2379197535079442, 'score': 1.0334408061921354}, {'X0': 0.15363211053107417, 'score': 1.0259759886283097}, {'X0': 0.22646299995528527, 'score': 0.8989939719741671}, {'X0': 0.2080318072029286, 'score': 0.8953741565437557}, {'X0': 0.22610665086008785, 'score': 1.0416307201402957}], [{'X0': 0.24701120505797314, 'score': 0.8758323162745401}, {'X0': 0.24030209173168288, 'score': 0.9066441085446558}, {'X0': 0.15462443005175494, 'score': 0.9141109476427163}, {'X0': 0.20777720387083914, 'score': 0.8917423161681883}, {'X0': 0.15108830549431113, 'score': 1.0292060716425113}, {'X0': 0.17683919329058315, 'score': 1.0206155813720386}, {'X0': 0.20776619015165923, 'score': 0.932684849716725}, {'X0': 0.1735018498835323, 'score': 0.9870073086639432}, {'X0': 0.20166466054417978, 'score': 0.898591120306508}, {'X0': 0.17159719902374404, 'score': 1.0759748620908853}, {'X0': 0.24898224277547368, 'score': 0.9408068638230105}, {'X0': 0.1948215083840144, 'score': 1.0102041889370401}, {'X0': 0.21739922608137868, 'score': 0.8994545244104669}, {'X0': 0.22369683735744128, 'score': 0.9200039053943087}, {'X0': 0.18632737771367844, 'score': 1.01201912052653}], [{'X0': 0.15357848200303584, 'score': 1.022474117991913}, {'X0': 0.1905674298648981, 'score': 1.0071780204406564}, {'X0': 0.18755668177019436, 'score': 0.968471355217517}, {'X0': 0.1540598429492068, 'score': 1.0153656674708065}, {'X0': 0.23459960689756798, 'score': 0.9642842562298871}, {'X0': 0.2007519836562831, 'score': 0.9556448840833184}, {'X0': 0.17165285605554204, 'score': 0.973031914970659}, {'X0': 0.18712125912401478, 'score': 0.9007815904305011}, {'X0': 0.21615284754922964, 'score': 0.959881660996014}, {'X0': 0.220648101619935, 'score': 0.9625517998041011}, {'X0': 0.2484987336804216, 'score': 0.9398086909794686}, {'X0': 0.22535885015055615, 'score': 0.9258533887584848}, {'X0': 0.23331663744999537, 'score': 0.9932861046215367}, {'X0': 0.24778919338110664, 'score': 1.0063178031249547}, {'X0': 0.1519211713504272, 'score': 0.9809226943230474}], [{'X0': 0.220967619850754, 'score': 0.9520567245760505}, {'X0': 0.20468629015352482, 'score': 0.987100558487316}, {'X0': 0.2206970015496425, 'score': 0.9846024570089698}, {'X0': 0.23675354567214763, 'score': 0.8985310149936007}, {'X0': 0.24269831529871363, 'score': 0.9629612321835259}, {'X0': 0.18943429431689038, 'score': 0.8867035882111429}, {'X0': 0.2154190008367342, 'score': 1.0750286355500536}, {'X0': 0.24673609898665547, 'score': 0.9572537430545737}, {'X0': 0.24840242670433055, 'score': 0.908887661505632}, {'X0': 0.22741592099458785, 'score': 0.9486977752920764}, {'X0': 0.224655924426909, 'score': 0.9074192050087841}, {'X0': 0.17432203844564825, 'score': 0.9476313674468151}, {'X0': 0.1662725150616028, 'score': 0.9713220228056941}, {'X0': 0.1850864580282098, 'score': 0.9266605451755472}, {'X0': 0.20963196859833932, 'score': 0.8331580574661506}], [{'X0': 0.2325670032447637, 'score': 0.9351276795096277}, {'X0': 0.21233288502502873, 'score': 0.9464575640802969}, {'X0': 0.24760956704256296, 'score': 0.9848749221307382}, {'X0': 0.2392160493548244, 'score': 0.9856003830677904}, {'X0': 0.23666028968121028, 'score': 0.9047272407840881}, {'X0': 0.15731721881555122, 'score': 1.0270714694264322}, {'X0': 0.20676514569635174, 'score': 1.006837657096791}, {'X0': 0.1747337834474659, 'score': 0.9710560925638829}, {'X0': 0.21342515747150448, 'score': 0.9271030129261075}, {'X0': 0.20186651517896065, 'score': 0.994077551453846}, {'X0': 0.16661437922420508, 'score': 0.9587003479257106}, {'X0': 0.17433960549353292, 'score': 0.9435427490310375}, {'X0': 0.18965887547617213, 'score': 0.8478409228185781}, {'X0': 0.21903057503254367, 'score': 1.0096604793579163}, {'X0': 0.17715477391915607, 'score': 0.9801994415586365}]]

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
    'X0': ['float', 'logit', [-0.5, 0.5]]
    }
}

def f(x, noise=0.05):
    res = 1 - x**2 
    return res, res + noise * np.random.randn(*x.shape)

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
                        alpha=0.1, n_initial_samples=15, n_trials=25, init_f=benchmark.generate_initialization,
                        bbox_eval_f=benchmark.evaluate_point, chat_engine=chat_engine, top_pct=top_pct)
    elif 'GP' in args.method:
        bo_agent = BOGP(task_context, n_initial_samples=15, n_trials=25, init_f=benchmark.generate_initialization,
                        bbox_eval_f=benchmark.evaluate_point)
    elif 'BO-SCULPT' in args.method:
        use_filtering = False if 'WOF' in args.method else True
        bo_agent = BOSCULPT(task_context, sm_mode, n_candidates=10, n_templates=2, n_gens=10, prompt_setting='partial_context',
                        alpha=0.1, n_initial_samples=15, n_trials=25, init_f=benchmark.generate_initialization,
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