from llambo.mine_discriminative_sm import LLM_DIS_SM
from llambo.mine_generative_sm import LLM_GEN_SM
from llambo.mine_acquisition_function import LLM_ACQ
from llambo.rate_limiter import RateLimiter
from llambo.warping import NumericalTransformer
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.utils.transforms import normalize, unnormalize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
import pandas as pd
import numpy as np
import random
import time
import pprint

class BOSCULPT:
    def __init__(self, 
                 task_context: dict, # dictionary describing task (see above)
                 sm_mode, # either 'generative' or 'discriminative'
                 n_candidates, # number of candidate points to sample at each iteration
                 n_templates, # number of templates for LLM queries
                 n_gens,    # number of generations for LLM, set at 5
                 alpha,    # alpha for LLM, recommended to be -0.2
                 n_initial_samples, # number of initial samples to evaluate
                 n_trials,   # number of trials to run,
                 init_f,        # function to generate initial configurations
                 bbox_eval_f,       # bbox function to evaluate a point
                 chat_engine,       # LLM chat engine
                 top_pct=None,      # only used for generative SM, top percentage of points to consider for generative SM
                 use_input_warping=False,       # whether to use input warping
                 prompt_setting=None,    # ablation on prompt design, either 'full_context' or 'partial_context' or 'no_context'
                 shuffle_features=False,     # whether to shuffle features in prompt generation
                 use_filtering=True
                 ):
        self.task_context = task_context
        assert sm_mode in ['generative', 'discriminative']
        assert top_pct is None if sm_mode == 'discriminative' else top_pct is not None
        self.model_name = task_context['model']
        self.lower_is_better = task_context['lower_is_better']
        lower_is_better = self.lower_is_better
        self.n_candidates = n_candidates
        self.n_template = n_templates
        self.n_gens = n_gens
        self.alpha = alpha
        self.n_initial_samples = n_initial_samples
        self.n_trials = n_trials
        self.use_filtering = use_filtering
        self.llm_query_cost = []    # list of cost for LLM calls in EACH TRIAL
        self.llm_query_time = []    # list of time taken for LLM calls in EACH TRIAL

        assert type(shuffle_features) == bool, 'shuffle_features should be a boolean'
        assert type(use_input_warping) == bool, 'use_input_warping should be a boolean'

        self.init_f = init_f
        self.bbox_eval_f = bbox_eval_f

        if use_input_warping:
            warping_transformer = NumericalTransformer(task_context['hyperparameter_constraints'])
        else:
            warping_transformer = None

        rate_limiter = RateLimiter(max_tokens=100000, time_frame=60, max_requests=720)
        
        print('='*150)
        print(f'[Search settings]: ' + '\n\t'
              f'n_candidates: {n_candidates}, n_templates: {n_templates}, n_gens: {n_gens}, ' + '\n\t'
              f'alpha: {alpha}, n_initial_samples: {n_initial_samples}, n_trials: {n_trials}, ' + '\n\t'
              f'using warping: {use_input_warping}, ablation: {prompt_setting}, '
              f'shuffle_features: {shuffle_features}')
        print(f'[Task]: ' + '\n\t'
              f'task type: {task_context["task"]}, sm: {sm_mode}, lower is better: {lower_is_better}')
        print(f'Hyperparameter search space: ')
        pprint.pprint(task_context['hyperparameter_constraints'])
        print('='*150)

        # initialize surrogate model and acquisition function
        if sm_mode == 'generative':
            self.surrogate_model = LLM_GEN_SM(task_context, n_gens*2, lower_is_better, top_pct,
                                              n_templates=n_templates, rate_limiter=None)
        else:
            self.surrogate_model = LLM_DIS_SM(task_context, n_gens*2, lower_is_better, 
                                              n_templates=n_templates, rate_limiter=rate_limiter, 
                                              warping_transformer=warping_transformer,
                                              chat_engine=chat_engine, prompt_setting=prompt_setting, 
                                              shuffle_features=shuffle_features)
            
        self.acq_func = LLM_ACQ(task_context, n_candidates, n_templates, lower_is_better, 
                                rate_limiter=rate_limiter, warping_transformer=warping_transformer, 
                                chat_engine=chat_engine, prompt_setting=prompt_setting, 
                                shuffle_features=shuffle_features)


    def _initialize(self):
        '''Initialize the optimization loop.'''
        start_time = time.time()
        # generate initial configurations
        init_configs = self.init_f(self.n_initial_samples)

        assert isinstance(init_configs, list), 'init_f() should return a list of configs (dictionaries)'
        for item in init_configs:
            assert isinstance(item, dict), 'init_f() should return a list of configs (dictionaries)'

        init_configs = pd.DataFrame(init_configs)
        assert init_configs.shape[0] == self.n_initial_samples, 'init_f() should return n_initial_samples number of configs'

        # create empty pandas dataframe for observed function values
        self.observed_fvals = pd.DataFrame()
        self.observed_configs = pd.DataFrame()

        for index, _ in init_configs.iterrows():
            one_config = init_configs.iloc[[index]]
            one_config, one_result = self._evaluate_config(one_config)

            if self.observed_fvals.empty:
                self.observed_fvals = one_result
            else:
                self.observed_fvals = pd.concat([self.observed_fvals, one_result], axis=0, ignore_index=True)

            if self.observed_configs.empty:
                self.observed_configs = one_config
            else:
                self.observed_configs = pd.concat([self.observed_configs, one_config], axis=0, ignore_index=True)

        print(f'[Initialization] COMPLETED: {self.observed_fvals.shape[0]} points evaluated...')
        end_time = time.time()

        time_taken = end_time - start_time
        return 0, time_taken
    
    def _evaluate_config(self, config):
        # can support batch mode in the future
        assert config.shape[0] == 1, 'batch mode not supported yet'
        config = config.to_dict('records')[0]

        eval_config, eval_results = self.bbox_eval_f(config)

        assert isinstance(eval_config, dict), 'bbox_eval_f() should return the evaluated config as a dictinoary'
        assert isinstance(eval_results, dict), 'bbox_eval_f() should return bbox evaluation results as a dictionary'
        assert 'score' in eval_results.keys(), 'score must be a key in results returned'

        return pd.DataFrame([eval_config]), pd.DataFrame([eval_results])

    def _update_observations(self, new_config, new_fval):
        '''Update the observed configurations and function values.'''
        # append new observations
        self.observed_configs = pd.concat([self.observed_configs, new_config], axis=0, ignore_index=True)
        self.observed_fvals = pd.concat([self.observed_fvals, new_fval], axis=0, ignore_index=True)

    def fit_gp_model(self, configs, fvals):
        with torch.no_grad():
            hyperparameter_names = configs.columns
            train_X = torch.from_numpy(configs[hyperparameter_names].to_numpy())
            train_Y = torch.from_numpy(fvals.to_numpy())
            gp = SingleTaskGP(
                train_X=train_X,
                train_Y=train_Y
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        return gp

    def process_gp_model(self, gp, configs, existed_candidates=None, n_candidates=1):
        s = time.time()
        with torch.no_grad():
            hyperparameter_names = configs.columns
            train_X = torch.from_numpy(configs[hyperparameter_names].to_numpy())
            if existed_candidates is not None:
                exist_X = torch.from_numpy(existed_candidates[hyperparameter_names].to_numpy())
            bounds = []
            for i in range(len(hyperparameter_names)):
                bounds.append(torch.Tensor(\
                    self.task_context['hyperparameter_constraints'][hyperparameter_names[i]][2]\
                        ).unsqueeze(1))
            bounds = torch.cat(bounds, dim=1)
            
        with torch.no_grad():
            pred = gp(train_X)
            baseline = pred.loc.max()
            thr = pred.covariance_matrix.diag().sqrt()[pred.loc.argmax()]
        if existed_candidates is not None:
            pred = gp(exist_X)
            exist_X = exist_X[(pred.loc + 2*pred.covariance_matrix.diag().sqrt()) > (baseline)]
            n_candidates = n_candidates - len(exist_X)
        logEI = qExpectedImprovement(model=gp, best_f=baseline)
        if n_candidates > 0:
            candidates, acq_value = optimize_acqf(
                logEI, bounds=bounds, q=n_candidates, num_restarts=20, 
                raw_samples=100, options = {"batch_limit": 5, "maxiter": 200}
            )
            if existed_candidates is not None:
                candidates = torch.cat([candidates, exist_X])
        else:
            candidates = exist_X
        best_candidate, _ = optimize_acqf(
                logEI, bounds=bounds, q=1, num_restarts=20, 
                raw_samples=100, options = {"batch_limit": 5, "maxiter": 200}
        )
        candidates = candidates.numpy()
        best_candidate = best_candidate.numpy()
        can_dict = {}
        best_can_dict = {}
        for i in range(candidates.shape[1]):
            can_dict[hyperparameter_names[i]] = candidates[:, i]
            best_can_dict[hyperparameter_names[i]] = best_candidate[:, i]
            if self.task_context['hyperparameter_constraints'][hyperparameter_names[i]][0] == 'int':
                can_dict[hyperparameter_names[i]] = np.round(can_dict[hyperparameter_names[i]])
                best_can_dict[hyperparameter_names[i]] = np.round(best_can_dict[hyperparameter_names[i]])
        candidates = pd.DataFrame(can_dict)
        candidates = candidates.drop_duplicates()
        best_candidate_point = pd.DataFrame(best_can_dict)
        e = time.time()
        return candidates, best_candidate_point, e - s

    def optimize(self, test_metric='generalization_score'):
        '''Run the optimization loop.'''
        # initialize
        cost, query_time = self._initialize()
        self.llm_query_cost.append(cost)
        self.llm_query_time.append(query_time)

        if self.lower_is_better:
            self.best_fval = self.observed_fvals['score'].min()
            best_gen_fval = self.observed_fvals[test_metric].min()
        else:
            self.best_fval = self.observed_fvals['score'].max()
            best_gen_fval = self.observed_fvals[test_metric].max()

        print(f'[Initialization] COMPLETED: best fval: {self.best_fval:.4f}, best generalization fval: {best_gen_fval:.4f}')
        print('='*150)

        # optimization loop
        prob = 0.5
        m = 0.05
        gp_best = 0.5
        best = 0.5
        llm_best = 0.5
        run_by_gp = False
        duplicated_count = 0
        prev_selection = None
        for trial_id in range(self.n_trials):
            trial_cost = 0
            trial_query_time = 0

            start_time = time.time()
            # get candidate point
            hyperparameter_names = [k for k in self.observed_configs.columns if not 'estimated' in k]
            gp = self.fit_gp_model(self.observed_configs[hyperparameter_names], self.observed_fvals[['score']])
            with torch.no_grad():
                reference_configs = {'sampled': self.observed_configs.copy()[~self.observed_configs.duplicated()]}
                gp_pred = gp(torch.from_numpy(reference_configs['sampled'][hyperparameter_names].to_numpy()))
                reference_configs['sampled']['S'] = self.observed_fvals['score'][~self.observed_configs.duplicated()]
                reference_configs['sampled']['S_avg'] = gp_pred.loc.numpy()
                reference_configs['sampled']['S_min'] = gp_pred.loc.numpy() - 2*gp_pred.covariance_matrix.diag().sqrt().numpy()
                reference_configs['sampled']['S_max'] = gp_pred.loc.numpy() + 2*gp_pred.covariance_matrix.diag().sqrt().numpy()
                reference_configs['sampled'] = reference_configs['sampled'].sort_values('S_max', ascending=False).head(5)
            gp_candidate_points, best_candidate_point, time_taken = self.process_gp_model(gp, self.observed_configs[hyperparameter_names], n_candidates=5)
            with torch.no_grad():
                gp_pred = gp(torch.from_numpy(gp_candidate_points[hyperparameter_names].to_numpy()))
                reference_configs['unsampled'] = gp_candidate_points.copy()
                reference_configs['unsampled']['S_avg'] = gp_pred.loc.numpy()
                reference_configs['unsampled']['S_min'] = gp_pred.loc.numpy() - 2*gp_pred.covariance_matrix.diag().sqrt().numpy()
                reference_configs['unsampled']['S_max'] = gp_pred.loc.numpy() + 2*gp_pred.covariance_matrix.diag().sqrt().numpy()
            
            candidate_points, cost, time_taken = self.acq_func.get_candidate_points(self.observed_configs, self.observed_fvals[['score']], reference_configs, alpha=self.alpha)
            candidate_points = candidate_points.drop_duplicates()
            
            if self.use_filtering:
                gp_candidate_points, best_gp_candidate_point, time_taken = self.process_gp_model(gp, self.observed_configs[hyperparameter_names], candidate_points, n_candidates=10)
                with torch.no_grad():
                    gp_pred = gp(torch.from_numpy(gp_candidate_points[hyperparameter_names].to_numpy()))
                    reference_configs['unsampled'] = gp_candidate_points.copy()
                    reference_configs['unsampled']['S_avg'] = gp_pred.loc.numpy()
                    reference_configs['unsampled']['S_min'] = gp_pred.loc.numpy() - 2*gp_pred.covariance_matrix.diag().sqrt().numpy()
                    reference_configs['unsampled']['S_max'] = gp_pred.loc.numpy() + 2*gp_pred.covariance_matrix.diag().sqrt().numpy()
                candidate_points = gp_candidate_points
            trial_query_time += time_taken

            
            trial_cost += cost
            trial_query_time += time_taken

            print('='*150)
            print('EXAMPLE POINTS PROPOSED')
            print(candidate_points)
            if self.use_filtering:
                print('EXAMPLE GP POINTS PROPOSED')
                print(best_gp_candidate_point)
            print('='*150)

            # select candidate point
            sel_candidate_point, cost, time_taken = self.surrogate_model.select_query_point(self.observed_configs, 
                                                                        self.observed_fvals[['score']], 
                                                                        candidate_points,
                                                                        reference_configs)
            if prev_selection is not None:
                if np.sum(sel_candidate_point.to_numpy() - prev_selection.to_numpy())**2 / \
                    np.sum(prev_selection.to_numpy()**2) < 0.01:
                    duplicated_count += 1
                else:
                    duplicated_count = 0
            prev_selection = sel_candidate_point
            if duplicated_count >= 2:
                sel_candidate_point = reference_configs['unsampled'].sort_values(['S_avg', 'S_max'], ascending=False).head(5)
                sel_candidate_point = sel_candidate_point[prev_selection.keys()].sample()
            trial_cost += cost
            trial_query_time += time_taken

            self.llm_query_cost.append(trial_cost)
            self.llm_query_time.append(trial_query_time)

            print('='*150)
            print('SELECTED CANDIDATE POINT')
            print(sel_candidate_point)
            print('='*150)


            # evaluate candidate point
            sel_candidate_point, sel_candidate_fval = self._evaluate_config(sel_candidate_point)
            # new = min(1.0, sel_candidate_fval['score'].values.max())
            # if new >= best:
            #     best = new * 0.1 + best * 0.9
            #     if run_by_gp:
            #         print(f'query by GP with prob {prob:.4f} and best fval {gp_best:.4f}!')
            #         prob += m
            #     else:
            #         print(f'query by LLM with prob {(1-prob):.4f} and best fval {llm_best:.4f}!')
            #         prob -= m
            # print(new, prob, random.uniform(0, 1.0), run_by_gp, best, '!!!')
            # if run_by_gp:
            #         # gp_best = new * 0.5 + gp_best * 0.5
            #     prob += m
            #     # else:
            #     #     prob -= m
            #     print(f'query by GP with prob {prob:.4f} and best fval {gp_best:.4f}!')
            # else:
            #     # if new >= llm_best:
            #     #     llm_best = new * 0.5 + llm_best * 0.5
            #     #     if llm_best > gp_best:
            #     #         prob -= m
            #     prob -= m
            #     # else:
            #     #     prob += m
            #     print(f'query by LLM with prob {(1-prob):.4f} and best fval {llm_best:.4f}!')
            # prob = min(max(0.05, prob), 0.95)
            # if llm_best > gp_best and not run_by_gp:
            #     prob -= m
            # elif llm_best <= gp_best and run_by_gp:
            #     prob += m
            # run_by_gp = False

            # update observations
            self._update_observations(sel_candidate_point, sel_candidate_fval)

            print('='*150)
            print('UPDATED OBSERVATIONS')
            print(self.observed_configs)
            print(self.observed_fvals)
            print('='*150)

            end_time = time.time()
            time_taken = end_time - start_time

            current_fval_cv = sel_candidate_fval['score'].values[0]
            current_fval_gen = sel_candidate_fval[test_metric].values[0]

            if self.lower_is_better:
                if current_fval_cv < self.best_fval:
                    self.best_fval = current_fval_cv
                    best_found = True
                else:
                    best_found = False
            else:
                if current_fval_cv > self.best_fval:
                    self.best_fval = current_fval_cv
                    best_found = True
                else:
                    best_found = False

            if best_found:
                print(f'[Trial {trial_id} completed, time taken: {time_taken:.2f}s] best fval (cv): {self.best_fval:.4f}, current fval (cv): {current_fval_cv:.4f}. Generalization fval: {current_fval_gen:.4f} NEW BEST FVAL FOUND!!')
            else: 
                print(f'[Trial {trial_id} completed, time taken: {time_taken:.2f}s] best fval (cv): {self.best_fval:.4f}, current fval (cv): {current_fval_cv:.4f}. Generalization fval: {current_fval_gen:.4f}.')
            print('='*150)

        # returns history of observed configurations and function values
        return self.observed_configs, self.observed_fvals