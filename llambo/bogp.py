import pandas as pd
import time
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.utils.transforms import normalize, unnormalize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement

class BOGP:
    def __init__(self, 
                 task_context,
                 n_initial_samples, # number of initial samples to evaluate
                 n_trials,   # number of trials to run,
                 init_f,        # function to generate initial configurations
                 bbox_eval_f,       # bbox function to evaluate a point
                 lower_is_better=False,
                **kwargs):
        self.task_context = task_context
        self.n_initial_samples = n_initial_samples
        self.n_trials = n_trials
        self.lower_is_better = lower_is_better
        self.llm_query_cost = []    # list of cost for LLM calls in EACH TRIAL
        self.llm_query_time = []    # list of time taken for LLM calls in EACH TRIAL
        
        self.init_f = init_f
        self.bbox_eval_f = bbox_eval_f

    def process_gp_model(self, configs, fvals, n_candidates=1):
        s = time.time()
        with torch.no_grad():
            hyperparameter_names = configs.columns
            train_X = torch.from_numpy(configs[hyperparameter_names].to_numpy())
            bounds = []
            for i in range(len(hyperparameter_names)):
                bounds.append(torch.Tensor(\
                    self.task_context['hyperparameter_constraints'][hyperparameter_names[i]][2]\
                        ).unsqueeze(1))
            bounds = torch.cat(bounds, dim=1)
            train_Y = torch.from_numpy(fvals.to_numpy())
            gp = SingleTaskGP(
                train_X=train_X,
                train_Y=train_Y
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        with torch.no_grad():
            baseline = gp(train_X).loc.max()
        logEI = qExpectedImprovement(model=gp, best_f=baseline)
        candidates, acq_value = optimize_acqf(
            logEI, bounds=bounds, q=n_candidates, num_restarts=20, 
            raw_samples=100, options = {"batch_limit": 5, "maxiter": 200}
        )
        candidates = candidates.numpy()
        can_dict = {}
        for i in range(candidates.shape[1]):
            can_dict[hyperparameter_names[i]] = candidates[:, i]
        candidates = pd.DataFrame(can_dict)
        e = time.time()
        return gp, candidates, e - s

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
        for trial_id in range(self.n_trials):
            trial_query_time = 0

            start_time = time.time()
            # get candidate point
            gp, sel_candidate_point, time_taken = self.process_gp_model(self.observed_configs, self.observed_fvals[['score']])
            trial_query_time += time_taken

            print('='*150)
            print('EXAMPLE POINTS PROPOSED')
            print(sel_candidate_point)
            print('='*150)

            self.llm_query_time.append(trial_query_time)

            print('='*150)
            print('SELECTED CANDIDATE POINT')
            print(sel_candidate_point)
            print('='*150)


            # evaluate candidate point
            sel_candidate_point, sel_candidate_fval = self._evaluate_config(sel_candidate_point)

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