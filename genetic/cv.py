# -*- coding: utf-8 -*-
from __future__ import division
import os
import warnings

import numpy as np
import random
from deap import base, creator, tools, algorithms
from collections import defaultdict
from sklearn.base import clone, is_classifier
from sklearn.model_selection._validation import _fit_and_score
from sklearn.model_selection._search import BaseSearchCV, check_cv, _check_param_grid
from sklearn.metrics.scorer import check_scoring
from sklearn.utils.validation import _num_samples, indexable


def enum(**enums):
    return type('Enum', (), enums)


param_types = enum(Categorical=1, Numerical=2)


def _get_param_types_maxint(params):
    name_values = list(params.items())
    types = []
    for _, possible_values in name_values:
        if isinstance(possible_values[0], float):
            types.append(param_types.Numerical)
        else:
            types.append(param_types.Categorical)
    maxints = [len(possible_values) - 1 for _, possible_values in name_values]
    return name_values, types, maxints


def _initIndividual(pcls, maxints):
    part = pcls(random.randint(0, maxint) for maxint in maxints)
    return part


def _mutIndividual(individual, up, indpb, gene_type=None):
    for i, up, rn in zip(range(len(up)), up, [random.random() for _ in range(len(up))]):
        if rn < indpb:
            individual[i] = random.randint(0, up)
    return individual,


def _cxIndividual(ind1, ind2, indpb, gene_type):
    for i, gt, rn in zip(range(len(ind1)), gene_type, [random.random() for _ in range(len(ind1))]):
        if rn > indpb:
            continue
        if gt is param_types.Categorical:
            ind1[i], ind2[i] = ind2[i], ind1[i]
        else:
            # Case when parameters are numerical
            if ind1[i] <= ind2[i]:
                ind1[i] = random.randint(ind1[i], ind2[i])
                ind2[i] = random.randint(ind1[i], ind2[i])
            else:
                ind1[i] = random.randint(ind2[i], ind1[i])
                ind2[i] = random.randint(ind2[i], ind1[i])

    return ind1, ind2


def _individual_to_params(individual, name_values):
    return dict((name, values[gene]) for gene, (name, values) in zip(individual, name_values))


def _evalFunction(individual, name_values, X, y, scorer, cv, iid, fit_params,
                  verbose=0, error_score='raise', score_cache={}):
    parameters = _individual_to_params(individual, name_values)
    score = 0
    n_test = 0

    paramkey = str(individual)
    if paramkey in score_cache:
        score = score_cache[paramkey]
    else:
        for train, test in cv.split(X, y):
            assert len(train) > 0 and len(test) > 0, "Training and/or testing not long enough for evaluation."
            _score = _fit_and_score(estimator=individual.est, X=X, y=y, scorer=scorer,
                                    train=train, test=test, verbose=verbose,
                                    parameters=parameters, fit_params=fit_params,
                                    error_score=error_score)[0]

            if iid:
                score += _score * len(test)
                n_test += len(test)
            else:
                score += _score
                n_test += 1

        assert n_test > 0, "No fitting was accomplished, check data and cross validation method."
        score /= float(n_test)
        score_cache[paramkey] = score

    return (score,)


class GeneticSearchCV(BaseSearchCV):

    def __init__(self, estimator, params, scoring=None, cv=4,
                 refit=True, verbose=False, population_size=50,
                 gene_mutation_prob=0.1, gene_crossover_prob=0.5,
                 tournament_size=3, generations_number=10, gene_type=None,
                 n_jobs=1, iid=True, error_score='raise',
                 fit_params={}):
        super(GeneticSearchCV, self).__init__(
            estimator=estimator, scoring=scoring, fit_params=fit_params,
            iid=iid, refit=refit, cv=cv, verbose=verbose,
            error_score=error_score)
        self.params = params
        self.population_size = population_size
        self.generations_number = generations_number
        self._individual_evals = {}
        self.gene_mutation_prob = gene_mutation_prob
        self.gene_crossover_prob = gene_crossover_prob
        self.tournament_size = tournament_size
        self.gene_type = gene_type
        self.all_history_, self.all_logbooks_ = [], []
        self._cv_results = None
        self.best_score_ = None
        self.best_params_ = None
        self.score_cache = {}
        self.n_jobs = n_jobs
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, est=clone(self.estimator), fitness=creator.FitnessMax)

    @property
    def possible_params(self):
        """ Used when assuming params is a list. """
        return self.params if isinstance(self.params, list) else [self.params]

    @property
    def cv_results_(self):
        if self._cv_results is None:  # This is to cache the answer until updated
            # Populate output and return
            # If not already fit, returns an empty dictionary
            possible_params = self.possible_params  # Pre-load property for use in this function
            out = defaultdict(list)
            for p, gen in enumerate(self.all_history_):
                # Get individuals and indexes, their list of scores,
                # and additionally the name_values for this set of parameters

                idxs, individuals, each_scores = zip(*[(idx, indiv, np.mean(indiv.fitness.values))
                                                       for idx, indiv in list(gen.genealogy_history.items())
                                                       if indiv.fitness.valid and not np.all(
                        np.isnan(indiv.fitness.values))])

                name_values, _, _ = _get_param_types_maxint(possible_params[p])

                # Add to output
                out['param_index'] += [p] * len(idxs)
                out['index'] += idxs
                out['params'] += [_individual_to_params(indiv, name_values)
                                  for indiv in individuals]
                out['mean_test_score'] += [np.nanmean(scores) for scores in each_scores]
                out['std_test_score'] += [np.nanstd(scores) for scores in each_scores]
                out['min_test_score'] += [np.nanmin(scores) for scores in each_scores]
                out['max_test_score'] += [np.nanmax(scores) for scores in each_scores]
                out['nan_test_score?'] += [np.any(np.isnan(scores)) for scores in each_scores]
            self._cv_results = out

        return self._cv_results

    @property
    def best_index_(self):
        """ Returns the absolute index (not the 'index' column) with the best max_score
            from cv_results_. """
        return np.argmax(self.cv_results_['max_test_score'])

    def fit(self, X, y=None):
        self.best_estimator_ = None
        self.best_mem_score_ = float("-inf")
        self.best_mem_params_ = None
        for possible_params in self.possible_params:
            _check_param_grid(possible_params)
            self._fit(X, y, possible_params)
        if self.refit:
            self.best_estimator_ = clone(self.estimator)
            self.best_estimator_.set_params(**self.best_mem_params_)
            if self.fit_params is not None:
                self.best_estimator_.fit(X, y, **self.fit_params)
            else:
                self.best_estimator_.fit(X, y)

    def _fit(self, X, y, parameter_dict):
        self._cv_results = None  # To indicate to the property the need to update
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)
        n_samples = _num_samples(X)
        X, y = indexable(X, y)

        if y is not None:
            if len(y) != n_samples:
                raise ValueError('Target variable (y) has a different number '
                                 'of samples (%i) than data (X: %i samples)'
                                 % (len(y), n_samples))
        cv = check_cv(self.cv, y=y, classifier=is_classifier(self.estimator))

        toolbox = base.Toolbox()

        name_values, gene_type, maxints = _get_param_types_maxint(parameter_dict)
        if self.gene_type is None:
            self.gene_type = gene_type

        if self.verbose:
            print("Types %s and maxint %s detected" % (self.gene_type, maxints))

        toolbox.register("individual", _initIndividual, creator.Individual, maxints=maxints)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # If n_jobs is an int, greater than 1 or less than 0 (indicating to use as
        # many jobs as possible) then we are going to create a default pool.
        # Windows users need to be warned of this feature as it only works properly
        # on linux. They need to encapsulate their pool in an if __name__ == "__main__"
        # wrapper so that pools are not recursively created when the module is reloaded in each map
        if isinstance(self.n_jobs, int):
            if self.n_jobs > 1 or self.n_jobs < 0:
                from multiprocessing import Pool  # Only imports if needed
                if os.name == 'nt':  # Checks if we are on Windows
                    warnings.warn(("Windows requires Pools to be declared from within "
                                   "an \'if __name__==\"__main__\":\' structure. In this "
                                   "case, n_jobs will accept map functions as well to "
                                   "facilitate custom parallelism. Please check to see "
                                   "that all code is working as expected."))
                pool = Pool(self.n_jobs)
                toolbox.register("map", pool.map)

        # If it's not an int, we are going to pass it as the map directly
        else:
            try:
                toolbox.register("map", self.n_jobs)
            except Exception:
                raise TypeError(
                    "n_jobs must be either an integer or map function. Received: {}".format(type(self.n_jobs)))

        toolbox.register("evaluate", _evalFunction,
                         name_values=name_values, X=X, y=y,
                         scorer=self.scorer_, cv=cv, iid=self.iid, verbose=self.verbose,
                         error_score=self.error_score, fit_params=self.fit_params,
                         score_cache=self.score_cache)

        toolbox.register("mate", _cxIndividual, indpb=self.gene_crossover_prob, gene_type=self.gene_type)

        toolbox.register("mutate", _mutIndividual, indpb=self.gene_mutation_prob, up=maxints)
        toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)

        pop = toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(1)

        # Stats
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.nanmean)
        stats.register("min", np.nanmin)
        stats.register("max", np.nanmax)
        stats.register("std", np.nanstd)

        # History
        hist = tools.History()
        toolbox.decorate("mate", hist.decorator)
        toolbox.decorate("mutate", hist.decorator)
        hist.update(pop)

        if self.verbose:
            print('--- Evolve in {0} possible combinations ---'.format(np.prod(np.array(maxints) + 1)))

        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2,
                                           ngen=self.generations_number, stats=stats,
                                           halloffame=hof, verbose=self.verbose)

        # Save History
        self.all_history_.append(hist)
        self.all_logbooks_.append(logbook)
        current_best_score_ = hof[0].fitness.values[0]
        current_best_params_ = _individual_to_params(hof[0], name_values)
        if self.verbose:
            print("Best individual is: %s\nwith fitness: %s" % (
                current_best_params_, current_best_score_))

        if current_best_score_ > self.best_mem_score_:
            self.best_mem_score_ = current_best_score_
            self.best_mem_params_ = current_best_params_

        # Check memoization, potentially unknown bug
        # assert str(hof[0]) in self.score_cache, "Best individual not stored in score_cache for cv_results_."

        # Close your pools if you made them
        if isinstance(self.n_jobs, int) and (self.n_jobs > 1 or self.n_jobs < 0):
            pool.close()
            pool.join()

        self.best_score_ = current_best_score_
        self.best_params_ = current_best_params_
