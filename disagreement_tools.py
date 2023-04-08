import numpy as np
import torch

def KL_divergence(num_models, batch_size, model_idxes, batch_idxes, model_means, model_vars, ensemble_model_means, ensemble_model_vars):

    model_index = np.arange(0, num_models).tolist()
    model_inds_rest = np.array([model_index[:i] + model_index[i + 1:] for i in model_idxes])

    model_means_rest = np.array(
        [ensemble_model_means[model_inds_rest[:, n], batch_idxes, 1:] for n in range(num_models - 1)])
    model_vars_rest = np.array(
        [ensemble_model_vars[model_inds_rest[:, n], batch_idxes, 1:] for n in range(num_models - 1)])
    model_means_rest = np.mean(model_means_rest, axis=0)
    model_vars_rest = np.mean(model_vars_rest + model_means_rest ** 2, axis=0) - model_means_rest ** 2

    KLdivergence_list = np.log(model_vars_rest / model_vars[:, 1:]) - 0.5 + (
            model_vars[:, 1:] + np.square(model_means[:, 1:] - model_means_rest)) / (2 * model_vars_rest)
    KL_result = np.sum(KLdivergence_list, axis=-1)
    rewards_exploration = np.reshape(KL_result, [batch_size, 1])

    return rewards_exploration


def model_disagreement(batch_size, ensemble_model_means):

    rewards_exploration = np.var(ensemble_model_means, axis=0)
    rewards_exploration = np.mean(rewards_exploration, axis=-1)

    rewards_exploration = np.reshape(rewards_exploration, [batch_size, 1])

    return rewards_exploration
