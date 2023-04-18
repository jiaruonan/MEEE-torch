import gym
import torch
import numpy as np
import disagreement_tools

class EnvSampler():
    def __init__(self, env, predict_env, max_path_length=1000):
        self.env = env
        self.predict_env = predict_env

        self.path_length = 0
        self.current_state = None
        self.max_path_length = max_path_length
        self.path_rewards = []
        self.sum_reward = 0

    def sample(self, agent, eval_t=False):
        if self.current_state is None:
            self.current_state = self.env.reset()

        cur_state = self.current_state  # (11,)
        
        # Optimism-based exploration: k-action plus uncertainty
        # action = argmax{Q + \lambda V}
        if eval_t == False:
            K = 5
            lambda_arg = 1
            action_list_np = [agent.select_action(self.current_state, eval_t) for _ in range(K)]
            action_stack_np = np.stack(action_list_np, 0)  # [K, action_dim]
            action_stack_torch = torch.from_numpy(action_stack_np).float()
            cur_state_list_np = [cur_state for _ in range(K)]
            cur_state_stack_np = np.stack(cur_state_list_np, 0)
            cur_state_stack_torch = torch.from_numpy(cur_state_stack_np).float()
            
            with torch.no_grad():
                qf1, qf2 = agent.critic(cur_state_stack_torch, action_stack_torch)  # [K, 1]

                inputs = torch.cat((cur_state_stack_torch, action_stack_torch), axis=-1)
                ensemble_model_means, _ = self.predict_env.model.ensemble_model(inputs[None, :, :].repeat([self.predict_env.model.network_size, 1, 1])) # [7, 5, 12]
                ensemble_model_means = ensemble_model_means.numpy()
                rewards_exploration = disagreement_tools.model_disagreement(K, ensemble_model_means)  # [K, 1]

                # next_obs, _, _, _ = self.predict_env.step(cur_state_stack_torch, action_stack_torch)
                # bug: AttributeError: 'StandardScaler' object has no attribute 'mu', 由于方法中初始化顺序不对.

                # choose the first Qnet to estimate Q_exploration
                Q_exploration = qf1.numpy() + lambda_arg * rewards_exploration
                
                action = action_list_np[np.argmax(Q_exploration)]

        else:
            action = agent.select_action(self.current_state, eval_t)

        next_state, reward, terminal, info = self.env.step(action)
        self.path_length += 1
        self.sum_reward += reward

        # TODO: Save the path to the env_pool
        if terminal or self.path_length >= self.max_path_length:
            self.current_state = None
            self.path_length = 0
            self.path_rewards.append(self.sum_reward)
            self.sum_reward = 0
        else:
            self.current_state = next_state

        return cur_state, action, next_state, reward, terminal, info
