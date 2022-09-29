from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        if ('sc2' in self.args.env) or ('group_matching' in self.args.env) or \
            ('particle' in self.args.env) or ('catch' in self.args.env):
            self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        ###0107###
        elif 'meltingpot' in self.args.env:
            self.args.env_args["env_id"]=0
            self.args.env_args["device"]=self.args.device
            self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        else:
            self.env = env_REGISTRY[self.args.env](env_args=self.args.env_args, args=args)

        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info(self.args)

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self, test=False, index=None, constrain_num=None):
        self.batch = self.new_batch()
        self.env.reset(test=test, index=index, constrain_num=constrain_num)
        self.t = 0

    def _get_pre_transition_data(self):
        if self.args.entity_scheme:
            masks = self.env.get_masks()
            if len(masks) == 2:
                obs_mask, entity_mask = masks
                gt_mask = None
            else:
                obs_mask, entity_mask, gt_mask = masks
            pre_transition_data = {
                "entities": [self.env.get_entities()],
                "obs_mask": [obs_mask],
                "entity_mask": [entity_mask],
                "avail_actions": [self.env.get_avail_actions()]
            }
            if gt_mask is not None:
                pre_transition_data["gt_mask"] = gt_mask
        else:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }
        return pre_transition_data

    def run(self, test_mode=False, test_scen=None, index=None, vid_writer=None, **kwargs):
        """
        test_mode: whether to use greedy action selection or sample actions
        test_scen: whether to run on test scenarios. defaults to matching test_mode.
        vid_writer: imageio video writer object
        """
        if test_scen is None:
            test_scen = test_mode
        if self.args.test_unseen:
            constrain_num=self.args.test_map_num if test_mode else self.args.train_map_num
        else:
            constrain_num=None
        self.reset(test=test_scen, index=index, constrain_num=constrain_num)
        if vid_writer is not None:
            vid_writer.append_data(self.env.render())
        else:
            pass
            # self.env.render(**self.args.render_args)
        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        # make sure things like dropout are disabled
        self.mac.eval()

        while not terminated:
            pre_transition_data = self._get_pre_transition_data()

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            if self.args.mac == "comm_mac":
                actions, p_msg, h_msg = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, ret_msg=True)
            else:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            reward, terminated, env_info = self.env.step(actions[0].cpu())
            if vid_writer is not None:
                vid_writer.append_data(self.env.render())
            elif self.args.render:
                self.env.render(**self.args.render_args)
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            if self.args.mac == "comm_mac":
                post_transition_data["self_message"] = p_msg
                post_transition_data["head_message"] = h_msg

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = self._get_pre_transition_data()
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        if self.args.mac == "comm_mac":
            actions, p_msg, h_msg = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, ret_msg=True)
            self.batch.update({"actions": actions, "self_message":p_msg, "head_message": h_msg}, ts=self.t)
        else:
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self.rm = self._log(cur_returns, cur_stats, log_prefix)
        elif not test_mode and self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        rm = np.mean(returns)
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
        return rm 