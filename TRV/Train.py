# --------------------------------------------------------------------------- #
# Hierarchical trainer
# --------------------------------------------------------------------------- #

class HierarchicalTrainer:
    def __init__(self, cfg: V2XConfig):
        self.cfg = cfg
        self.env = V2XEnv(cfg)
        self.rng = np.random.default_rng(cfg.seed + 1)

        manager_in_dim = 11 + cfg.n_cloud_modes  # ...as before, plus the broadcast cloud coordination mode
        worker_in_dim = (3 + cfg.n_rbs + cfg.n_edge_servers
                          + len(VLAPipeline.INTENTS) + cfg.vla_action_dim + 4  # neighbor features
                          + 4    # lat_pos, steering_angle, rel_long_dist, rel_lat_dist
                          + 4    # RT, K, rv, disaster_active
                          + 3    # nearest_obstacle_dist, construction_zone, lane_conflict
                          + cfg.n_cloud_modes)   # broadcast cloud coordination mode
        driver_in_dim = 7        # v_s, mean_neighbor_speed, min_dist, left_space, right_space, lat_pos, steering
        cloud_in_dim = 3         # cloud digital twin forecast: mean_slow_q, mean_edge_load, mean_obstacle_exposure

        self.managers = [ActorCritic(manager_in_dim, 16, 16, cfg.manager_action_dim,
                                      actor_lr=cfg.manager_actor_lr, critic_lr=cfg.manager_critic_lr,
                                      gamma=cfg.gamma, seed=cfg.seed + 10 + c)
                          for c in range(cfg.n_clusters)]
        self.workers = [ActorCritic(worker_in_dim, 24, 24, cfg.worker_action_dim,
                                     actor_lr=cfg.worker_actor_lr, critic_lr=cfg.worker_critic_lr,
                                     gamma=cfg.gamma, seed=cfg.seed + 100 + i)
                         for i in range(cfg.n_vehicles)]
        # The driving action (forward/left/right/reverse) is a genuinely
        # different decision from comms/offload, with its own reward
        # (r_s + r_n + r_dt), so it gets its own actor-critic per vehicle
        # rather than being folded into the worker's action space.
        self.drivers = [ActorCritic(driver_in_dim, 16, 16, len(DRIVE_ACTIONS),
                                     actor_lr=cfg.worker_actor_lr, critic_lr=cfg.worker_critic_lr,
                                     gamma=cfg.gamma, seed=cfg.seed + 200 + i)
                         for i in range(cfg.n_vehicles)]
        # The cloud tier: ONE centralized controller (not per-cluster, unlike
        # managers), acting on an even slower cadence from the CLOUD digital
        # twin's forecast of slow-moving-vehicle/infrastructure state, and
        # broadcasting a discrete coordination mode every manager and worker
        # conditions on -- this is the "centralized control" IoV component
        # sitting above the per-region managers.
        self.cloud = ActorCritic(cloud_in_dim, 16, 16, cfg.n_cloud_modes,
                                  actor_lr=cfg.cloud_actor_lr, critic_lr=cfg.cloud_critic_lr,
                                  gamma=cfg.gamma, seed=cfg.seed + 900)

    def run_episode(self, n_steps: int, train: bool = True, greedy: bool = False):
        cfg = self.cfg
        env = self.env
        env.reset()
        ep_reward, ep_drive_reward = 0.0, 0.0
        ep_stats = {k: [] for k in ("active_reliability", "hazard_rate", "active_frac",
                                     "speed_reward", "safety_penalty", "safety_violation_rate",
                                     "tendency_reward", "mean_min_distance",
                                     "risk_tolerance", "adaptive_capacity", "risk_value", "risk_reward",
                                     "system_risk_tolerance", "system_adaptive_capacity", "system_risk_value",
                                     "disaster_active", "obstacle_penalty", "obstacle_conflict_rate")}
        ep_recon_mse = []

        manager_caches = [None] * cfg.n_clusters
        manager_actions = [0] * cfg.n_clusters
        cluster_reward_accum = np.zeros(cfg.n_clusters)
        pending_worker = None   # (caches, actions, rewards) from the previous step
        pending_driver = None   # (caches, actions, rewards) from the previous step

        cloud_period_steps = cfg.cloud_period * cfg.high_level_period
        cloud_cache, cloud_action = None, 0
        cloud_reward_accum = 0.0

        for t in range(n_steps):
            env.tick_disaster()   # stochastic natural-disaster onset/continuation/end
            env.sync_twins()
            intents, decoded_actions = env.run_vla()   # encode -> process -> decode, per vehicle

            if t % cloud_period_steps == 0:
                cloud_state_now = env._cloud_state()
                if train and t > 0:
                    avg_r = cloud_reward_accum / cloud_period_steps
                    self.cloud.update(cloud_cache, cloud_action, avg_r,
                                       next_x=cloud_state_now, gamma_power=cloud_period_steps)
                cloud_reward_accum = 0.0
                cloud_action, probs, cloud_cache = self.cloud.act(cloud_state_now, self.rng, greedy)
                env.apply_cloud_mode(cloud_action)

            if t % cfg.high_level_period == 0:
                manager_states_now = env._manager_states()
                # TD update for the PREVIOUS decision window: its bootstrap
                # target is this window's start state, discounted by
                # gamma ** high_level_period (a semi-MDP / options-style
                # actor-critic update -- the "reward" is the window's mean
                # per-step reward, i.e. a normalized reward rate).
                if train and t > 0:
                    for c in range(cfg.n_clusters):
                        avg_r = cluster_reward_accum[c] / cfg.high_level_period
                        self.managers[c].update(manager_caches[c], manager_actions[c], avg_r,
                                                 next_x=manager_states_now[c],
                                                 gamma_power=cfg.high_level_period)
                cluster_reward_accum[:] = 0.0

                for c in range(cfg.n_clusters):
                    a, probs, cache = self.managers[c].act(manager_states_now[c], self.rng, greedy)
                    manager_actions[c] = a
                    manager_caches[c] = cache
                env.apply_manager_actions(manager_actions)

            subgoals = np.zeros((cfg.n_vehicles, cfg.n_rbs))
            for i in range(cfg.n_vehicles):
                subgoals[i, env.rb_assignment[i]] = 1.0
            worker_states = env._worker_states(subgoals, intents, decoded_actions)
            driver_states = env._driver_states()

            # TD update for the PREVIOUS step, now that we know its next state.
            if train and pending_worker is not None:
                p_caches, p_actions, p_rewards = pending_worker
                for i in range(cfg.n_vehicles):
                    self.workers[i].update(p_caches[i], p_actions[i], p_rewards[i],
                                            next_x=worker_states[i], gamma_power=1)
            if train and pending_driver is not None:
                d_caches, d_actions, d_rewards = pending_driver
                for i in range(cfg.n_vehicles):
                    self.drivers[i].update(d_caches[i], d_actions[i], d_rewards[i],
                                            next_x=driver_states[i], gamma_power=1)

            worker_actions, worker_caches = [], []
            drive_actions, driver_caches = [], []
            for i in range(cfg.n_vehicles):
                a, probs, cache = self.workers[i].act(worker_states[i], self.rng, greedy)
                worker_actions.append(a)
                worker_caches.append(cache)
                da, dprobs, dcache = self.drivers[i].act(driver_states[i], self.rng, greedy)
                drive_actions.append(da)
                driver_caches.append(dcache)
            worker_actions = np.array(worker_actions)
            drive_actions = np.array(drive_actions)

            worker_reward, driving_reward, info = env.step(worker_actions, intents, drive_actions)
            pending_worker = (worker_caches, worker_actions, worker_reward)
            pending_driver = (driver_caches, drive_actions, driving_reward)

            cluster_reward_accum += np.array(env.cluster_reward(worker_reward))
            cloud_reward_accum += worker_reward.mean()
            ep_reward += worker_reward.mean()
            ep_drive_reward += driving_reward.mean()
            for k in ep_stats:
                ep_stats[k].append(info[k])
            ep_recon_mse.append(env._last_recon_mse)

        if train:
            # Episode-boundary (truncated-horizon) updates: no next state,
            # so the bootstrap target is just the final reward/window mean.
            if pending_worker is not None:
                p_caches, p_actions, p_rewards = pending_worker
                for i in range(cfg.n_vehicles):
                    self.workers[i].update(p_caches[i], p_actions[i], p_rewards[i], next_x=None)
            if pending_driver is not None:
                d_caches, d_actions, d_rewards = pending_driver
                for i in range(cfg.n_vehicles):
                    self.drivers[i].update(d_caches[i], d_actions[i], d_rewards[i], next_x=None)

            leftover = n_steps % cfg.high_level_period or cfg.high_level_period
            for c in range(cfg.n_clusters):
                self.managers[c].update(manager_caches[c], manager_actions[c],
                                         cluster_reward_accum[c] / leftover, next_x=None)

            cloud_leftover = n_steps % cloud_period_steps or cloud_period_steps
            self.cloud.update(cloud_cache, cloud_action, cloud_reward_accum / cloud_leftover, next_x=None)

        result = {k: float(np.mean(v)) for k, v in ep_stats.items()}
        result["reward"] = ep_reward / n_steps
        result["drive_reward"] = ep_drive_reward / n_steps
        result["recon_mse"] = float(np.mean(ep_recon_mse))
        return result

    def train(self, n_episodes: int = 200, steps_per_episode: int = 60, log_every: int = 20):
        history = []
        for ep in range(1, n_episodes + 1):
            stats = self.run_episode(steps_per_episode, train=True)
            history.append(stats)
            if ep % log_every == 0 or ep == 1:
                recent = history[-log_every:]
                avg = {k: float(np.mean([h[k] for h in recent])) for k in stats}
                print(f"episode {ep:4d} | reward {avg['reward']:+.4f} | drive_r {avg['drive_reward']:+.4f} | "
                      f"reliability|active {avg['active_reliability']:.3f} | "
                      f"hazard rate {avg['hazard_rate']:.3f} | "
                      f"active-tx frac {avg['active_frac']:.3f} | "
                      f"r_s {avg['speed_reward']:+.4f} | "
                      f"r_n {avg['safety_penalty']:+.4f} "
                      f"(violation {avg['safety_violation_rate']:.3f}, min_dist {avg['mean_min_distance']:.2f}m) | "
                      f"r_dt {avg['tendency_reward']:+.4f} | "
                      f"RT {avg['risk_tolerance']:.3f} K {avg['adaptive_capacity']:.3f} "
                      f"rv {avg['risk_value']:.3f} risk_r {avg['risk_reward']:+.4f} "
                      f"(system RT {avg['system_risk_tolerance']:.3f} K {avg['system_adaptive_capacity']:.3f} "
                      f"rv {avg['system_risk_value']:.3f}) | "
                      f"disaster frac {avg['disaster_active']:.3f} | "
                      f"obstacle_pen {avg['obstacle_penalty']:+.4f} "
                      f"(conflict {avg['obstacle_conflict_rate']:.3f})")
        return history

    def evaluate(self, n_episodes: int = 10, steps_per_episode: int = 60):
        stats = [self.run_episode(steps_per_episode, train=False, greedy=True)
                 for _ in range(n_episodes)]
        return {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    cfg = V2XConfig()
    trainer = HierarchicalTrainer(cfg)

    print("=== Hierarchical MARL for V2X: cloud + edge digital twins + VLA + adaptive comm interval ===")
    print(f"{cfg.n_vehicles} vehicles / {cfg.n_clusters} clusters / {cfg.n_rbs} RBs / "
          f"{cfg.n_edge_servers} edge servers / comm intervals {cfg.comm_intervals} / "
          f"worker action dim = {cfg.worker_action_dim} / manager action dim = {cfg.manager_action_dim} / "
          f"driving actions = {DRIVE_ACTIONS} / cloud modes = {CLOUD_MODES} "
          f"(every {cfg.cloud_period * cfg.high_level_period} steps)\n")

    trainer.train(n_episodes=200, steps_per_episode=60, log_every=20)

    print("\n=== Greedy evaluation (post-training) ===")
    r = trainer.evaluate(n_episodes=10, steps_per_episode=60)
    print(f"mean reward: {r['reward']:+.4f} | mean drive_reward: {r['drive_reward']:+.4f} | "
          f"reliability|active: {r['active_reliability']:.3f} | "
          f"hazard rate: {r['hazard_rate']:.3f} | VLA recon MSE: {r['recon_mse']:.4f} | "
          f"active-tx frac: {r['active_frac']:.3f}\n"
          f"r_s: {r['speed_reward']:+.4f} | r_n: {r['safety_penalty']:+.4f} "
          f"(violation rate {r['safety_violation_rate']:.3f}, min_dist {r['mean_min_distance']:.2f}m) | "
          f"r_dt: {r['tendency_reward']:+.4f}\n"
          f"RT: {r['risk_tolerance']:.3f} | K: {r['adaptive_capacity']:.3f} | rv: {r['risk_value']:.3f} | "
          f"risk_r: {r['risk_reward']:+.4f}\n"
          f"system RT: {r['system_risk_tolerance']:.3f} | system K: {r['system_adaptive_capacity']:.3f} | "
          f"system rv: {r['system_risk_value']:.3f}\n"
          f"disaster-active frac: {r['disaster_active']:.3f}\n"
          f"obstacle_pen: {r['obstacle_penalty']:+.4f} (conflict rate {r['obstacle_conflict_rate']:.3f})")


if __name__ == "__main__":
    main()
