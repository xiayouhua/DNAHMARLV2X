# --------------------------------------------------------------------------- #
# V2X environment
# --------------------------------------------------------------------------- #
 
class V2XEnv:
    def __init__(self, cfg: V2XConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.n = cfg.n_vehicles
        self.clusters = np.array_split(np.arange(self.n), cfg.n_clusters)
        self.cluster_of = np.zeros(self.n, dtype=int)
        for c, idx in enumerate(self.clusters):
            self.cluster_of[idx] = c
        self.cluster_lang = [np.eye(len(LANG_TASKS))[c % len(LANG_TASKS)]
                              for c in range(cfg.n_clusters)]
 
        self.edges = [EdgeServer(cfg.edge_capacity_cycles) for _ in range(cfg.n_edge_servers)]
        self.vla = VLAPipeline(message_dim=cfg.vla_message_dim, vision_dim=cfg.vision_dim,
                                lang_dim=len(LANG_TASKS), action_dim=cfg.vla_action_dim,
                                seed=cfg.seed + 7)
 
        obstacle_rng = np.random.default_rng(cfg.seed + 99)   # fixed infrastructure, independent of episode rng
        self.obstacles = self._place_obstacles(obstacle_rng)
 
        self.dataset_lib = None
        if cfg.dataset_name and cfg.dataset_root:
            try:
                self.dataset_lib = DrivingDatasetLibrary(cfg.dataset_name, cfg.dataset_root, **cfg.dataset_kwargs)
            except Exception:
                self.dataset_lib = None   # bad name/kwargs -> fall back to the synthetic draw, same as no dataset
 
        # OD-pair path-change trip history (t = steps without a lane
        # change, t_c = steps with one) -- persists ACROSS episodes/resets,
        # since it represents a vehicle's accumulating trip history along a
        # fixed OD pair, not something that resets every simulated episode.
        # Start at t=1, t_c=0 (RT=1, p=0, K=0, rv=1): no observed path
        # changes yet.
        self.trip_no_change = np.ones(self.n)
        self.trip_change = np.zeros(self.n)
        self._last_RT = np.ones(self.n)
        self._last_K = np.zeros(self.n)
        self._last_rv = np.ones(self.n)
 
        # Disaster-conditioned version of the same counts: Risk Tolerance is
        # specifically defined as path changes "during a natural disaster",
        # so this only accumulates on steps where disaster_active is True.
        self.trip_no_change_disaster = np.ones(self.n)
        self.trip_change_disaster = np.zeros(self.n)
        self._last_RT_disaster = np.ones(self.n)
        self._last_K_disaster = np.zeros(self.n)
        self._last_rv_disaster = np.ones(self.n)
 
        # digital twins: one per cluster over [mean_queue, mean_gain] (EDGE
        # twins -- modeled from nearby FAST-moving vehicles), one shared over
        # edge-server queue lengths, and one fleet-wide CLOUD twin over
        # [mean_slow_queue, mean_edge_load, mean_obstacle_exposure] modeled
        # from SLOW-moving vehicles and surrounding infrastructure.
        self.cluster_twins = [DigitalTwin(2, cfg.twin_alpha, cfg.twin_beta,
                                           cfg.twin_obs_noise_std, seed=cfg.seed + 20 + c)
                               for c in range(cfg.n_clusters)]
        self.edge_twin = DigitalTwin(cfg.n_edge_servers, cfg.twin_alpha, cfg.twin_beta,
                                      cfg.twin_obs_noise_std, seed=cfg.seed + 50)
        self.cloud_twin = DigitalTwin(3, cfg.twin_alpha, cfg.twin_beta,
                                       cfg.twin_obs_noise_std, seed=cfg.seed + 60)
 
        self.reset()
 
    def _place_obstacles(self, rng: np.random.Generator):
        """Fixed roadside infrastructure, placed once (not re-randomized
        every episode): WALLS block the RSU's line-of-sight over a
        longitudinal stretch of road; CONSTRUCTION SITES close a lane-width
        lateral band over a longitudinal stretch, forcing a lane change."""
        cfg = self.cfg
        L = cfg.road_length_m
        lat_max = (cfg.n_lanes * cfg.lane_width_m) / 2.0
        starts = rng.uniform(0, L, cfg.n_walls + cfg.n_construction_sites)
        obstacles = []
        for i in range(cfg.n_walls):
            s = starts[i]
            obstacles.append(dict(kind="wall", start=s, end=(s + cfg.wall_length_m) % L))
        for i in range(cfg.n_construction_sites):
            s = starts[cfg.n_walls + i]
            lat_center = rng.uniform(-lat_max + cfg.construction_lane_half_width_m,
                                      lat_max - cfg.construction_lane_half_width_m)
            obstacles.append(dict(kind="construction", start=s,
                                   end=(s + cfg.construction_length_m) % L,
                                   lat_center=lat_center))
        return obstacles
 
    def _pos_in_segment(self, pos: np.ndarray, start: float, end: float) -> np.ndarray:
        """Whether each position falls in [start, end] along the circular
        road (end < start means the segment wraps around 0)."""
        if end >= start:
            return (pos >= start) & (pos <= end)
        return (pos >= start) | (pos <= end)
 
    def _obstacle_features(self):
        """Per vehicle: wall signal-blockage (dB), whether it's inside a
        construction zone, whether it's laterally stuck in that zone's
        closed lane band, and the forward (wrap-aware) distance to the
        nearest obstacle -- what both the driving-quality reward and the
        worker's state need to actually notice and react to obstacles."""
        cfg = self.cfg
        L = cfg.road_length_m
        wall_blockage_db = np.zeros(self.n)
        construction_zone = np.zeros(self.n, dtype=bool)
        lane_conflict = np.zeros(self.n, dtype=bool)
        nearest_dist = np.full(self.n, L / 2.0)
        for obs in self.obstacles:
            in_seg = self._pos_in_segment(self.pos, obs["start"], obs["end"])
            if obs["kind"] == "wall":
                wall_blockage_db = np.where(in_seg, np.maximum(wall_blockage_db, cfg.wall_blockage_db),
                                             wall_blockage_db)
            else:
                construction_zone |= in_seg
                lane_conflict |= in_seg & (np.abs(self.lat_pos - obs["lat_center"])
                                           < cfg.construction_lane_half_width_m)
            forward_dist = (obs["start"] - self.pos) % L
            nearest_dist = np.minimum(nearest_dist, forward_dist)
        return wall_blockage_db, construction_zone, lane_conflict, nearest_dist
 
    def reset(self):
        cfg = self.cfg
        self.pos = self.rng.uniform(0, cfg.road_length_m, self.n)
        if self.dataset_lib is not None:
            self.vel = self.dataset_lib.sample_speeds(self.n, self.rng)   # real empirical speed distribution
        else:
            self.vel = self.rng.uniform(15, 30, self.n)                   # synthetic fallback (unchanged default)
        self.queue = self.rng.uniform(0, cfg.max_queue_bits * 0.3, self.n)
        self.rb_assignment = np.zeros(self.n, dtype=int)
        self.cluster_interval = np.ones(cfg.n_clusters, dtype=int)   # default: transmit every step
        self.t = 0
        lat_max = (cfg.n_lanes * cfg.lane_width_m) / 2.0
        self.lat_pos = self.rng.uniform(-lat_max, lat_max, self.n)   # lateral (cross-road) position, meters
        self.steering_angle = np.zeros(self.n)                       # current steering angle, radians
        self.maneuver_dir = self.rng.choice([-1.0, 1.0], self.n)     # each vehicle's lane-change bias
        self.cloud_mode = 0   # CLOUD_MODES[0] = "normal"
        self.steps_since_delivery = np.zeros(self.n)   # for Transmission Delay (TD)
        self.collision_count_episode = 0               # for Number of Collisions (NC), reset each episode
        self.was_colliding = np.zeros(self.n, dtype=bool)   # previous step's collision state, for event counting
        self.disaster_active = False
        self.disaster_remaining = 0
        self.disaster_edge_id = None
        for e in self.edges:
            e.tasks = []
            e.capacity = e.capacity_base
        self.cluster_twins = [DigitalTwin(2, cfg.twin_alpha, cfg.twin_beta,
                                           cfg.twin_obs_noise_std, seed=cfg.seed + 20 + c)
                               for c in range(cfg.n_clusters)]
        self.edge_twin = DigitalTwin(cfg.n_edge_servers, cfg.twin_alpha, cfg.twin_beta,
                                      cfg.twin_obs_noise_std, seed=cfg.seed + 50)
        self.cloud_twin = DigitalTwin(3, cfg.twin_alpha, cfg.twin_beta,
                                       cfg.twin_obs_noise_std, seed=cfg.seed + 60)
        self._last_obstacle_feats = self._obstacle_features()
        return self._manager_states(), None
 
    def tick_disaster(self):
        """Stochastically start/continue/end a natural-disaster event (e.g.
        storm/flood): while active it adds extra signal blockage, raises the
        local hazard rate, knocks down one edge server's capacity, and
        lowers the safe speed -- the network has to *adapt* to it via the
        disaster_active flag surfaced in both manager and worker state."""
        cfg = self.cfg
        if self.disaster_active:
            self.disaster_remaining -= 1
            if self.disaster_remaining <= 0:
                self.disaster_active = False
                self.disaster_edge_id = None
        elif self.rng.random() < cfg.disaster_prob_per_step:
            self.disaster_active = True
            self.disaster_remaining = cfg.disaster_duration_steps
            self.disaster_edge_id = int(self.rng.integers(0, cfg.n_edge_servers))
 
        for e_id, edge in enumerate(self.edges):
            if self.disaster_active and e_id == self.disaster_edge_id:
                edge.capacity = edge.capacity_base * cfg.disaster_edge_capacity_factor
            else:
                edge.capacity = edge.capacity_base
 
    # ---- physical layer -----------------------------------------------------
 
    def _zone_of(self, pos: np.ndarray) -> np.ndarray:
        """Which environmental-complexity zone a position falls in: 0 =
        open/highway ... n_zones-1 = dense-urban. Complexity here is an
        environment property (blockage, background hazard rate) distinct
        from -- but compounded by -- however many vehicles happen to be
        nearby (see `_neighbor_features`)."""
        frac = pos / self.cfg.road_length_m
        return np.clip((frac * self.cfg.n_zones).astype(int), 0, self.cfg.n_zones - 1)
 
    def _apply_drive_kinematics(self, drive_actions: np.ndarray):
        """Advance velocity, steering angle, and lateral (cross-road)
        position according to the discrete driving action (forward / left /
        right / reverse) -- this is now the sole physical actuator for
        speed and steering; the VLA `maneuver` intent remains a semantic
        signal (message content, worker-state context, hazard reward) but
        no longer drives the vehicle directly. FORWARD accelerates toward
        the local desired speed (zone/disaster/construction-adjusted);
        REVERSE decelerates (and can go slightly negative); LEFT/RIGHT set
        a steering target, which the vehicle's heading eases toward -- a
        simple kinematic single-track (bicycle) lateral-motion model."""
        cfg = self.cfg
        zone = self._zone_of(self.pos)
        _, construction_zone, _, _ = self._last_obstacle_feats
        desired_speed = np.clip(cfg.zone_speed_base - zone * cfg.zone_speed_drop_per_level,
                                 5.0, cfg.zone_speed_base)
        if self.disaster_active:
            desired_speed = desired_speed * cfg.disaster_speed_factor
        desired_speed = np.where(construction_zone, desired_speed * cfg.construction_speed_factor, desired_speed)
 
        accel = np.where(drive_actions == DRIVE_FORWARD, cfg.drive_accel_gain * (desired_speed - self.vel),
                          np.where(drive_actions == DRIVE_REVERSE, -cfg.drive_reverse_decel, 0.0))
        self.vel = np.clip(self.vel + accel * 0.1, cfg.drive_min_speed,
                            cfg.zone_speed_base * cfg.drive_max_speed_factor)
 
        steer_target = np.where(
            drive_actions == DRIVE_LEFT, cfg.max_steering_rad * cfg.drive_turn_steering_frac,
            np.where(drive_actions == DRIVE_RIGHT, -cfg.max_steering_rad * cfg.drive_turn_steering_frac, 0.0))
        noise = self.rng.normal(0, cfg.steering_noise_std, self.n)
        old_steering = self.steering_angle.copy()
        self.steering_angle = np.clip(
            (1 - cfg.steering_smoothing) * self.steering_angle
            + cfg.steering_smoothing * steer_target + noise,
            -cfg.max_steering_rad, cfg.max_steering_rad)
        self._last_steering_delta = self.steering_angle - old_steering
        lat_max = (cfg.n_lanes * cfg.lane_width_m) / 2.0
        self.lat_pos = np.clip(self.lat_pos + self.vel * np.sin(self.steering_angle) * 0.1,
                                -lat_max, lat_max)
 
    def _k_nearest(self, k: int):
        """For every vehicle: 2D (longitudinal-wrap + lateral) Euclidean
        distance, and the speeds, of its k nearest other vehicles (fewer if
        the fleet is smaller than k+1). Used by both the speed reward
        (neighbor speeds) and the safe-distance reward (neighbor distances)."""
        cfg = self.cfg
        L = cfg.road_length_m
        k = min(k, self.n - 1)
        if k <= 0:
            empty = np.zeros((self.n, 0))
            return empty.astype(int), empty, empty
        long_diff = self.pos[None, :] - self.pos[:, None]
        long_diff = (long_diff + L / 2) % L - L / 2
        lat_diff = self.lat_pos[None, :] - self.lat_pos[:, None]
        dist = np.sqrt(long_diff ** 2 + lat_diff ** 2)
        np.fill_diagonal(dist, np.inf)
        idx = np.argsort(dist, axis=1)[:, :k]
        dists = np.take_along_axis(dist, idx, axis=1)
        speeds = self.vel[idx]
        return idx, dists, speeds
 
    def _lateral_space(self):
        """Per vehicle: open lateral space to the left and to the right --
        distance to the road boundary on that side, reduced by any nearby
        (within `lane_change_window_m` longitudinally) vehicle occupying
        that side. Feeds the driving-tendency reward's "which side has more
        room" judgment."""
        cfg = self.cfg
        L = cfg.road_length_m
        lat_max = (cfg.n_lanes * cfg.lane_width_m) / 2.0
        long_diff = self.pos[None, :] - self.pos[:, None]
        long_diff = (long_diff + L / 2) % L - L / 2
        lat_diff = self.lat_pos[None, :] - self.lat_pos[:, None]   # lat_diff[i,j] = lat_pos[j]-lat_pos[i]
        nearby = np.abs(long_diff) < cfg.lane_change_window_m
        np.fill_diagonal(nearby, False)
        left_space = lat_max + self.lat_pos    # distance to the left boundary
        right_space = lat_max - self.lat_pos    # distance to the right boundary
        for i in range(self.n):
            left_mask = nearby[i] & (lat_diff[i] < 0)
            right_mask = nearby[i] & (lat_diff[i] > 0)
            if left_mask.any():
                left_space[i] = min(left_space[i], np.abs(lat_diff[i][left_mask]).min())
            if right_mask.any():
                right_space[i] = min(right_space[i], np.abs(lat_diff[i][right_mask]).min())
        return left_space, right_space
 
    def _raw_neighbor_gaps(self):
        """Per vehicle: signed longitudinal gap (meters) and lateral gap
        (meters) to its single nearest neighbor (by longitudinal distance,
        wrapping around the circular road). Feeds the state-building
        normalized version (`_nearest_neighbor_rel`)."""
        cfg = self.cfg
        L = cfg.road_length_m
        long_gap = np.zeros(self.n)
        lat_gap = np.zeros(self.n)
        for i in range(self.n):
            dx_signed = self.pos - self.pos[i]
            dx_signed = (dx_signed + L / 2) % L - L / 2   # shortest signed gap on a circular road
            dx_signed[i] = np.inf
            j = int(np.argmin(np.abs(dx_signed)))
            long_gap[i] = dx_signed[j]
            lat_gap[i] = self.lat_pos[j] - self.lat_pos[i]
        return long_gap, lat_gap
 
    def _nearest_neighbor_rel(self):
        """Normalized (state-ready) version of `_raw_neighbor_gaps`: signed
        relative LONGITUDINAL distance and relative LATERAL distance to the
        nearest neighbor -- the two "relative position" features requested
        alongside steering angle."""
        cfg = self.cfg
        lat_max = (cfg.n_lanes * cfg.lane_width_m) / 2.0
        long_gap, lat_gap = self._raw_neighbor_gaps()
        rel_long = np.clip(long_gap / cfg.comm_range_m, -2.0, 2.0)
        rel_lat = lat_gap / max(lat_max, 1e-6)
        return rel_long, rel_lat
 
    def _channel_gain_db(self):
        # n_rsus RSUs evenly distributed along the road (not one at the
        # midpoint); each vehicle connects to its NEAREST RSU, distance
        # measured with proper circular wraparound since the road loops.
        L = self.cfg.road_length_m
        n_rsus = self.cfg.n_rsus
        rsu_positions = np.linspace(0, L, n_rsus, endpoint=False) + L / (2 * n_rsus)
        diff = np.abs(self.pos[:, None] - rsu_positions[None, :])
        diff = np.minimum(diff, L - diff)   # circular wraparound
        dist = diff.min(axis=1)             # nearest RSU per vehicle
        pl = path_loss_db(dist)
        zone = self._zone_of(self.pos)
        extra_blockage_db = zone * self.cfg.zone_blockage_db_per_level
        if self.disaster_active:
            extra_blockage_db = extra_blockage_db + self.cfg.disaster_blockage_db
        wall_blockage_db, _, _, _ = self._last_obstacle_feats
        extra_blockage_db = extra_blockage_db + wall_blockage_db
        fading_db = 10 * np.log10(rayleigh_fading(self.rng, self.n) + 1e-6)
        return -pl - extra_blockage_db + fading_db
 
    def _neighbor_features(self, intents: np.ndarray) -> np.ndarray:
        """V2V: driving/warning information vehicles exchange directly with
        each other. Per vehicle: [neighbor_count_norm, mean_relative_speed_norm,
        mean_neighbor_queue_norm, neighbor_hazard_frac] aggregated over all
        other vehicles within `comm_range_m` -- the "surrounding vehicles"
        half of each worker's observation (the other half is its own
        state)."""
        cfg = self.cfg
        L = cfg.road_length_m
        hazard_idx = VLAPipeline.INTENTS.index("hazard_alert")
        feats = np.zeros((self.n, 4))
        for i in range(self.n):
            dx = np.abs(self.pos - self.pos[i])
            dist = np.minimum(dx, L - dx)
            mask = (dist <= cfg.comm_range_m) & (np.arange(self.n) != i)
            idxs = np.where(mask)[0]
            if len(idxs) > 0:
                rel_speed = float(np.mean(np.abs(self.vel[idxs] - self.vel[i]))) / 30.0
                mean_q = float(np.mean(self.queue[idxs])) / cfg.max_queue_bits
                hazard_frac = float(np.mean(intents[idxs] == hazard_idx))
            else:
                rel_speed = mean_q = hazard_frac = 0.0
            feats[i] = [len(idxs) / max(self.n - 1, 1), rel_speed, mean_q, hazard_frac]
        return feats
 
    # ---- state builders -------------------------------------------------------
 
    def _is_fast_vehicle(self) -> np.ndarray:
        """Which vehicles are currently "fast" (modeled by EDGE digital
        twins) vs "slow" (modeled by the CLOUD digital twin)."""
        return self.vel > self.cfg.fast_vehicle_speed_threshold
 
    def sync_twins(self):
        """Sync every digital twin with the current real state (called once
        per environment step, independent of when managers/workers actually
        act on the resulting estimate).
 
        EDGE digital twins (`cluster_twins`) model nearby FAST-moving
        vehicles and their surrounding roads -- each cluster's twin syncs
        from that cluster's fast vehicles specifically (falling back to the
        whole cluster if none are currently fast), serving the
        communication network and sub-objective (RB/interval) generation
        for the hierarchical MARL's managers and workers.
 
        The CLOUD digital twin (`cloud_twin`) models SLOW-moving vehicles
        and surrounding traffic infrastructure/objects at fleet scale --
        it syncs from every slow vehicle's queue state, the current edge-
        server load, and how much of the fleet is currently affected by a
        static obstacle, serving the centralized `CloudController`."""
        self._last_obstacle_feats = self._obstacle_features()   # cache once per step
        gain_db = self._channel_gain_db()
        self._last_gain_db = gain_db
        is_fast = self._is_fast_vehicle()
        for c, idx in enumerate(self.clusters):
            fast_idx = idx[is_fast[idx]]
            use_idx = fast_idx if len(fast_idx) > 0 else idx
            mean_q = self.queue[use_idx].mean() / self.cfg.max_queue_bits
            mean_g = (gain_db[use_idx].mean() + 130) / 40.0
            self.cluster_twins[c].sync(np.array([mean_q, mean_g]))
        edge_loads = np.array([e.queue_len() for e in self.edges], dtype=float) / 10.0
        self.edge_twin.sync(edge_loads)
 
        slow_idx = np.where(~is_fast)[0]
        use_slow = slow_idx if len(slow_idx) > 0 else np.arange(self.n)
        mean_slow_q = self.queue[use_slow].mean() / self.cfg.max_queue_bits
        mean_edge_load = float(edge_loads.mean()) if len(edge_loads) else 0.0
        wall_blockage_db, construction_zone, _, _ = self._last_obstacle_feats
        mean_obstacle_exposure = float(((wall_blockage_db > 0) | construction_zone).mean())
        self.cloud_twin.sync(np.array([mean_slow_q, mean_edge_load, mean_obstacle_exposure]))
 
    def _cloud_state(self) -> np.ndarray:
        """State for the centralized CloudController: the cloud digital
        twin's forecast over its whole coordination window."""
        forecast = self.cloud_twin.predict(self.cfg.cloud_period * self.cfg.high_level_period)
        return np.clip(forecast, 0, 2)
 
    def apply_cloud_mode(self, mode: int):
        """Broadcast the CloudController's chosen global coordination mode
        -- every manager and worker sees it as part of their own state."""
        self.cloud_mode = mode
 
    def _manager_states(self):
        """Per-cluster state for the high-level policy, built from the
        digital twin's FORECAST (proactive) rather than a raw reading, plus
        the cluster's environmental complexity (zone), how dense its
        immediate neighborhood is, whether a natural-disaster event is
        currently active, its cluster's mean adaptive capacity K and risk
        value rv (see `_update_risk_metrics`), how much of the cluster
        is currently affected by a static obstacle (wall/construction
        site), AND the cloud controller's broadcast global coordination
        mode -- the signals a manager needs to decide a resource block, a
        communication interval, and how urgently to adapt to a degraded/
        dangerous/obstructed situation, in the context of what the rest of
        the fleet is doing."""
        states = []
        edge_forecast = self.edge_twin.predict(self.cfg.high_level_period)
        mean_edge_load = float(np.clip(edge_forecast.mean(), 0, 2))
        zone = self._zone_of(self.pos) / max(self.cfg.n_zones - 1, 1)
        neighbor_density = self._neighbor_features(np.zeros(self.n, dtype=int))[:, 0]  # density only, intent-agnostic
        disaster_flag = 1.0 if self.disaster_active else 0.0
        wall_blockage_db, construction_zone, lane_conflict, _ = self._last_obstacle_feats
        obstacle_exposure = ((wall_blockage_db > 0) | construction_zone).astype(float)
        cloud_mode_onehot = np.eye(self.cfg.n_cloud_modes)[self.cloud_mode]
        for c, idx in enumerate(self.clusters):
            fc = self.cluster_twins[c].predict(self.cfg.high_level_period)
            mean_q, mean_g = float(np.clip(fc[0], 0, 2)), float(np.clip(fc[1], 0, 2))
            size = len(idx) / self.n
            hazard_frac = self._last_hazard_frac[c] if hasattr(self, "_last_hazard_frac") else 0.0
            mean_zone = float(zone[idx].mean())
            mean_density = float(neighbor_density[idx].mean())
            mean_K = float(self._last_K[idx].mean())
            mean_rv = float(self._last_rv[idx].mean())
            mean_obstacle = float(obstacle_exposure[idx].mean())
            states.append(np.concatenate([[mean_q, mean_g, size, mean_edge_load, hazard_frac,
                                            mean_zone, mean_density, disaster_flag, mean_K, mean_rv,
                                            mean_obstacle], cloud_mode_onehot]))
        return states
 
    def _worker_states(self, subgoals: np.ndarray, intents: np.ndarray, decoded_actions: np.ndarray):
        cfg = self.cfg
        gain_db = self._last_gain_db
        edge_forecast = np.clip(self.edge_twin.predict(1), 0, 2)  # twin estimate, not true queue
        neighbor_feats = self._neighbor_features(intents)
        rel_long, rel_lat = self._nearest_neighbor_rel()
        lat_max = (cfg.n_lanes * cfg.lane_width_m) / 2.0
        disaster_flag = 1.0 if self.disaster_active else 0.0
        _, construction_zone, lane_conflict, nearest_obs_dist = self._last_obstacle_feats
        nearest_obs_norm = np.clip(nearest_obs_dist / cfg.comm_range_m, 0, 2)
        cloud_mode_onehot = np.eye(cfg.n_cloud_modes)[self.cloud_mode]
        states = []
        for i in range(self.n):
            q = self.queue[i] / self.cfg.max_queue_bits
            g = (gain_db[i] + 130) / 40.0
            v = self.vel[i] / 30.0
            lat = self.lat_pos[i] / max(lat_max, 1e-6)
            steer = self.steering_angle[i] / cfg.max_steering_rad
            driving_feats = np.array([lat, steer, rel_long[i], rel_lat[i]])
            risk_feats = np.array([self._last_RT[i], self._last_K[i], self._last_rv[i], disaster_flag])
            obstacle_feats = np.array([nearest_obs_norm[i], float(construction_zone[i]),
                                        float(lane_conflict[i])])
            intent_onehot = np.eye(len(VLAPipeline.INTENTS))[intents[i]]
            states.append(np.concatenate([[q, g, v], subgoals[i], edge_forecast,
                                           intent_onehot, decoded_actions[i], neighbor_feats[i],
                                           driving_feats, risk_feats, obstacle_feats, cloud_mode_onehot]))
        return states
 
    # ---- perception (VLA: encode -> process -> decode) -------------------------
 
    def run_vla(self):
        """For every vehicle: build the V2X message it is transmitting this
        step, run it through the VLA pipeline (encode -> process -> decode),
        and collect (intents, decoded_action hints, reconstruction error,
        per-cluster hazard fraction) for the rest of the environment/agents
        to use."""
        cfg = self.cfg
        gain_db = self._last_gain_db
        zone = self._zone_of(self.pos)
        _, construction_zone, _, _ = self._last_obstacle_feats
        hazard_mult = 1.0 + zone * cfg.zone_hazard_multiplier_per_level
        if self.disaster_active:
            hazard_mult = hazard_mult * cfg.disaster_hazard_multiplier
        hazard_mult = np.where(construction_zone, hazard_mult * cfg.construction_hazard_multiplier, hazard_mult)
        hazard_prob_local = np.clip(cfg.hazard_prob * hazard_mult, 0, 1)
        hazard_flag = (self.rng.random(self.n) < hazard_prob_local).astype(float)
        self._last_hazard_flag = hazard_flag
 
        intents = np.zeros(self.n, dtype=int)
        decoded_actions = np.zeros((self.n, cfg.vla_action_dim))
        recon_mse = np.zeros(self.n)
        cluster_hazard_count = np.zeros(cfg.n_clusters)
 
        for i in range(self.n):
            # ---- the information this vehicle is actually transmitting: a
            # compact V2X status message (position, speed, lateral offset,
            # steering angle, queue backlog, raw hazard sensor reading,
            # channel quality, current RB) ----
            message = np.array([
                self.pos[i] / cfg.road_length_m,
                self.vel[i] / 30.0,
                self.lat_pos[i] / max((cfg.n_lanes * cfg.lane_width_m) / 2.0, 1e-6),
                self.steering_angle[i] / cfg.max_steering_rad,
                self.queue[i] / cfg.max_queue_bits,
                hazard_flag[i],
                (gain_db[i] + 130) / 40.0,
                self.rb_assignment[i] / max(cfg.n_rbs - 1, 1),
            ])
            vision = self.rng.normal(0, 0.3, cfg.vision_dim)
            vision[0] += hazard_flag[i]                       # informative hazard channel
            lang = self.cluster_lang[self.cluster_of[i]]       # app context for this vehicle's cluster
 
            intent_idx, probs, recon_message, decoded_action = self.vla.run(
                message, vision, lang, self.rng)
 
            intents[i] = intent_idx
            decoded_actions[i] = decoded_action
            recon_mse[i] = float(np.mean((message - recon_message) ** 2))
            if intent_idx == VLAPipeline.INTENTS.index("hazard_alert"):
                cluster_hazard_count[self.cluster_of[i]] += 1
 
        self._last_hazard_frac = [cluster_hazard_count[c] / len(idx)
                                   for c, idx in enumerate(self.clusters)]
        self._last_recon_mse = float(recon_mse.mean())
        return intents, decoded_actions
 
    # ---- dynamics -----------------------------------------------------------
 
    def apply_manager_actions(self, combined_actions: list[int]):
        """combined_actions[c] encodes (rb_choice, interval_choice) for
        cluster c via combined = rb_idx * n_intervals + interval_idx."""
        n_intervals = len(self.cfg.comm_intervals)
        for c, idx in enumerate(self.clusters):
            rb_idx = combined_actions[c] // n_intervals
            interval_idx = combined_actions[c] % n_intervals
            self.rb_assignment[idx] = rb_idx
            self.cluster_interval[c] = self.cfg.comm_intervals[interval_idx]
 
    def driving_reward(self, drive_actions: np.ndarray):
        """The user-specified driving reward, evaluated after this step's
        motion:
          r_s  = (v_s - mean(v_1..v_k)) * p_s
          r_n  = p_n * sum(1 / (d_i - r))                  [p_n < 0]
          r_dt = -p_dt*(rz-15) if left, +p_dt*(rz-15) if right, else 0
          r    = r_s + r_n + r_dt
        `rz` is the "driving factor": 15 plus the (clipped) difference
        between right-side and left-side open space in meters, so rz > 15
        means the right has more room and rz < 15 means the left does --
        turning toward the side that actually has more room earns positive
        reward, turning the wrong way is penalized. Must be called AFTER
        this step's motion update (`_apply_drive_kinematics`), since it
        evaluates the resulting speed/position, not the pre-motion state.
        Returns (total_reward, diagnostics dict, safe-distance violation
        mask) -- the violation mask is also needed by `step()` for the
        risk-value calculation."""
        cfg = self.cfg
        idx, dists, speeds = self._k_nearest(cfg.n_surrounding_vehicles)
 
        if dists.shape[1] > 0:
            mean_neighbor_speed = speeds.mean(axis=1)
            d_eff = np.maximum(dists, cfg.safety_radius_m + 1e-2)   # guard d_i -> r+ (formula assumes d_i > r)
            r_n = cfg.safety_distance_penalty_factor * np.sum(1.0 / (d_eff - cfg.safety_radius_m), axis=1)
            min_dist = dists.min(axis=1)
        else:
            mean_neighbor_speed = self.vel.copy()
            r_n = np.zeros(self.n)
            min_dist = np.full(self.n, cfg.road_length_m)
        r_s = (self.vel - mean_neighbor_speed) * cfg.speed_reward_factor
        safety_violation = min_dist < (cfg.safety_radius_m + cfg.safety_violation_margin_m)
        collision = min_dist < cfg.safety_radius_m   # true physical overlap
        # Number of Collisions (NC) counts distinct EVENTS -- the transition
        # into a collision -- not every step a sustained near-overlap
        # continues, which would otherwise wildly overcount a single stuck
        # encounter as dozens of "collisions".
        new_collision_event = collision & (~self.was_colliding)
        self.was_colliding = collision
 
        left_space, right_space = self._lateral_space()
        pivot = cfg.driving_tendency_neutral
        rz = pivot + np.clip(right_space - left_space, -pivot, pivot)
        r_dt = np.where(drive_actions == DRIVE_LEFT, -cfg.driving_tendency_factor * (rz - pivot),
                         np.where(drive_actions == DRIVE_RIGHT, cfg.driving_tendency_factor * (rz - pivot), 0.0))
 
        total = r_s + r_n + r_dt
        info = dict(speed_reward=float(r_s.mean()), safety_penalty=float(r_n.mean()),
                    tendency_reward=float(r_dt.mean()), mean_min_distance=float(min_dist.mean()),
                    safety_violation_rate=float(np.mean(safety_violation)),
                    n_collisions=int(new_collision_event.sum()))
        return total, info, safety_violation
 
    @staticmethod
    def _risk_from_counts(t: np.ndarray, t_c: np.ndarray):
        """RT, K, rv from raw (t, t_c) path-change counts:
          RT = t / (t + t_c)
          p  = t_c / (t + t_c)            ( = 1 - RT )
          K  = p / (1 - p)   if 0 < p < 0.5
          K  = 1             if 0.5 <= p < 1
          rv = 1 / (RT + K)
        """
        total = t + t_c
        RT = t / total
        p = t_c / total
        K = np.where(p < 0.5, p / np.maximum(1.0 - p, 1e-9), 1.0)
        rv = 1.0 / np.maximum(RT + K, 1e-9)
        return RT, K, rv
 
    def _update_risk_metrics(self, drive_actions: np.ndarray):
        """Update every vehicle's OD-pair path-change trip history from
        this step's driving action (a LEFT/RIGHT is a "path change",
        FORWARD/REVERSE is not), then derive risk tolerance RT, adaptive
        capacity K, and risk value rv -- both over ALL steps, and over
        DISASTER steps only (RT is specifically defined as path changes
        "during a natural disaster", so that version is the one meant for
        direct comparison purposes). Also pools the whole fleet's counts
        into system-wide aggregates of both -- K is meant to characterize
        a *transportation system's* adaptive capacity, not just one
        vehicle's."""
        is_change = (drive_actions == DRIVE_LEFT) | (drive_actions == DRIVE_RIGHT)
        self.trip_change = self.trip_change + is_change.astype(float)
        self.trip_no_change = self.trip_no_change + (~is_change).astype(float)
 
        RT, K, rv = self._risk_from_counts(self.trip_no_change, self.trip_change)
        self._last_RT, self._last_K, self._last_rv = RT, K, rv
 
        sys_RT, sys_K, sys_rv = self._risk_from_counts(
            np.array([self.trip_no_change.sum()]), np.array([self.trip_change.sum()]))
        self._last_system_RT = float(sys_RT[0])
        self._last_system_K = float(sys_K[0])
        self._last_system_rv = float(sys_rv[0])
 
        if self.disaster_active:
            self.trip_change_disaster = self.trip_change_disaster + is_change.astype(float)
            self.trip_no_change_disaster = self.trip_no_change_disaster + (~is_change).astype(float)
        RT_d, K_d, rv_d = self._risk_from_counts(self.trip_no_change_disaster, self.trip_change_disaster)
        self._last_RT_disaster, self._last_K_disaster, self._last_rv_disaster = RT_d, K_d, rv_d
 
        sys_RT_d, sys_K_d, sys_rv_d = self._risk_from_counts(
            np.array([self.trip_no_change_disaster.sum()]), np.array([self.trip_change_disaster.sum()]))
        self._last_system_RT_disaster = float(sys_RT_d[0])
        self._last_system_K_disaster = float(sys_K_d[0])
        self._last_system_rv_disaster = float(sys_rv_d[0])
        return RT, K, rv
 
    def _driver_states(self):
        """Compact per-vehicle state for the driving-action policy: own
        speed, mean neighbor speed, nearest-neighbor distance, left/right
        open space, lateral position, and steering angle."""
        cfg = self.cfg
        lat_max = (cfg.n_lanes * cfg.lane_width_m) / 2.0
        idx, dists, speeds = self._k_nearest(cfg.n_surrounding_vehicles)
        if dists.shape[1] > 0:
            mean_neighbor_speed = speeds.mean(axis=1)
            min_dist = dists.min(axis=1)
        else:
            mean_neighbor_speed = self.vel.copy()
            min_dist = np.full(self.n, cfg.road_length_m)
        left_space, right_space = self._lateral_space()
        states = []
        for i in range(self.n):
            v_s = self.vel[i] / 30.0
            v_mean = mean_neighbor_speed[i] / 30.0
            d_min = np.clip(min_dist[i] / 20.0, 0, 3)
            lsp = np.clip(left_space[i] / max(lat_max, 1e-6), 0, 2)
            rsp = np.clip(right_space[i] / max(lat_max, 1e-6), 0, 2)
            lat = self.lat_pos[i] / max(lat_max, 1e-6)
            steer = self.steering_angle[i] / cfg.max_steering_rad
            states.append(np.array([v_s, v_mean, d_min, lsp, rsp, lat, steer]))
        return states
 
    def step(self, combined_actions: np.ndarray, intents: np.ndarray, drive_actions: np.ndarray):
        """combined_actions[i] encodes (power_level, offload_choice) for
        vehicle i via combined = power_idx * n_offload_choices + offload_idx.
        drive_actions[i] is one of DRIVE_FORWARD/LEFT/RIGHT/REVERSE."""
        cfg = self.cfg
        n_off = cfg.n_offload_choices
        power_idx = combined_actions // n_off
        offload_idx = combined_actions % n_off
 
        # ---- adaptive communication interval: a vehicle only gets a real
        # transmission opportunity on steps aligned with its cluster's
        # currently-chosen interval; off-steps it stays silent (no airtime,
        # no interference contributed to others, no energy spent) but also
        # earns no throughput and cannot offload to an edge server that step. ----
        interval_per_vehicle = self.cluster_interval[self.cluster_of]
        active_mask = (self.t % interval_per_vehicle) == 0
        self.t += 1
 
        # ---- communication (V2I: vehicle <-> RSU/infrastructure link) ----
        p_levels_dbm = np.linspace(0, cfg.p_max_dbm, cfg.n_power_levels)
        tx_power_dbm = p_levels_dbm[power_idx]
        tx_power_mw = 10 ** (tx_power_dbm / 10)
        tx_power_mw = np.where(active_mask, tx_power_mw, 0.0)   # silent vehicles emit nothing
 
        gain_db = self._last_gain_db
        gain_lin = 10 ** (gain_db / 10)
        noise_mw = 10 ** (cfg.noise_dbm / 10)
        rx_power_mw = tx_power_mw * gain_lin
 
        sinr_db = np.zeros(self.n)
        throughput_bits = np.zeros(self.n)
        for i in range(self.n):
            same_rb = np.where((self.rb_assignment == self.rb_assignment[i]) &
                                (np.arange(self.n) != i))[0]
            interference_mw = rx_power_mw[same_rb].sum() if len(same_rb) else 0.0
            sinr_lin = rx_power_mw[i] / (noise_mw + interference_mw + 1e-15)
            sinr_db[i] = 10 * np.log10(sinr_lin + 1e-15)
            rb_share_hz = cfg.bandwidth_hz / cfg.n_rbs
            throughput_bits[i] = rb_share_hz * np.log2(1 + sinr_lin)
 
        served = np.minimum(self.queue, throughput_bits)
        self.queue = self.queue - served
        arrivals = self.rng.exponential(cfg.arrival_rate_bits, self.n)
        self.queue = np.clip(self.queue + arrivals, 0, cfg.max_queue_bits)
 
        reliability_ok = (sinr_db >= cfg.sinr_threshold_db).astype(float)
        energy_cost = tx_power_mw / (10 ** (cfg.p_max_dbm / 10))
 
        comm_reward_raw = (served / cfg.max_queue_bits) - 0.05 * energy_cost - 0.5 * (1 - reliability_ok)
        comm_reward = active_mask * comm_reward_raw   # silence is a choice, not a failed transmission
 
        # ---- Transmission Delay (TD): steps since this vehicle last had a
        # successful (active + reliable) delivery, i.e. how far the actual
        # arrival time has drifted past the "expected" immediate (every-
        # step) arrival. Reset to 0 on a successful delivery this step. ----
        delivered = active_mask & reliability_ok.astype(bool)
        transmission_delay = self.steps_since_delivery.copy()   # delay observed going into this step
        self.steps_since_delivery = np.where(delivered, 0.0, self.steps_since_delivery + 1.0)
 
        # ---- edge / local compute offloading (V2I: vehicle <-> edge server) ----
        has_task = self.rng.random(self.n) < cfg.task_prob
        task_cycles = self.rng.exponential(cfg.task_cycles_mean, self.n) * has_task
        task_reward = np.zeros(self.n)
        for i in range(self.n):
            if not has_task[i]:
                continue
            if offload_idx[i] == 0:  # LOCAL
                if task_cycles[i] <= cfg.local_capacity_cycles:
                    task_reward[i] += cfg.task_success_bonus
                else:
                    task_reward[i] -= cfg.task_fail_penalty
            else:
                edge_id = offload_idx[i] - 1
                if reliability_ok[i]:
                    self.edges[edge_id].enqueue(task_cycles[i], cfg.task_deadline_steps, i)
                else:
                    task_reward[i] -= cfg.task_fail_penalty  # offload request lost in transit
 
        for edge in self.edges:
            completed, failed = edge.step()
            for i in completed:
                task_reward[i] += cfg.task_success_bonus
            for i in failed:
                task_reward[i] -= cfg.task_fail_penalty
 
        # ---- VLA-driven hazard-priority reward shaping ----
        hazard_idx = VLAPipeline.INTENTS.index("hazard_alert")
        is_hazard_intent = (intents == hazard_idx)
        intent_reward = np.where(
            is_hazard_intent,
            np.where(reliability_ok.astype(bool), cfg.hazard_success_bonus, -cfg.hazard_fail_penalty),
            0.0,
        )
 
        worker_reward = comm_reward + task_reward + intent_reward
 
        self.pos = (self.pos + self.vel * 0.1) % cfg.road_length_m
        wall_blockage_db, construction_zone, lane_conflict, nearest_obs_dist = self._obstacle_features()
        self._apply_drive_kinematics(drive_actions)
 
        # ---- driving reward: r_s + r_n + r_dt, evaluated now that this
        # step's motion has happened ----
        drive_total, drive_info, safety_violation = self.driving_reward(drive_actions)
        self.collision_count_episode += drive_info["n_collisions"]
 
        # ---- obstacle penalty: stuck in a construction site's closed lane
        # band -- pushes the vehicle to actually change lanes rather than
        # just tolerate the obstacle ----
        obstacle_penalty = np.where(lane_conflict, -cfg.obstacle_penalty_weight, 0.0)
 
        # ---- risk tolerance RT, adaptive capacity K, and risk value rv:
        # derived from each vehicle's OD-pair path-change trip history
        # (t = steps without a lane change, t_c = steps with one):
        #   RT = t/(t+t_c), p = t_c/(t+t_c) = 1-RT
        #   K  = p/(1-p) if 0<p<0.5, else 1 if 0.5<=p<1
        #   rv = 1/(RT+K)
        # rv directly drives risk_reward; unlike the driving reward, this is
        # scoped to the comm/offload agent since it reflects a persistent
        # trait of the vehicle's trip, not this step's specific action. ----
        RT, K, rv = self._update_risk_metrics(drive_actions)
        risk_reward = -rv * cfg.risk_penalty_weight
 
        # The comm/offload agent's reward stays scoped to comms, compute
        # offload, hazard-message delivery, OD-pair risk value, and
        # obstacles; the driving action (forward/left/right/reverse) is a
        # separate decision with its own reward (drive_total) and its own
        # actor-critic (see HierarchicalTrainer.drivers) -- they are
        # trained side by side, not summed into one scalar.
        worker_reward = worker_reward + risk_reward + obstacle_penalty
 
        info = dict(sinr_db=sinr_db, throughput_bits=throughput_bits,
                    reliability=reliability_ok, queue=self.queue.copy(),
                    edge_queue_lens=[e.queue_len() for e in self.edges],
                    hazard_rate=float(is_hazard_intent.mean()),
                    active_frac=float(active_mask.mean()),
                    active_reliability=float(reliability_ok[active_mask].mean())
                                        if active_mask.any() else 0.0,
                    speed_reward=drive_info["speed_reward"],
                    safety_penalty=drive_info["safety_penalty"],
                    safety_violation_rate=drive_info["safety_violation_rate"],
                    tendency_reward=drive_info["tendency_reward"],
                    mean_min_distance=drive_info["mean_min_distance"],
                    risk_tolerance=float(RT.mean()),
                    adaptive_capacity=float(K.mean()),
                    risk_value=float(rv.mean()),
                    risk_reward=float(risk_reward.mean()),
                    system_risk_tolerance=self._last_system_RT,
                    system_adaptive_capacity=self._last_system_K,
                    system_risk_value=self._last_system_rv,
                    # disaster-conditioned versions: RT is specifically defined
                    # as path changes "during a natural disaster" -- this is
                    # the version to use for the four-metric comparison.
                    risk_tolerance_disaster=float(self._last_RT_disaster.mean()),
                    adaptive_capacity_disaster=float(self._last_K_disaster.mean()),
                    risk_value_disaster=float(self._last_rv_disaster.mean()),
                    system_risk_tolerance_disaster=self._last_system_RT_disaster,
                    disaster_active=float(self.disaster_active),
                    obstacle_penalty=float(obstacle_penalty.mean()),
                    obstacle_conflict_rate=float(lane_conflict.mean()),
                    # ---- the four comparison metrics ----
                    transmission_delay=float(transmission_delay.mean()),         # TD
                    reward_value=float(drive_total.mean()),                     # RV = r_s + r_n + r_dt
                    n_collisions=drive_info["n_collisions"],                    # NC (this step, event-based)
                    collisions_episode_total=self.collision_count_episode)      # NC (cumulative this episode)
        return worker_reward, drive_total, info
 
    def cluster_reward(self, worker_reward: np.ndarray) -> list[float]:
        return [worker_reward[idx].mean() for idx in self.clusters]
 
