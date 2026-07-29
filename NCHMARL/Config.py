# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
 
LANG_TASKS = ["intersection_monitoring", "platooning", "vru_detection"]
 
# The cloud controller's discrete global coordination mode, broadcast to
# every manager and worker (see V2XConfig.n_cloud_modes and point 11 of
# the module docstring).
CLOUD_MODES = ["normal", "cautious", "aggressive"]
 
# Discrete driving action space: forward / left turn / right turn / reverse
DRIVE_ACTIONS = ["forward", "left", "right", "reverse"]
DRIVE_FORWARD, DRIVE_LEFT, DRIVE_RIGHT, DRIVE_REVERSE = range(4)
 
 
@dataclass
class V2XConfig:
    n_vehicles: int = 12
    n_clusters: int = 3
    n_rbs: int = 4
    n_power_levels: int = 5
    p_max_dbm: float = 30.0
    bandwidth_hz: float = 1.0e6
    noise_dbm: float = -95.0
    sinr_threshold_db: float = 0.0
    n_rsus: int = 8                    # distributed evenly along the road, not a single one at the midpoint
    max_queue_bits: float = 5.0e5
    arrival_rate_bits: float = 6.0e4
    road_length_m: float = 16000.0
    high_level_period: int = 5
    seed: int = 0
 
    # --- edge computing ---
    n_edge_servers: int = 2
    edge_capacity_cycles: float = 3.0e8      # cycles/step per edge server
    local_capacity_cycles: float = 5.0e7     # on-board compute, cycles/step
    task_prob: float = 0.6                   # chance a vehicle has a task this step
    task_cycles_mean: float = 8.0e7
    task_deadline_steps: int = 3
    task_success_bonus: float = 0.3
    task_fail_penalty: float = 0.3
 
    # --- digital twin ---
    twin_alpha: float = 0.35
    twin_beta: float = 0.25
    twin_obs_noise_std: float = 0.02
 
    # --- OD-pair path-change risk model ---
    # Each vehicle is treated as one continuous trip along a fixed OD pair
    # (start -> loop). Its cumulative count of steps WITHOUT a path (lane)
    # change (t) vs WITH one (t_c) defines:
    #   RT = t / (t + t_c)                     risk tolerance
    #   p  = t_c / (t + t_c)  ( = 1 - RT )      empirical path-change probability
    #   K  = p / (1 - p)  if 0 < p < 0.5        adaptive capacity
    #   K  = 1            if 0.5 <= p < 1
    #   rv = 1 / (RT + K)                       risk value
    risk_penalty_weight: float = 0.3  # scales risk_reward = -rv * this
 
    # --- natural-disaster events (e.g. storm/flood): temporary, severe,
    # area-wide degradation the network must adapt to ---
    disaster_prob_per_step: float = 0.01
    disaster_duration_steps: int = 15
    disaster_blockage_db: float = 10.0
    disaster_hazard_multiplier: float = 3.0
    disaster_edge_capacity_factor: float = 0.2   # affected edge server's capacity during a disaster
    disaster_speed_factor: float = 0.6           # lower safe speed during a disaster
 
    # --- static obstacles: walls (block RSU line-of-sight) and construction
    # sites (close a lane, need a lane change to clear) ---
    n_walls: int = 2
    wall_length_m: float = 40.0
    wall_blockage_db: float = 15.0
    n_construction_sites: int = 2
    construction_length_m: float = 60.0
    construction_lane_half_width_m: float = 1.2   # half-width of the closed lateral band
    construction_speed_factor: float = 0.5
    construction_hazard_multiplier: float = 2.0
    obstacle_penalty_weight: float = 0.4          # penalty while stuck in a closed lane band
 
    # --- optional real-world dataset grounding (KITTI / nuScenes / ONCE) ---
    # If set, vehicle initial speeds are bootstrap-sampled from the named
    # dataset's real empirical speed distribution instead of a synthetic
    # uniform(15, 30) draw. Leave dataset_name=None (the default) for
    # unchanged behavior -- see the DrivingDatasetLibrary docstring for
    # why the actual dataset files aren't (and can't be) bundled here.
    dataset_name: str | None = None       # "kitti" | "nuscenes" | "once" | None
    dataset_root: str | None = None       # local path to your own copy of the dataset
    dataset_kwargs: dict = field(default_factory=dict)   # e.g. split="v1.0-mini", seq_id="000000"
 
    # --- VLA / hazard scenario ---
    hazard_prob: float = 0.05
    hazard_success_bonus: float = 0.25
    hazard_fail_penalty: float = 0.35
    vision_dim: int = 8
    vla_message_dim: int = 8     # size of the V2X status message each vehicle transmits
    vla_action_dim: int = 2      # size of the decoded action hint handed to workers
 
    # --- surrounding-vehicle awareness ---
    comm_range_m: float = 150.0   # radius within which a vehicle "sees" neighbors
 
    # --- lateral driving dynamics (lane position + steering) ---
    n_lanes: int = 3
    lane_width_m: float = 3.5
    max_steering_rad: float = 0.15      # ~8.6 degrees, small-angle kinematic model
    steering_smoothing: float = 0.3     # how fast steering angle moves toward its target
    steering_noise_std: float = 0.02    # lane-keeping jitter
 
    # --- discrete driving action space: forward / left / right / reverse ---
    drive_accel_gain: float = 0.5        # how hard "forward" accelerates toward the desired speed
    drive_reverse_decel: float = 3.0     # m/s^2-ish deceleration applied by "reverse"
    drive_turn_steering_frac: float = 1.0  # fraction of max_steering_rad used as left/right's target
    drive_min_speed: float = -8.0        # allow limited reverse speed
    drive_max_speed_factor: float = 1.15  # vs zone_speed_base
 
    # --- driving reward: r = r_s (speed) + r_n (safe distance) + r_dt (tendency) ---
    n_surrounding_vehicles: int = 10     # k nearest other vehicles considered by r_s / r_n
    speed_reward_factor: float = 0.05          # p_s
    safety_distance_penalty_factor: float = -0.02   # p_n (negative, per spec)
    safety_radius_m: float = 4.0               # r -- vehicle safety-bubble radius
    safety_violation_margin_m: float = 1.0     # extra margin beyond r counted as a diagnostic "violation"
    driving_tendency_factor: float = 0.01      # p_dt
    lane_change_window_m: float = 25.0         # forward/back window used to judge left/right space
    driving_tendency_neutral: float = 15.0     # the "15" pivot in the tendency formula
 
    # --- environmental-complexity speed context (still shapes the "desired
    # speed" the "forward" action accelerates toward, even though the r_s
    # reward itself is neighbor-relative, not limit-relative) ---
    zone_speed_base: float = 30.0            # m/s speed limit in zone 0 (open/highway)
    zone_speed_drop_per_level: float = 6.0   # m/s lower speed limit per zone level
 
 
    # --- environmental complexity (spatial zones along the road) ---
    # zone 0 = open/highway, 1 = suburban, 2 = dense-urban: higher zones add
    # extra signal blockage and raise the local hazard rate.
    n_zones: int = 3
    zone_blockage_db_per_level: float = 4.0
    zone_hazard_multiplier_per_level: float = 1.0   # hazard_prob *= (1 + level*this)
 
    # --- adaptive communication interval (congestion / duty-cycle control) ---
    # Managers pick how often (in steps) their cluster's vehicles get a
    # transmission opportunity -- shorter intervals cut latency/staleness
    # but add airtime and mutual interference; longer intervals save both
    # at the risk of missing time-critical (hazard) windows.
    comm_intervals: tuple = (1, 2, 4)
 
    # --- IoV architecture: edge vs. cloud digital twins ---
    # Edge digital twins model nearby FAST-moving vehicles and their
    # surrounding roads/objects -- they're what `cluster_twins` (per-
    # region communication state) sync from. Cloud digital twins model
    # SLOW-moving vehicles and surrounding traffic infrastructure -- a
    # single fleet-wide twin (`cloud_twin`) feeding one centralized
    # `CloudController`. See the module docstring, point 11.
    fast_vehicle_speed_threshold: float = 20.0   # m/s; above = "fast" (edge), below = "slow" (cloud)
    cloud_period: int = 3           # cloud acts every cloud_period * high_level_period steps
    n_cloud_modes: int = 3          # discrete global coordination signal broadcast to every manager/worker
    cloud_actor_lr: float = 2e-3
    cloud_critic_lr: float = 8e-3
 
    # --- actor-critic learning ---
    gamma: float = 0.95                 # discount factor (both levels)
    manager_actor_lr: float = 3e-3
    manager_critic_lr: float = 1e-2
    worker_actor_lr: float = 5e-3
    worker_critic_lr: float = 1e-2
 
    @property
    def n_offload_choices(self) -> int:
        return 1 + self.n_edge_servers   # 0 = local, 1..n_edge = that edge server
 
    @property
    def worker_action_dim(self) -> int:
        return self.n_power_levels * self.n_offload_choices
 
    @property
    def manager_action_dim(self) -> int:
        return self.n_rbs * len(self.comm_intervals)
