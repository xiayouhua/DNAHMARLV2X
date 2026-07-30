# DNAHMARLV2X
Driving Network Algorithm Based on Hierarchical Multi-agent Reinforcement Learning for the Internet of Vehicles
"""
Hierarchical Multi-Agent RL for V2X Communication
====================================================
Edge Computing + Digital Twins + Vision-Language-Action (VLA) edition
----------------------------------------------------------------------
 
Extends the base hierarchical MARL V2X system with three additions
commonly discussed in 6G / next-gen V2X architectures:
 
1. EDGE SERVERS (MEC)
   Each vehicle periodically generates a compute task (e.g. cooperative
   perception fusion, trajectory prediction). It can be processed
   on-board ("local", fast to decide, limited compute) or offloaded
   over the air to one of several Multi-access Edge Computing (MEC)
   servers co-located with the RSU (more compute, but costs airtime +
   queueing delay). Workers now pick a *joint* (power level, offload
   target) action, so the RL problem is a joint communication +
   computation offloading decision — exactly the sense in which real
   V2X edge systems are "communication AND computation" co-design
   problems.
 
2. DIGITAL TWINS
   A lightweight `DigitalTwin` (an alpha-beta / Holt's-style
   exponential-smoothing filter) mirrors the real network state
   (per-cluster queue/channel statistics, per-edge-server load) and
   produces a forecast. Managers condition their resource-block
   decisions on the twin's *forecast* rather than a raw instantaneous
   reading, and workers use a twin-forecasted edge load (rather than
   polling every edge server every step) to pick offload targets —
   illustrating "twin-assisted" proactive network control, which is
   how digital twins are actually used in O-RAN / 6G MEC proposals:
   as a synchronized, predictive shadow of the real system that
   control loops query instead of the (expensive-to-poll) real thing.
 
3. VISION-LANGUAGE-ACTION (VLA) PIPELINE: ENCODE -> PROCESS -> DECODE
   Real VLA models (RT-2, OpenVLA, etc.) map camera frames + a
   language instruction directly to actions/embeddings. We obviously
   can't run a real multi-billion-parameter VLA checkpoint in this
   environment, so `VLAPipeline` is an explicit, clearly-labeled STUB
   that mimics the *interface* as three stages:
     - encode(): compresses the vehicle's transmitted V2X message
       (position, speed, queue backlog, a raw hazard sensor reading,
       channel quality, current RB) together with a simulated camera
       embedding into a compact latent code -- i.e. what conceptually
       goes "over the air" is the semantic code, not the raw payload.
     - process(): the VLA core -- takes the latent code plus a
       language embedding describing the vehicle's current V2X
       application context (e.g. "platooning", "intersection
       monitoring") and emits a discrete driving *intent* (normal /
       hazard_alert / maneuver) plus a continuous action embedding.
     - decode(): reconstructs an estimate of the original message
       (for a reconstruction-fidelity diagnostic) and a small decoded
       *action hint* that downstream RL workers condition on.
   Intent and the decoded action hint both feed into the worker's RL
   state (so scheduling can react to them), and intent modulates
   reward (a delivered hazard-alert message is worth more than
   routine telemetry, an undelivered one is penalized harder) — this
   is "perception-informed scheduling": a foundation model tells the
   network *what matters right now*, and a lightweight RL policy
   decides *how to serve it* under a real radio/compute budget. To
   plug in a real VLA, replace the three methods with forward passes
   through real encoder / VLA-core / decoder checkpoints and keep the
   same input/output shapes at each stage boundary.
 
4. ENVIRONMENTAL COMPLEXITY, SURROUNDING-VEHICLE AWARENESS, AND
   ADAPTIVE COMMUNICATION INTERVALS
   Three further additions aimed squarely at efficient operation under
   realistic, spatially- and temporally-varying conditions:
     - Environmental complexity: the road is split into zones (open/
       highway -> suburban -> dense-urban). Higher zones add extra
       signal blockage (harder comms) and a higher background hazard
       rate (harder safety) -- an environment property, independent of
       however many vehicles happen to be nearby.
     - Surrounding-vehicle awareness: every worker's observation now
       has an "ego" half (its own queue/channel/speed) and a
       "neighborhood" half -- the count, mean relative speed, mean
       queue backlog, and hazard-intent fraction of every vehicle
       within communication range. Managers likewise see their
       cluster's mean environmental-complexity zone and neighbor
       density, not just their own aggregate traffic stats.
     - Adaptive communication interval: managers now pick a *joint*
       (resource block, transmission interval) action per decision
       period. A short interval (transmit every step) minimizes
       latency/staleness at the cost of airtime and mutual
       interference; a long interval (transmit every 2nd/4th step)
       saves both but risks missing a time-critical window -- exactly
       the tradeoff real V2X congestion-control schemes (e.g.
       decentralized congestion control / adaptive beaconing) are
       built to manage, and here it's learned rather than hand-tuned.
 
Everything is still pure NumPy (no torch/tf available in this
environment). Both hierarchy levels are trained with actor-critic (see
point 5) rather than plain REINFORCE.
 
5. ACTOR-CRITIC LEARNING 
   Every manager and every worker is now an `ActorCritic` pair: an
   `MLPPolicy` actor plus an `MLPValue` critic that learns the
   state-value function V(s) via TD targets. The critic's estimate
   *is* the baseline now (state-dependent and learned, rather than one
   running scalar shared across all states), and the policy-gradient
   step uses the TD error (target - V(s)) as its advantage. Workers
   bootstrap one step ahead (ordinary TD(0) across consecutive
   environment steps); managers bootstrap across their whole decision
   window, discounting the critic's next-state estimate by
   `gamma ** high_level_period` -- the semi-MDP / options-framework
   version of the same idea, since a manager's "step" spans several
   base timesteps.
 
6. RICHER DRIVING STATE: LATERAL POSITION, STEERING ANGLE, RELATIVE
   LONGITUDINAL/LATERAL DISTANCE
   Vehicles previously only had a 1-D longitudinal position. They now
   also carry a lateral (cross-road/lane) position and a steering
   angle, updated by a small kinematic single-track (bicycle) model:
   a `maneuver` VLA intent steers the vehicle toward its persistent
   lane-change bias, otherwise steering relaxes back to straight-ahead
   plus lane-keeping jitter. Each worker's observation is extended
   with its own (lateral position, steering angle) plus the signed
   relative LONGITUDINAL distance and relative LATERAL distance to its
   single nearest neighbor -- classic ACC/platooning-style relative-
   motion features -- and the transmitted V2X message (see point 3)
   now carries lateral position and steering angle too, so the VLA
   pipeline's encode/decode stages compress and reconstruct genuine
   driving state, not just comms/queue telemetry.
 
7. DISCRETE DRIVING ACTION SPACE, AND THE EXACT SPEED / SAFE-DISTANCE /
   DRIVING-TENDENCY REWARD FORMULAS
   Physical driving is now its own decision, separate from the comms/
   offload action: every vehicle picks one of four discrete actions
   each step -- forward, left, right, reverse (`DRIVE_ACTIONS`) -- via
   its own `ActorCritic` (`HierarchicalTrainer.drivers`), trained on
   its own reward, computed by `V2XEnv.driving_reward()`. FORWARD
   accelerates toward the local desired speed (zone/disaster/
   construction-adjusted); REVERSE decelerates (and can go slightly
   negative); LEFT/RIGHT set a steering target the heading eases
   toward. The VLA `maneuver` intent no longer actuates steering
   directly -- it remains a semantic signal (message content, state
   context, hazard reward) -- since actual lane-change/speed control
   is now this explicit action.
 
   The reward is exactly:  r = r_s + r_n + r_dt
     - r_s = (v_s - mean(v_1..v_k)) * p_s -- reward for being faster
       than the surrounding traffic's mean speed (k = up to
       n_surrounding_vehicles nearest vehicles by real 2-D distance,
       not just the single nearest neighbor used elsewhere).
     - r_n = p_n * sum(1 / (d_i - r)), p_n < 0 -- a potential-field
       safe-distance penalty that blows up as any of the k nearest
       vehicles' distance d_i approaches the safety radius r from
       above (d_i is clipped just above r to avoid a divide-by-zero
       singularity if the simulated dynamics ever put two vehicles
       closer than r).
     - r_dt: with `rz` = 15 + clip(right_space - left_space, -15, 15)
       (right_space/left_space = open lateral room on each side, in
       meters, reduced by any nearby vehicle occupying that side) --
       r_dt = -p_dt*(rz-15) if the action was LEFT, +p_dt*(rz-15) if
       RIGHT, else 0. Turning toward the side that actually has more
       room earns positive reward; turning the wrong way is penalized.
       NOTE: the "15" pivot and the shape of the formula were
       specified exactly; rz's underlying units/scale were not --
       meters (via `_lateral_space`) was the choice made here to make
       it concrete; rescale `driving_tendency_neutral` /
       `lane_change_window_m` if a specific intended range is wanted.
 
   This reward trains the drivers only; the comm/offload workers' own
   reward is unaffected (comms/compute/hazard-message/risk/obstacle
   terms), and the two agents per vehicle are updated independently.
 
8. OD-PAIR RISK TOLERANCE, ADAPTIVE CAPACITY, RISK VALUE, AND
   NATURAL-DISASTER ADAPTABILITY
     - Each vehicle is treated as one continuous trip along a fixed OD
       pair. `_update_risk_metrics()` tracks its cumulative count of
       steps WITHOUT a path (lane) change, t, vs WITH one, t_c (a
       LEFT/RIGHT driving action counts as a path change), and derives:
         RT = t / (t + t_c)                       risk tolerance
         p  = t_c / (t + t_c)  ( = 1 - RT )        path-change probability
         K  = p / (1 - p)  if 0 < p < 0.5          adaptive capacity
         K  = 1            if 0.5 <= p < 1
         rv = 1 / (RT + K)                         risk value
       These counts persist ACROSS episodes (initialized once, not
       reset every episode) since they represent accumulating trip
       history, not a per-episode quantity. rv feeds risk_reward
       (-rv * risk_penalty_weight) directly into the comm/offload
       worker's reward; RT, K, and rv are also part of its state.
       A pooled, fleet-wide version of the same formulas
       (system_risk_tolerance / _adaptive_capacity / _risk_value) is
       tracked too, since K is meant to characterize a *transportation
       system's* adaptive capacity, not just one vehicle's -- it's
       exposed as a training diagnostic and as part of each cluster's
       manager state (mean K, mean rv).
       Note: because the driving-tendency reward (point 7) rewards
       turning toward whichever side has more room, a trained driver
       often ends up changing lanes more than half the time, which
       saturates K at 1 and pushes rv toward its ceiling -- risk_reward
       then becomes a large, fairly constant penalty. That's the
       formula behaving exactly as specified; `risk_penalty_weight` is
       the knob to retune if a more varying signal is wanted.
     - Natural-disaster adaptability: `tick_disaster()` stochastically
       starts, continues, and ends a storm/flood-like event. While
       active it adds extra signal blockage, multiplies the local
       hazard rate, knocks one edge server's capacity down, and lowers
       the safe speed. `disaster_active` is surfaced in both manager
       and worker state so the network can actually *adapt* -- e.g. a
       manager can shorten its communication interval despite the
       congestion cost once it sees a disaster is underway.
 
9. STATIC OBSTACLES: WALLS AND CONSTRUCTION SITES
   Fixed roadside infrastructure, placed once at environment
   construction (not re-randomized every episode, since real
   infrastructure doesn't move):
     - Walls block the RSU's line-of-sight over a longitudinal stretch
       of road, adding heavy extra signal blockage to any vehicle
       currently inside that stretch -- on top of the zone/disaster
       blockage already in the channel model.
     - Construction sites close a lane-width lateral band over a
       longitudinal stretch: while a vehicle is in that stretch AND
       still laterally inside the closed band, it's in "lane
       conflict" -- the local speed limit drops, the local hazard
       rate rises, and an obstacle penalty (scaled by
       obstacle_penalty_weight) applies until it actually steers out
       of the closed band. Lane conflict also feeds into risk value.
   Each worker's state gets the (wrap-aware) forward distance to the
   nearest obstacle, whether it's currently in a construction zone,
   and whether it's still in lane conflict, so the policy has what it
   needs to plan a lane change (via the existing `maneuver` VLA intent
   and steering kinematics) before it's forced to.
 
10. OPTIONAL REAL-WORLD DATASET GROUNDING: KITTI, nuScenes, ONCE
    `KITTIAdapter` / `NuScenesAdapter` / `ONCEAdapter` (unified by
    `DrivingDatasetLibrary`) parse each dataset's real, publicly-
    documented file layout -- KITTI's raw OXTS GPS/IMU logs, nuScenes'
    `ego_pose.json`, ONCE's per-sequence frame poses -- to extract real
    empirical vehicle speed trajectories. If `V2XConfig.dataset_name`
    and `dataset_root` are set to point at your own local copy of one
    of these datasets, `V2XEnv.reset()` bootstrap-samples each
    vehicle's initial speed from that real distribution instead of a
    flat synthetic uniform(15, 30) draw. None of the three datasets
    are (or could be) bundled here: KITTI raw is ~180GB, nuScenes full
    is 300GB+, ONCE is ~1TB, each requires accepting the provider's own
    license on their site, and this sandboxed environment has no
    network access regardless. With no dataset configured (the
    default), behavior is completely unchanged; with a bad/missing
    path, it degrades gracefully to the same synthetic draw rather
    than erroring. I validated the three parsers against small,
    hand-built mock files in each dataset's real format (not included
    in this file) to confirm the parsing logic itself is correct,
    since the genuine datasets aren't available to test against here.
 
11. FIVE-PART IoV ARCHITECTURE: VEHICLES, ROADS, TRAFFIC ENVIRONMENT,
    EDGE DIGITAL TWINS, CLOUD DIGITAL TWINS
    The simulation maps onto all five parts:
      - VEHICLES: the fleet (`V2XEnv.pos/vel/lat_pos/...`), each running
        a worker (comms/offload), a driver (physical motion), and its
        own OD-pair risk history.
      - ROADS: the circular road itself (`road_length_m`, `n_lanes`,
        `lane_width_m`) plus static infrastructure (walls, construction
        sites).
      - TRAFFIC ENVIRONMENT: the spatial complexity zones, natural-
        disaster process, and hazard rates that vary across them.
      - EDGE DIGITAL TWINS (`cluster_twins`): model nearby FAST-moving
        vehicles (`_is_fast_vehicle`, above `fast_vehicle_speed_threshold`)
        and their surrounding roads -- each syncs from its cluster's fast
        vehicles specifically. They serve the communication network
        (RB/power decisions) and sub-objective generation (the resource-
        block/interval choice each manager makes) of the hierarchical
        MARL, and their forecasts feed directly into the managers' and
        workers' Actor-Critic state.
      - CLOUD DIGITAL TWINS (`cloud_twin` + `CloudController`): model
        SLOW-moving vehicles and surrounding traffic infrastructure at
        FLEET scale (one twin, not per-cluster). They serve centralized
        control: a single `CloudController` actor-critic acts on a
        slower cadence (`cloud_period * high_level_period` steps) from
        the cloud twin's forecast, and broadcasts a discrete
        coordination mode (`CLOUD_MODES`) that every manager and worker
        conditions on -- centralized guidance sitting above the
        per-region managers, without overriding their individual
        decisions.
    V2V and V2I are both physically present, just not previously
    labeled as such:
      - V2V (`_neighbor_features`): vehicles exchange driving state and
        hazard warnings directly with nearby vehicles within
        `comm_range_m` -- feeds each worker's neighbor-density/relative-
        speed/hazard-fraction state.
      - V2I: the vehicle <-> RSU link (SINR/throughput, in `step()`)
        exchanges traffic/hazard messages with roadside infrastructure;
        the vehicle <-> edge-server link (task offloading) exchanges
        compute workload with edge infrastructure; and vehicles'
        aggregated state syncing into the edge/cloud digital twins is
        itself a V2I data flow feeding the transportation system's
        infrastructure-side models.
"""
