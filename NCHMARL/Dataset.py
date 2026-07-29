# --------------------------------------------------------------------------- #
# Real-world driving dataset adapters: KITTI, nuScenes, ONCE
# --------------------------------------------------------------------------- #
#
# These parse each dataset's real, publicly-documented file layout to pull
# out empirical vehicle speed trajectories, so `V2XEnv` can seed vehicles'
# initial speeds from real driving data instead of a flat synthetic
# uniform(15, 30) draw. None of the three datasets are bundled here -- KITTI
# raw is ~180GB, nuScenes full is 300GB+, ONCE is ~1TB, each requires
# accepting its provider's own license on their site, and this sandboxed
# environment has no network access to fetch them regardless. Point
# `dataset_root` (in V2XConfig) at your own local copy; with no dataset
# configured (the default), behavior is completely unchanged from before.
#
#   KITTI:    https://www.cvlibs.net/datasets/kitti/           (raw OXTS logs)
#   nuScenes: https://www.nuscenes.org/                        (ego_pose.json)
#   ONCE:     https://once-for-auto-driving.github.io/         (per-frame pose)
 
class KITTIAdapter:
    """
    Reads KITTI raw OXTS (GPS/IMU) logs at `<root>/oxts/data/*.txt` -- one
    text file per frame, each a single line of 30 space-separated floats.
    Field index 8 is `vf`, forward velocity in m/s (see KITTI's
    `dataformat.txt`); that's the empirical speed signal extracted here.
    """
 
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
 
    def speed_samples(self) -> np.ndarray:
        oxts_dir = os.path.join(self.root_dir, "oxts", "data")
        speeds = []
        if os.path.isdir(oxts_dir):
            for fname in sorted(os.listdir(oxts_dir)):
                if not fname.endswith(".txt"):
                    continue
                with open(os.path.join(oxts_dir, fname)) as f:
                    vals = f.read().split()
                if len(vals) >= 9:
                    speeds.append(float(vals[8]))   # vf: forward velocity
        return np.array(speeds, dtype=float)
 
 
class NuScenesAdapter:
    """
    Reads a nuScenes-style `<root>/<split>/ego_pose.json`: a list of records
    with `translation` [x, y, z] (meters) and `timestamp` (microseconds).
    Speed is estimated by finite-differencing consecutive ego positions,
    since nuScenes doesn't store speed directly on the ego_pose record.
    """
 
    def __init__(self, root_dir: str, split: str = "v1.0-mini"):
        self.root_dir = root_dir
        self.split = split
 
    def speed_samples(self) -> np.ndarray:
        path = os.path.join(self.root_dir, self.split, "ego_pose.json")
        if not os.path.isfile(path):
            return np.array([])
        with open(path) as f:
            poses = json.load(f)
        poses = sorted(poses, key=lambda r: r["timestamp"])
        speeds = []
        for a, b in zip(poses[:-1], poses[1:]):
            dt = (b["timestamp"] - a["timestamp"]) / 1e6   # microseconds -> seconds
            if dt <= 0:
                continue
            dx = np.array(b["translation"][:2]) - np.array(a["translation"][:2])
            speeds.append(float(np.linalg.norm(dx)) / dt)   # planar speed
        return np.array(speeds, dtype=float)
 
 
class ONCEAdapter:
    """
    Reads a ONCE-style per-sequence annotation file at
    `<root>/<seq_id>/<seq_id>.json`, containing a `frames` list where each
    frame has a `pose` (translation + rotation). ONCE logs at roughly 10 Hz;
    speed is estimated the same way as nuScenes, by finite-differencing
    consecutive frame positions.
    """
 
    def __init__(self, root_dir: str, seq_id: str, frame_rate_hz: float = 10.0):
        self.root_dir = root_dir
        self.seq_id = seq_id
        self.frame_rate_hz = frame_rate_hz
 
    def speed_samples(self) -> np.ndarray:
        path = os.path.join(self.root_dir, self.seq_id, f"{self.seq_id}.json")
        if not os.path.isfile(path):
            return np.array([])
        with open(path) as f:
            meta = json.load(f)
        frames = [fr for fr in meta.get("frames", []) if "pose" in fr]
        frames.sort(key=lambda fr: float(fr["frame_id"]))
        speeds = []
        dt = 1.0 / self.frame_rate_hz
        for a, b in zip(frames[:-1], frames[1:]):
            dx = np.array(b["pose"][:2]) - np.array(a["pose"][:2])
            speeds.append(float(np.linalg.norm(dx)) / dt)
        return np.array(speeds, dtype=float)
 
 
class DrivingDatasetLibrary:
    """
    Thin unifying wrapper over the three adapters: pick a dataset by name,
    get back an empirical speed distribution (m/s) to bootstrap-sample from
    instead of the synthetic uniform(15, 30) draw. If the requested
    `root_dir` doesn't exist or doesn't contain the expected files, every
    method degrades gracefully to an empty array -- `V2XEnv` then falls
    back to the original synthetic draw automatically, which is the common
    case here since none of these datasets are actually present in this
    sandboxed environment.
    """
 
    ADAPTERS = {"kitti": KITTIAdapter, "nuscenes": NuScenesAdapter, "once": ONCEAdapter}
 
    def __init__(self, name: str, root_dir: str, **kwargs):
        name = name.lower()
        if name not in self.ADAPTERS:
            raise ValueError(f"Unknown dataset '{name}', expected one of {list(self.ADAPTERS)}")
        self.name = name
        self.adapter = self.ADAPTERS[name](root_dir, **kwargs)
 
    def speed_samples(self) -> np.ndarray:
        try:
            return self.adapter.speed_samples()
        except Exception:
            return np.array([])
 
    def sample_speeds(self, n: int, rng: np.random.Generator,
                       fallback_low: float = 15.0, fallback_high: float = 30.0) -> np.ndarray:
        """Bootstrap-resample n speeds from the empirical distribution, or
        fall back to the original uniform draw if no real samples were found
        (e.g. dataset_root not set, or points at a directory that doesn't
        actually contain the dataset)."""
        samples = self.speed_samples()
        if samples.size == 0:
            return rng.uniform(fallback_low, fallback_high, n)
        return rng.choice(samples, size=n, replace=True)
