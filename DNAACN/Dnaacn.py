# --------------------------------------------------------------------------- #
# Actor-Critic: actor (policy) + critic (state-value) MLPs, trained with TD
# --------------------------------------------------------------------------- #

class MLPPolicy:
    """The ACTOR half of an actor-critic pair: a 2-layer MLP + softmax policy.
    Unlike a plain-REINFORCE actor, `update()` here expects an advantage that
    has already been computed against a learned critic (TD target minus
    V(s)) rather than maintaining its own moving-average baseline -- the
    critic *is* the baseline now, and a learned, state-dependent baseline
    is lower-variance than a single running scalar."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 lr: float = 5e-3, seed: int | None = None):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / in_dim), (in_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, out_dim))
        self.b2 = np.zeros(out_dim)
        self.lr = lr

    def forward(self, x: np.ndarray):
        z1 = x @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        probs = softmax(z2)
        return probs, (x, z1, a1, probs)

    def act(self, x: np.ndarray, rng: np.random.Generator, greedy: bool = False):
        probs, cache = self.forward(x)
        action = int(np.argmax(probs)) if greedy else int(rng.choice(len(probs), p=probs))
        return action, probs, cache

    def update(self, cache, action: int, advantage: float):
        """Policy-gradient step using an externally supplied advantage
        (TD-error from a critic). loss = -log(pi(a|s)) * advantage."""
        x, z1, a1, probs = cache
        onehot = np.zeros_like(probs)
        onehot[action] = 1.0
        dz2 = (probs - onehot) * advantage

        dW2 = np.outer(a1, dz2)
        db2 = dz2
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = np.outer(x, dz1)
        db1 = dz1

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2


class MLPValue:
    """The CRITIC half: a 2-layer MLP regressing scalar state-value V(s),
    trained with a TD target (r + gamma * V(s')) via ordinary squared-error
    gradient descent."""

    def __init__(self, in_dim: int, hidden_dim: int, lr: float = 1e-2,
                 seed: int | None = None):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / in_dim), (in_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, 1))
        self.b2 = np.zeros(1)
        self.lr = lr

    def forward(self, x: np.ndarray):
        z1 = x @ self.W1 + self.b1
        a1 = relu(z1)
        v = (a1 @ self.W2 + self.b2).item()
        return v, (x, z1, a1)

    def update(self, cache, target: float) -> float:
        """One gradient step toward `target`. Returns the TD error
        (target - V(s)), which doubles as the actor's advantage."""
        x, z1, a1 = cache
        v = (a1 @ self.W2 + self.b2).item()
        td_error = target - v

        d_out = np.array([-td_error])           # d(0.5*(v-target)^2)/dv = -(target-v)
        dW2 = np.outer(a1, d_out)
        db2 = d_out
        da1 = d_out @ self.W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = np.outer(x, dz1)
        db1 = dz1

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        return td_error


class ActorCritic:
    """Ties one MLPPolicy (actor) and one MLPValue (critic) together and
    implements the TD actor-critic update: bootstrap a target from the
    critic's estimate of the next state, use (target - V(s)) as the
    advantage for the policy-gradient step, and regress the critic toward
    the same target. `gamma_power` lets a higher-level ("manager")
    decision that spans several base timesteps discount by gamma**period
    instead of gamma**1, matching the semi-MDP / options-framework version
    of actor-critic used for the manager level below."""

    def __init__(self, in_dim: int, hidden_actor: int, hidden_critic: int, out_dim: int,
                 actor_lr: float = 5e-3, critic_lr: float = 1e-2, gamma: float = 0.95,
                 seed: int | None = None):
        self.actor = MLPPolicy(in_dim, hidden_actor, out_dim, lr=actor_lr, seed=seed)
        self.critic = MLPValue(in_dim, hidden_critic, lr=critic_lr,
                                seed=(seed + 5000) if seed is not None else None)
        self.gamma = gamma

    def act(self, x: np.ndarray, rng: np.random.Generator, greedy: bool = False):
        return self.actor.act(x, rng, greedy)

    def update(self, actor_cache, action: int, reward: float,
               next_x: np.ndarray | None, gamma_power: int = 1) -> float:
        """TD update for one transition (s, a, r, s'). Pass next_x=None for
        a terminal / truncated transition (bootstrap target = r alone)."""
        x = actor_cache[0]
        _, critic_cache = self.critic.forward(x)
        if next_x is None:
            v_next = 0.0
        else:
            v_next, _ = self.critic.forward(next_x)
        target = reward + (self.gamma ** gamma_power) * v_next
        td_error = self.critic.update(critic_cache, target)   # = advantage
        self.actor.update(actor_cache, action, td_error)
        return td_error
