from init_variables import *
from utils import *


########################################################
# Network & Policy Creation
########################################################


# A cyclic buffer of bounded size that holds the transitions observed recently
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class PolicyNetworkDQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(PolicyNetworkDQN, self).__init__()
        layers = [
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        q_values = self.model(x)
        return q_values


class AnnealedEpsGreedyPolicy(object):
    def __init__(self, eps_start=0.9, eps_end=0.05, eps_decay=400):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0

    def select_action(self, q_values):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            return np.argmax(q_values)
        else:
            return random.randrange(len(q_values))


def update_model(memory, policy_net, target_net):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
                                  dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])

    state_batch = torch.stack(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.stack(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = reward_batch[:, 0] + (GAMMA * next_state_values)

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def env_initial_state():
    return np.repeat(0, 2 * T)


def env_step(t, state, action):
    next_state = np.repeat(0, len(state))
    next_state[0] = price_grid[action]
    next_state[1:T] = state[0:T - 1]
    next_state[T + t] = 1
    reward = profit_t_response(next_state[0], next_state[1])
    return next_state, reward


def to_tensor(x):
    return torch.from_numpy(np.array(x).astype(np.float32))


def to_tensor_long(x):
    return torch.tensor([[x]], device=device, dtype=torch.long)


# Environment simulator
def plus(x):
    return 0 if x < 0 else x


def minus(x):
    return 0 if x > 0 else -x


def shock(x):
    return np.sqrt(x)


# Demand at time step t for current price p_t and previous price p_t_1
def q_t(p_t, p_t_1, q_0, k, a, b):
    return plus(q_0 - k * p_t - a * shock(plus(p_t - p_t_1)) + b * shock(minus(p_t - p_t_1)))


# Profit at time step t
def profit_t(p_t, p_t_1, q_0, k, a, b, unit_cost):
    profit_return = q_t(p_t, p_t_1, q_0, k, a, b) * (p_t - unit_cost)
    return profit_return;


# Total profit for price vector p over len(p) time steps
def profit_total(p, unit_cost, q_0, k, a, b):
    profit_return = profit_t(p[0], p[0], q_0, k, 0, 0, unit_cost) + sum(
        map(lambda t: profit_t(p[t], p[t - 1], q_0, k, a, b, unit_cost), range(len(p))))
    return profit_return;


# Partial bindings for readability
def profit_t_response(p_t, p_t_1):
    profit_return = profit_t(p_t, p_t_1, q_0, k, a_q, b_q, unit_cost)
    return profit_return;


def profit_response(p):
    profit_return = profit_total(p, unit_cost, q_0, k, a_q, b_q)
    return profit_return;


#########################################################
# INITIALIZE NETWORK & POLICY
#########################################################
policy_net = PolicyNetworkDQN(2 * T, len(price_grid)).to(device)
target_net = PolicyNetworkDQN(2 * T, len(price_grid)).to(device)
policy = AnnealedEpsGreedyPolicy()
memory = ReplayMemory(10000)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.AdamW(policy_net.parameters(), lr=0.005)
