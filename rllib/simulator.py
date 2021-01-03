from rllib.rl import *
from utils import *


num_episodes = 1000
return_trace = []
p_trace = []  # price schedules used in each episode


# Visualize price-demand functions
for i in range(len(price_grid)):
    for j in range(len(price_change_grid)):
        profit_map[i, j] = profit_t_response(price_grid[i], price_grid[i] * price_change_grid[j])

plt.figure(figsize=(16, 5))
for i in range(len(price_change_grid)):
    if math.isclose(price_change_grid[i], 1.0):
        color = 'red'
    else:
        color = (0.6, 0.3, price_change_grid[i] / 2.0)
    plt.plot(price_grid, profit_map[:, i], c=color)
plt.xlabel("Price ($)")
plt.ylabel("Profit")
plt.legend(np.int_(np.round((1 - price_change_grid) * 100)), loc='lower right', title="Price change (%)",
           fancybox=False, framealpha=0.6)
plt.show()


##########################################################
# Run Simulation
##########################################################

for i_episode in range(num_episodes):
    state = env_initial_state()
    reward_trace = []
    p = []
    for t in range(T):
        # Select and perform an action
        with torch.no_grad():
          q_values = policy_net(to_tensor(state))
        action = policy.select_action(q_values.detach().numpy())

        next_state, reward = env_step(t, state, action)

        # Store the transition in memory
        memory.push(to_tensor(state),
                    to_tensor_long(action),
                    to_tensor(next_state) if t != T - 1 else None,
                    to_tensor([reward]))

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        update_model(memory, policy_net, target_net)

        reward_trace.append(reward)
        p.append(price_grid[action])

    return_trace.append(sum(reward_trace))
    p_trace.append(p)

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

        print(f'Episode {i_episode} of {num_episodes} ({i_episode/num_episodes*100:.2f}%)')

# .................
plot_return_trace(return_trace)
fig = plt.figure(figsize=(16, 5))
plot_price_schedules(p_trace, 5, 1, fig.number)
fig.show()

for profit in sorted(profit_response(s) for s in p_trace)[-10:]:
    print(f'Best profit results: {profit}')

profits = np.array([ profit_response(np.repeat(p, T)) for p in price_grid ])
p_idx = np.argmax(profits)
price_opt_const = price_grid[p_idx]
print(f'Optimal price is {price_opt_const}, achieved profit is {profits[p_idx]}')


# Debugging Q-values computations
transitions = memory.sample(10)
batch = Transition(*zip(*transitions))
non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])

state_batch = torch.stack(batch.state)
action_batch = torch.cat(batch.action)
reward_batch = torch.stack(batch.reward)

state_action_values = policy_net(state_batch).gather(1, action_batch)
next_state_values = torch.zeros(len(transitions), device=device)
next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
expected_state_action_values = (next_state_values * GAMMA) + reward_batch[:, 0]
q_trace = []

# Playing several episods and recording Q-values with the corresponding actual return
num_episodes = 100
return_trace = []
q_values_rewards_trace = np.zeros((num_episodes, T, 2, ))
for i_episode in range(num_episodes):
    state = env_initial_state()
    for t in range(T):
        # Select and perform an action
        with torch.no_grad():
          q_values = policy_net(to_tensor(state)).detach().numpy()
        action = policy.select_action(q_values)

        next_state, reward = env_step(t, state, action)

        # Move to the next state
        state = next_state

        q_values_rewards_trace[i_episode][t][0] = q_values[action]
        for tau in range(t):
          q_values_rewards_trace[i_episode][tau][1] += reward * (GAMMA ** (t - tau))


# Visualizing the distribution of Q-value vs actual returns
values = np.reshape(q_values_rewards_trace, (num_episodes * T, 2, ))
df = pd.DataFrame(data=values, columns=['Q-value', 'Return'])
g = sns.jointplot(x="Q-value", y="Return", data=df, kind="kde", color="crimson", height=10, aspect=1.0)
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+", alpha=0.1)
g.ax_joint.collections[0].set_alpha(0)
x0, x1 = g.ax_joint.get_xlim()
y0, y1 = g.ax_joint.get_ylim()
lims = [max(x0, y0), min(x1, y1)]
g.ax_joint.plot(lims, lims, ':k')


print('DONE..')
