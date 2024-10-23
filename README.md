# SARSA Learning Algorithm


## AIM
To develop a Python program to find the optimal policy for the given RL environment using SARSA-Learning and compare the state values with the Monte Carlo method.
## PROBLEM STATEMENT
Explain the problem statement.

## SARSA LEARNING ALGORITHM
Include the steps involved in the SARSA Learning algorithm

## SARSA LEARNING FUNCTION
### Name:MITHUN M S
### Register Number:212222240067

```
def sarsa(env,
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.5,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    select_action = lambda state, Q, epsilon: np.argmax(Q[state]) \
        if np.random.random() > epsilon \
        else np.random.randint(len(Q[state]))
    alphas = decay_schedule(init_alpha, 
                           min_alpha, 
                           alpha_decay_ratio, 
                           n_episodes)
    epsilons = decay_schedule(init_epsilon, 
                              min_epsilon, 
                              epsilon_decay_ratio, 
                              n_episodes)
    for e in tqdm(range(n_episodes), leave=False):
        state = env.reset()
        action = select_action(state, Q, epsilons[e])
        # Initialize done before the loop
        done = False  
        while not done:
          next_state, reward, done, _ = env.step(action)
          next_action = select_action(next_state, Q, epsilons[e])
          td_target = reward + gamma * Q[next_state][next_action]
          td_error = td_target - Q[state][action]
          Q[state][action] = Q[state][action] + alphas[e] * td_error
          state = next_state
          action = next_action
        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))
    V = np.max(Q,axis=1) # Fix: Assign V instead of v
    pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return Q, V, pi, Q_track, pi_track # Fix: Return V instead of v
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/60ac1a09-f5a8-41e3-954a-8e87309833ba)

![download](https://github.com/user-attachments/assets/071c86c7-e938-4055-b116-ed5c56d98266)

![download (1)](https://github.com/user-attachments/assets/6101f609-66fb-437a-8649-d7e4f9bdcc45)


## RESULT:

Write your result here
