# Q Learning Algorithm


## AIM
To develop and evaluate the Q-learning algorithm's performance in navigating the environment and achieving the desired goal.

## PROBLEM STATEMENT
The goal of this project is to implement a Q-learning algorithm that enables an agent to learn optimal actions in a dynamic environment to maximize cumulative rewards.

## Q LEARNING ALGORITHM:
Step 1:
Initialize the Q-table with zeros for all state-action pairs based on the environment's observation and action space.

Step 2:
Define the action selection method using an epsilon-greedy strategy to balance exploration and exploitation.

Step 3:
Create decay schedules for the learning rate (alpha) and epsilon to progressively reduce their values over episodes.

Step 4:
Loop through a specified number of episodes, resetting the environment at the start of each episode.

Step 5:
Within each episode, continue until the episode is done, selecting actions based on the current state and the epsilon value.

Step 6:
Execute the chosen action to obtain the next state and reward, and compute the temporal difference (TD) target.

Step 7:
Update the Q-value for the current state-action pair using the TD error and the learning rate for that episode.

Step 8:
Track the Q-values, value function, and policy after each episode for analysis and evaluation.


## Q LEARNING FUNCTION
### Name:M.K.Suriya prakash
### Register Number:212224110053
```

def q_learning(env,
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

    # Corrected indentation for the select_action lambda
    select_action = lambda state, Q, epsilon: \
        np.argmax(Q[state]) if np.random.random() > epsilon \
        else np.random.randint(len(Q[state]))

    alphas = decay_schedule(
      init_alpha, min_alpha,
      alpha_decay_ratio, n_episodes)

    epsilons = decay_schedule(
      init_epsilon, min_epsilon,
      epsilon_decay_ratio, n_episodes)

    for e in tqdm(range(n_episodes), leave=False):
        state, done = env.reset(), False
        while not done:
            action = select_action(state, Q, epsilons[e])
            next_state, reward, done, _ = env.step(action)
            td_target = reward + gamma * Q[next_state].max() * (not done)
            td_error = td_target - Q[state][action]
            Q[state][action] = Q[state][action] + alphas[e] * td_error
            state = next_state
        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))
    V = np.max(Q, axis=1)
    pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return Q, V, pi, Q_track, pi_track
```





## OUTPUT:
Mention the optimal policy, optimal value function , success rate for the optimal policy
### Optimal Policy:
![image](https://github.com/user-attachments/assets/5fd5c109-3df2-4d70-896b-ed5c6b6f7624)

### Optimal Value function:
![Screenshot 2025-05-21 111738](https://github.com/user-attachments/assets/9542d61d-ba47-4552-b945-9de20c1ed850)

### Success rate of optimal policy:
![image](https://github.com/user-attachments/assets/708e720c-ec1a-4c07-a88e-1800baead582)




### Plot comparing the state value functions of Monte Carlo method and Qlearning.
### Monte Carlo method:
![image](https://github.com/user-attachments/assets/88ce3433-8539-4d90-b5b6-31f6dd152f79)
### Qlearning:
![image](https://github.com/user-attachments/assets/3d4cfdc5-6964-4eac-bbbc-5a683d954906)


## RESULT:

Thus the python program to implement Q-learning is implemented successfully
