# Technical Details

This report contains technical details on the approach used in this project.

## Implementation

The reinforcement learning multi-agent used in this project is based on a multi-agent deep deterministic policy gradient [1].

### Actor Updates

In this approach, two policies `π_i(o_i)` and `π_i'(o_i)` as well as two action-value functions `Q_i(o_1, o_2, ..., a_1, a_2, ...)` and `Q_i'(o_1, o_2, ..., a_1, a_2, ...)` are used per agent, where `i` is the index of the agent in the multi-agent, `o_1, o_2, ...` are the observations and `a_1, a_2, ...` the actions of the agents in the multi-agent. Each of the functions above is approximated by its own neural network as described further below. The first of the two policy types is updated by backpropagation based on the loss

`L_actor_i = - Q_i(o_1, o_2, ..., a_1, ..., a_{i-1}, π_i(o_i), a_{i+1}, ...)`

averaged over a mini-batch. The second policy type is updated via a soft update according to

`π_i'(o_i) <- (1 - τ) * π_i'(o_i) + τ * π_i(o_i)` (1)

with the soft update rate `τ`. Note that this update is not performed every frame but only every `frames per update` frame.

### Critic Updates

The first of the two action-value function types `Q_i(o_1, o_2, ..., a_1, a_2, ...)` and `Q_i'(o_1, o_2, ..., a_1, a_2, ...)` is updated by backpropagation with the loss

`L_critic_i = (r_i + γ * Q_i'(o_1', o_2', ..., π_1'(o_1'), π_2'(o_2'), ...) - Q_i(o_1, o_2, ..., a_1, a_2, ...))^2`

averaged over also a mini-batch, where `r_i` is the reward when going from the current observation `o_i` to the next observation `o_i'` of the agent with index `i` and `γ` is the discount factor.

The second action-value function type is updated via a soft update according to

`Q_i'(o_1, o_2, ..., a_1, a_2, ...) <- (1 - τ) * Q_i'(o_1, o_2, ..., a_1, a_2, ...) + τ * Q_i(o_1, o_2, ..., a_1, a_2, ...)`

similar to the replacement rule (1) and with the same frequency.

### Network topology

Each of the policies `π_i(o_i)` and `π_i'(o_i)` is represented by a fully connected neural network consisting of 2 hidden layers with 64 neurons per layer, yielding the network architecture

`24 -> 64 -> 64 -> 2`

The numbers 24 and 2 come from the sizes of the observation and action spaces, because a policy network takes an observation as an input and outputs an action.

For the action-value functions `Q_i(o_1, o_2, ..., a_1, a_2, ...)` and `Q_i'(o_1, o_2, ..., a_1, a_2, ...)`, the inputs are observations and actions while the outputs are the scalar Q-values. As the observations and actions are concatenated when entering the corresponding neural network, this yields the in- and output sizes 52 (=2*(24+2)) and 1. The fully connected network architectures chosen for the deep Q-networks are therefore similar to the one shown above, but now of the form

`52 -> 64 -> 64 -> 64 -> 1`

with 3 hidden layers. The hidden layers of all architectures have the `rectified linear unit` (=relu) as the activation function. The output layers of the policy networks have a `tanh` activation function whereas the output layers of the deep Q-networks are purely linear.

### Backpropagation

Backpropagation of the neural networks behind the policies `π_i(o_i)` and action-value functions `Q_i(o_1, o_2, ..., a_1, a_2, ...)` is done with mini-batch gradient descent based on the the learning rate `α=0.001` and `batch size=64`. The optimizer used for that purpose is an Adam optimizer.

### Policy

The policy of the `i`-th agent is based on using the policy network `π_i(o_i)` in combination with a noise `N`:

`a_i = π_i(o_i) + N`

The noise is given by an Ornstein-Uhlenbeck process [2], i.e. it is updated according to

`N <- (1 - θ) * N + ε * W`

where `W` stands for the Wiener process and is a 4-vector for each update step distributed according to a multivariate Gaussian. The constant `θ` is the decay rate of the noise and `ε` is the standard deviation of the newly added noise.

### Replay memory

Also, a replay memory is used, which can store 10000 elements, where the oldest elements are discared if the limit of the memory is reached.

## Hyperparameters

A summary of the hyperparameters used to solve the environment is given in the following. The summary is split into a table of the selected values and a detailed description of the meaning of the hyperparameters.

### Selected Values

- `α = 0.001`
- `γ = 0.99`
- `ε interval = [0.2, 0.2]`
- `ε decay factor = 1`
- `batch size = 64`
- `loss = mse`
- `τ = 0.01`
- `frames per update = 4`
- `θ = 0.2`

- `max replay memory size = 100000`

- `activation function = relu`
- `actor number of hidden layers = 2`
- `actor hidden neurons per hidden layer = 64`
- `critic number of hidden layers = 3`
- `critic hidden neurons per hidden layer = 64`
- `use batch normalization = False` (activating batch normalization astonishingly decreases the learning performance)

### Agent
- `α = The learning rate, where higher values mean that the multi-agent learns faster`
- `γ = The discount factor in [0, 1], where higher values mean a stronger influence of future rewards on the now`
- `ε interval = The interval in which epsilon decay can take place`
- `ε decay factor = The epsilon decay factor in [0, 1], where higher values mean a slower decay`
- `batch size = The mini-batch size, i.e. the number of samples simultaneously used in backpropagation`
- `loss = The loss function to be optimized in gradient descent`
- `τ = The soft-update rate in [0, 1], where higher values mean that the target networks becomes equal to the normal networks faster`
- `frames per update = The number of frames which have to pass for a soft-update step of the target networks`
- `θ = The decay rate of the Ornstein-Uhlenbeck noise in [0, 1], where higher values mean a faster decay`

### Replay memory
- `max replay memory size = The maximally possible size of the replay memory`

### Neural networks
- `activation function = The activation function of the hidden layers`
- `actor number of hidden layers = The number of hidden layers in each of the actor neural networks`
- `actor hidden neurons per hidden layer = The number of neurons per hidden layer in the actors`
- `critic number of hidden layers = The number of hidden layers in each of the critic neural networks`
- `critic hidden neurons per hidden layer = The number of neurons per hidden layer in the critics`
- `use batch normalization = Determines whether batch normalization on the observation inputs of the actors and critics is activated`

## Solution

As explained in the [README.md](README.md), the environment is considered as solved, if the average collective score over 100 consecutive episodes is at least +0.5. A solution of the environment was achieved in 5044 episodes, as shown by the following screenshot from [Main.ipynb](Main.ipynb):

![Episodes_Number](https://user-images.githubusercontent.com/92691697/139137963-b1550f11-4c6b-494a-9efb-51f1e0805c8c.PNG)

The collective score per episode over the training process is shown in the following screenshot. This score is defined as the maximum of the individual scores of the two agents. An individual score is the non-discounted return, i.e. cumulative reward per episode.

![Score](https://user-images.githubusercontent.com/92691697/139138187-b2e1c1f0-3ea5-44eb-a6d1-e8207cc738c2.PNG)

## Limitation

When looking at the policies `π_i(o_i)` and `π_i'(o_i)` as well as action-value functions `Q_i(o_1, o_2, ..., a_1, a_2, ...)` and `Q_i'(o_1, o_2, ..., a_1, a_2, ...)`, we observe a nonlocal behavior. Although each policy takes only the observation `o_i` of the corresponding agent `i`, each action-value functions need the observations and actions of all agents in the multi-agent. Therefore, the action-value functions are nonlocal. As these functions enter the training process of the agents of the multi-agent, each agent is trained as if it can access the observations of all agents. This makes the training unnatural from a biological perspective, where observations cannot be exchanged between animals. However, at least the inference is local.

## Ideas for improvements

Although the environment has been solved by the present approach, there are several possible ways to make improvements. Such improvements will impact in how many episodes the average collective score of +0.5 mentioned above is reached. And they will also affect the maximum average collective score reachable if the training would continue indefinitely.

The suggested improvements are the following ones:
- Exploitation of symmetry: As both agents solve the same tasks, the number of training parameters could be reduced by using the same actor and critic networks for both agents.  The difference to using two DDPG agents would then be the presence of a nonlocal critic.
- Continued manual adjustment of the hyperparameters: A certain amount of manual hyperparameter tuning (including network topology) was invested in this project. However, the upper limit has not yet been reached here. Unfortunetly, the tweaking of the hyperparameters becomes the more time intensive, the more fine-tuned they are.
- Auto machine learning: The hyperparameters can also be tuned automatically by performing a grid search or even better a random search.
- Modification of the amount the past is covered by observation space: The neural networks provided in this repository are time delay neural networks, because the observations consist of stacks of 3 frames. By storing these frames in a separate memory, it would be easily possible to change the stack size and introduce it as a hyperparameter which could be optimized.
- Policy ensembles: According to [1], robustness could be improved by using policy ensembles for each agent of the multi-agent.
- Recurrent neural networks: By using recurrent layers, the past states could be handled beyond mere time delay neural networks.
- Prioritized replay memory: The replay memory used in this project is not prioritized such that there is an improvement option.
- Distributed Distributional MADDPG [3]: In DDPG, every state, action pair (s,a) has only a single scalar value Q. Distributional approaches extend this by providing a distribution over multiple Q-values. Such an extension is also possible for MADDPG.
- Twin Delayed Multi-Agent Deep Deterministic (=TD3) [3]: DDPG can be extended by using the two action-value functions in a different way, having the policy network being soft-updated at a lower rate than the deep Q-network and by introducing an extra noise in the loss of the critic. This can be generalized to MADDPG.
- Attention: Primarily used in natural language processing, attention layers could also be explored in this context of this project.

### References

[1] Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments, 2017, [arxiv.org/pdf/1706.02275.pdf](https://arxiv.org/pdf/1706.02275.pdf)  
[2] [en.wikipedia.org/wiki/Ornstein–Uhlenbeck_process](https://en.wikipedia.org/wiki/Ornstein–Uhlenbeck_process)  
[3] [lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)
