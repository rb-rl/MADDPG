# Technical Details

This report contains technical details on the approach used in this project.

## Implementation

The reinforcement learning multi-agent used in this project is based on a multi-agent deep deterministic policy gradient [1].

### Actor Updates

In this approach, two policies `π_i(o_i)` and `π_i'(o_i)` as well as two action-value functions `Q_i(o_1, o_2, ..., a_1, a_2, ...)` and `Q_i'(o_1, o_2, ..., a_1, a_2, ...)` are used per agent, where `i` is the index of the agent in the multi-agent, `o_1, o_2, ...` are the observations and `a_1, a_2, ...` the actions of the agents in the multi-agent. Each of these functions is approximated by its own neural network as described further below. The first of the two policies types is updated by backpropagation based on the loss

`L_actor_i = - Q_i(o_1, o_2, ..., a_1, ..., a_{i-1}, π_i(o_i), a_{i+1}, ...)`

averaged over a mini-batch. The second policy type is updated via a soft update according to

`π_i'(o_i) <- (1 - τ) * π_i'(o_i) + τ * π_i(o_i)` (1)

with the soft update rate `τ`. Note that this update is not performed every frame but only every `frames per update` frame.

### References

[1] Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments, 2017, [arxiv.org/pdf/1706.02275.pdf](https://arxiv.org/pdf/1706.02275.pdf)  
[2] [en.wikipedia.org/wiki/Ornstein–Uhlenbeck_process](https://en.wikipedia.org/wiki/Ornstein–Uhlenbeck_process)  
