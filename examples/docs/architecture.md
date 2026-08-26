# Reference architecture notes

The agent performs an explicit state update after every structured observation.
Canonical action hashes prevent identical tool actions from being executed twice.
A maximum step count and a no-progress counter guarantee bounded termination.
The model backend can be autoregressive or diffusion-based because generation is lazy and pluggable.
