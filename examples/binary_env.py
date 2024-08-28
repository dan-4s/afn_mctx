"""
Simple 2-level binary tree environment for testing of MCTS for AFNs.
"""

import numpy as np
from typing import NamedTuple
from jax import Array
import jax.numpy as jnp
import jax


# Create a simple 2-level binary tree environment.
class GameState(NamedTuple):
  player: Array = jnp.int32(0)
  board: Array = jnp.int32(0) # Starting at the root: s0 as defined by Game!
  game_over: Array = jnp.bool(False)


class Game:
  """
  We use a simple binary tree of 2 levels, with s7 as a post-terminal node:
           s6 -> R1 = 10, R2 = 0.1 - s7 -> R1 = 1000, R2 = 1000
         /
      s2 - s5 -> R1 = 1, R2 = 1
      /
    s0
      \ 
      s1 - s4 -> R1 = 1, R2 = 1
         \
           s3 -> R1 = 0.1, R2 = 10

  We can solve the above tree to get a root flow of 1 and a root policy of
  [0.091, 0.91].
  """
  alphazero_rewards: Array = jnp.array([-1, 0, 0, 1, 0])
  # afn_rewards: Array = jnp.array([0.1, 1, 1, 10, 1/1000])
  afn_rewards: Array = jnp.array([1, 1, 0.3678, 0.3678, 1000]) # These are for a TTT simulation.

  def init(self) -> GameState:
    return GameState()
  
  def step(self, state: GameState, action: Array) -> GameState:
    # Here we just want to update the state correctly according to the action.
    # Please excuse the horrendousness of this state transition. I did not
    # think of a smarter way of doing this.
    new_board = jnp.select(
      condlist=[
        # We need logical_and() here to avoid using the built-in Python 'and'
        # operator, which will fail a Jax tracer check.
        jnp.logical_and(state.board == 0, action == 0),
        jnp.logical_and(state.board == 0, action == 1),
        jnp.logical_and(state.board == 1, action == 0),
        jnp.logical_and(state.board == 1, action == 1),
        jnp.logical_and(state.board == 2, action == 0),
        jnp.logical_and(state.board == 2, action == 1),
        jnp.logical_or(state.board == 6, state.board == 5), # Any action leads to s7.
        jnp.logical_or(state.board == 3, state.board == 4), # Any action leads to s7.
      ],
      choicelist=[1, 2, 3, 4, 5, 6, 7, 7],
      default=state.board,
    )
    new_game_over = jnp.select(
      condlist=[jnp.logical_and(new_board >= 3, new_board <= 6)],
      choicelist=[True],
      default=False,
    )
    return state._replace(
      player=1 - state.player,
      board=new_board,
      game_over=new_game_over,
    )
  
  def is_terminal(self, state: GameState) -> Array:
    return state.game_over
  
  def rewards(self, state: GameState, is_afn: bool = True) -> Array:
    """
    Rewards must be given from the opposing player's point of view! This is
    because the reward is technically tallied when the state is terminal, but
    the "winner" is the player that played last before the state became
    terminal. As such, in this simple environment, the reward must be negated
    to be correctly used in a tree search. To use this method, invert or negate
    the reward!!!
    """
    rewards = self.afn_rewards if(is_afn) else self.alphazero_rewards
    return jnp.select(
      condlist=[state.board == 3, state.board == 4, state.board == 5, state.board == 6, state.board == 7],
      choicelist=rewards,
      default=0,
    )


if __name__ == "__main__":
  # This file used to have tests, but they were a pain to maintain.
  print("Not a runnable file.")
