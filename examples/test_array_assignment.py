"""
Test file for array assignment inside a matrix.
"""

import chex
import jax
import jax.numpy as jnp


def update_row_in_mat(positions: chex.Array):
    QF = jnp.zeros(shape=(10,2))
    # Fails, non-concrete array indices.
    # QF = QF.at[positions > 4].set(jnp.array([1, 2]))

    # Since updating multiple rows is hard as fuck, I will do this in a more functional programming way.
    def create_single_QF(single_pos: chex.Array):
        return jnp.select(
            condlist=[single_pos < 3, single_pos < 6, single_pos < 10],
            choicelist=[jnp.array([1, 2]), jnp.array([3, 4]), jnp.array([5, 6])],
            default=0.5
        )
    QF = jax.vmap(create_single_QF)(positions)
    return QF




if __name__ == "__main__":
    jitted_func = jax.jit(update_row_in_mat)
    positions = jnp.arange(0, 10)
    out = jitted_func(positions)
    print(out)


