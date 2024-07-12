"""
File to test Jax while-loops and for-loops.
"""

import functools
import jax
import chex
import numpy as np


def base_do_something_for(
    rng_key: chex.PRNGKey,
    num_sims: int,
):
    """
    Notice: we don't need to use Jax loops here, we just end up using them
    in many cases because they are way faster than standard Python loops.
    """
    def body_fn_fori(sim, loop_state):
        rng_key, count = loop_state
        rng_key, rand_key = jax.random.split(rng_key)
        count += jax.random.randint(rand_key, (), 0, 10)
        # count += sim + 1 # Simple test for what 'sim' returns -> the loop index
        loop_state = (rng_key, count)
        # print(sim)
        return loop_state
    
    initial_loop_state = (rng_key, 0)
    _, final_count = jax.lax.fori_loop(0, num_sims, body_fn_fori, initial_loop_state)
    
    return final_count

# Vmap only on the first argument: the rng_keys.
@functools.partial(jax.vmap, in_axes=[0, None], out_axes=0)
def vectorised_do_something_for(
    rng_key: chex.PRNGKey,
    num_sims: int,
):
    def body_fn_fori(sim, loop_state):
        rng_key, count = loop_state
        rng_key, rand_key = jax.random.split(rng_key)
        # count += jax.random.randint(rand_key, (), 0, 10)
        count += sim + 1 # Simple test for what 'sim' returns -> the loop index
        loop_state = (rng_key, count)
        return loop_state
    
    initial_loop_state = (rng_key, 0)
    _, final_count = jax.lax.fori_loop(0, num_sims, body_fn_fori, initial_loop_state)
    
    return final_count

if __name__ == "__main__":
    # The first fori loop is just at the top level, non-vectorised.
    rng_key = jax.random.key(np.random.randint(np.iinfo(int).max))
    rng_key, base_key = jax.random.split(rng_key, 2)
    count = base_do_something_for(base_key, 10)
    del base_key
    print(f"Base fori loop final count: {count}")

    # Now we want a vectorised fori loop.
    rng_key, vec_key = jax.random.split(rng_key, 2)
    all_vec_keys = jax.random.split(vec_key, 10)
    counts = vectorised_do_something_for(all_vec_keys, 10)
    del all_vec_keys
    print(f"Vectorised fori loop final counts: {counts}")



    





