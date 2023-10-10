import jax
import jax.numpy as jnp

from sim import DEFAULT_TX_POWER, init_static_network


if __name__ == '__main__':
    # Example network topology:
    # ----------------------
    # |  0 -> 1    2 <- 3  |
    # ----------------------

    # Position of the nodes given by X and Y coordinates
    pos = jnp.array([
        [0., 0.],
        [10., 0.],
        [20., 0.],
        [30., 0.]
    ])

    # Transmission matrix indicating which node is transmitting to which node
    # In this example, node 0 is transmitting to node 1 and node 2 is transmitting to node 3
    tx = jnp.array([
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0]
    ])

    # Modulation and coding scheme of the nodes
    mcs = jnp.ones(4, dtype=jnp.int32) * 4

    # Transmission power of the nodes
    tx_power = jnp.ones(4) * DEFAULT_TX_POWER

    # Standard deviation of the additive white Gaussian noise
    noise = 1.

    # JAX random number generator key
    key = jax.random.PRNGKey(42)

    # Initialize the function with the given parameters
    throughput_fn = init_static_network(pos, mcs, tx_power, noise)

    # Calculate the network throughput
    throughput = throughput_fn(key, tx)
    print(throughput)
