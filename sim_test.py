import jax
import jax.numpy as jnp

import  matplotlib.pyplot as plt


from sim import DEFAULT_TX_POWER, init_static_network, MIN_SNRS


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
    # In this example, node 0 is transmitting to node 1 and node 3 is transmitting to node 2
    tx = jnp.array([
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0]
    ])

    rx = jnp.array([
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0]
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
    throughput_fn= init_static_network(pos, mcs, tx_power, noise)
    print("\n",throughput_fn)
    

    # Calculate the network throughput
    throughput, snr, sinr, success_probability,tx_nodes, snr_plus_interference,  expected_data_rate, interference = throughput_fn(key, tx, rx)
    print("\n",throughput)
    

    #plt.plot(sinr, jax.scipy.stats.norm.cdf(sinr, loc=MIN_SNRS[mcs], scale=2.))
    #plt.show()

    print("\n mcs:", mcs)
    print("\n sinr:", sinr)
    #print("\n sinr2:", jnp.power(10,sinr/10))
    sinr2=jnp.power(10,sinr/10)

    observation=jnp.linspace(-5,1,50)
    cdf_norm=jax.scipy.stats.norm.cdf(observation, loc=MIN_SNRS[mcs[0]], scale=2.)
    plt.plot(observation, cdf_norm)
    #plt.show()

    print("\n success probability:",jax.scipy.stats.norm.cdf(sinr[0],  loc=MIN_SNRS[mcs[0]], scale=2.))
    print("\n success probability:", success_probability)
    print("\n throughput:", throughput)
    #print("\n tx_nodes:", tx_nodes)
    print("\n snr_plus_interference:", snr_plus_interference)
    print("\n snr_interference:", interference)
    print("\n snr:", snr)
    print("\n  expected_data_rate:",  expected_data_rate)
    B=20
    print("\n throughput from Matlab [R]:", B * jnp.log2(1+sinr2))

