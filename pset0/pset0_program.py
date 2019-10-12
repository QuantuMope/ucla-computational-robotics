import numpy as np
import matplotlib.pyplot as plt

def coin_simulate(num_of_flips, coin_type):
    sequence = []
    if coin_type == "fair":
        sequence = ["H" if np.random.uniform() < 0.5 else "T" for _ in range(num_of_flips)]
    elif coin_type == "bias":
        sequence = ["H" if np.random.uniform() < 0.25 else "T" for _ in range(num_of_flips)]
    return sequence

for i in range(5):
    print("======== 40 Flip Sequence {} ========".format(i+1))
    print("============= Bias ==================")
    print(coin_simulate(40, "bias"))
    print("============= Fair ==================")
    print(coin_simulate(40, "fair"))
    print("\n")


def coin_simulate_with_likelihood(num_of_flips, coin_type="fair"):
    """
        probability of choosing biased coin is 1/4
        probability of choosing fair coin is 3/4
        for bias:
        choosing head is 1/4
        choosing tails is 3/4
    """
    sequence = []
    bias = [0.25] # starts at 0 flips
    fair = [0.75]
    p_h = 0.5 # probability of heads
    if coin_type == "bias": p_h = 0.25
    for i in range(num_of_flips):
        if np.random.uniform() < p_h:
            sequence.append("H")
            bias.append(bias[i]*p_h)
            fair.append(fair[i]*0.50)
        else:
            sequence.append("T")
            bias.append(bias[i]*(1-p_h))
            fair.append(fair[i]*0.50)

    likelihood_bias = [bias[i]/(bias[i]+fair[i]) for i in range(len(bias))]
    return sequence, likelihood_bias


plt.figure(1)
flips = [i for i in range(101)]
for _ in range(5):
    coin_sequence, bias_prob = coin_simulate_with_likelihood(100, "fair")
    plt.plot(flips, bias_prob)
plt.title("Evolving Bias Probability with Fair Coin")
plt.ylabel("Probability that coin is biased")
plt.xlabel("Number of fair coin flips")

plt.figure(2)
for _ in range(5):
    coin_sequence, bias_prob = coin_simulate_with_likelihood(100, "bias")
    plt.plot(flips, bias_prob)
plt.title("Evolving Bias Probability with Bias Coin")
plt.ylabel("Probability that coin is biased")
plt.xlabel("Number of bias coin flips")

plt.show()
