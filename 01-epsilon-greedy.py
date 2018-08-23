import random

class EpsilonGreedy(object):
    def __init__(self, epsilon, n_arms):
        self.epsilon = epsilon
        self.initialize(n_arms)

    def initialize(self, n_arms):
        self.n_arms = n_arms
        self.counts = [0 for _ in range(n_arms)]
        self.scores = [0 for _ in range(n_arms)]
        return

    def select_arm(self):
        if(random.random()) > self.epsilon:
            # Exploit
            v = self.scores
            return v.index(max(v))
        else:
            # Explore
            return random.randrange(len(self.scores))

    def update(self, chosen_arm,  reward):
        updated_count = self.counts[chosen_arm] + 1
        total_score_chosen_arm = self.scores[chosen_arm] * self.counts[chosen_arm] + reward
        self.scores[chosen_arm] = total_score_chosen_arm / updated_count
        return

class BernoulliArm(object):
    def __init__(self, p):
        self.p = p

    def draw(self):
        if random.random() > self.p:
            return 0
        else:
            return 1

def test_algorithm(algo, arms, num_sims, horizon):
    chosen_arms = [0.0 for i in range(num_sims * horizon)] 
    rewards = [0.0 for i in range(num_sims * horizon)] 
    cumulative_rewards = [0.0 for i in range(num_sims * horizon)] 
    sim_nums = [0.0 for i in range(num_sims * horizon)] 
    times = [0.0 for i in range(num_sims * horizon)]
    
    for sim in range(num_sims): 
        sim = sim + 1 
        algo.initialize(len(arms))
        
        for t in range(horizon): 
            t = t + 1 
            index = (sim - 1) * horizon + t - 1

            sim_nums[index] = sim 
            times[index] = t
            
            chosen_arm = algo.select_arm() 
            chosen_arms[index] = chosen_arm
            
            reward = arms[chosen_arms[index]].draw() 
            rewards[index] = reward
            
            if t == 1: 
                cumulative_rewards[index] = reward 
            else: 
                cumulative_rewards[index] = cumulative_rewards[index - 1] + reward 
            algo.update(chosen_arm, reward) 
    return [sim_nums, times, chosen_arms, rewards, cumulative_rewards]


def test_algorithm2(algo, arms, num_sims, horizon):
    n_episodes = num_sims * horizon

    chosen_arms = [0.0 for i in range(n_episodes)] 
    rewards = [0.0 for i in range(n_episodes)] 
    cumulative_rewards = [0.0 for i in range(n_episodes)] 
    sim_nums = [0.0 for i in range(n_episodes)] 
    times = [0.0 for i in range(n_episodes)]
    
    for sim in range(num_sims): 
        algo.initialize(len(arms))

        for t in range(horizon): 
            index = sim * horizon + t

            sim_nums[index] = sim + 1
            times[index] = t + 1
            
            chosen_arms[index] = algo.select_arm() 
            
            rewards[index] = arms[chosen_arm].draw() 
            
            if t == 1: 
                cumulative_rewards[index] = reward 
            else: 
                cumulative_rewards[index] = cumulative_rewards[index - 1] + reward 
            algo.update(chosen_arm, reward) 
    return [sim_nums, times, chosen_arms, rewards, cumulative_rewards]


means = [0.1, 0.1, 0.1, 0.1, 0.9]
random.shuffle(means)
arms = [BernoulliArm(mu) for mu in means]
n_arms = len(means)
print("Best arm is ", means.index(max(means)))

f = open("standard_results.tsv", "w")

for epsilon in [0.1, 0.2, 0.3, 0.4, 0.5]:
    algo = EpsilonGreedy(epsilon, 1)
    algo.initialize(n_arms)
    results = test_algorithm(algo, arms, 5000, 250)
    for i in range(len(results[0])):
        f.write(str(epsilon) + "\t") 
        f.write("\t".join([str(results[j][i]) for j in range(len(results))]) + "\n")
f.close()

# algo = EpsilonGreedy(0.1, 1)
# algo.initialize(n_arms)
# from pprint import pprint
# results = test_algorithm(algo, arms, 1000, 200)
# banners = ['sim_nums', 'times', 'chosen_arms', 'rewards', 'cumulative_rewards']
# for b, i in zip(banners, results):
#     print(b)
#     print(i)
#     print('\n\n')
# print(results[-1])