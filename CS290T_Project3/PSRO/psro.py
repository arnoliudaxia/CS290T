
from env import RPSEnv
from agent import Agent, tabular_Q
import numpy as np
import argparse, time, itertools
from tqdm import tqdm
from scipy.optimize import linprog

def solve_nash(R_matrix):
    """
    Input the payoff matrix R_matrix, Using linear programming to solve mixed Nash equilibrium in two player zero sum games.
    The output is a vector of length n, where n is the number of policys in the current policy pool. 
    Each value in the output vector is a probability, representing the probability of selecting the corresponding policy, and the sum of all values is 1.
    """
    A_ub = R_matrix
    D=A_ub.shape[0]
    b_ub = np.zeros(D)
    A_eq = np.zeros([D,D])
    b_eq = np.zeros(D)
    A_eq[0,:]=1
    b_eq[0]=1
    c=np.ones(D)

    re=linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0,1))

    nash_p = np.maximum(re.x,0) # just to make sure non-negative weights    

    return nash_p


# Evaluate the average reward of policy p1 on p2 through multiple simulations
def estimate_reward(env, num_episodes, p1, p2):
    R = 0  # Accumulated total reward
    for i in range(num_episodes):
        state, info = env.reset(opponent=p2, train=False)
        done = False
        while not done:
            action = p1.step(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            done = terminated or truncated
            if done:
                R += reward
    return R / num_episodes



# Evaluate the reward of each policy pi[i] in the policy pool for the Nash policy, and then average it
def exploitability_nash(env,nash_pi,pi,Ne=300):
    R = 0
    nash_agent = Agent(nash_pi)
    for i in tqdm(range(pi.shape[0]), desc="Computing exploitability",position=1,leave=False):
        R+=max(estimate_reward(env, Ne, Agent(pi[i]), Agent(nash_pi)),0)
    return R/pi.shape[0]


# Construct the payoff matrix R for the policy pool
def gamescape(env, pi, Ne): 
    R = np.zeros([len(pi), len(pi)])      
    """
    For each policy pi[i], evaluate its reward against all other policies in the pool, and construct the payoff matrix R.
    R [i, j] is the reward of policy i for policy j. (Note R should be an antisymmetric matrix with all diagonal elements being 0)
    You may use estimate_reward(env,Ne,Agent(pi[i]),Agent(pi[j]) to get R[i,j].
    """
    for i in tqdm(range(len(pi)), desc="Computing payoff matrix", position=1, leave=False):
        for j in range(len(pi)):
            if i != j:  # Skip diagonal elements (they should be 0)
                R[i, j] = estimate_reward(env, Ne, Agent(pi[i]), Agent(pi[j]))
                # Ensure antisymmetry: R[i,j] = -R[j,i]
                R[j, i] = -R[i, j]
    
    return R



# PSRO algorithm framework
def PSRO_Q(env, num_iters=1000, num_steps_per_iter = 10000, eps=0.1, alpha=0.1, save_interval=1, evaluation_episodes=1000):

    # initialize a list as policys pool, the first policy is [0.1, 0.2, 0.7] (only needs to focus on the first row)
    pi =  np.array([[[0.1, 0.2, 0.7],[1., 0., 0.],[0., 0., 1.]]])

    expls = [1]
    divs = [0]
    pbar = tqdm(range(1,num_iters+1), desc="Iter", position=0)
    for niter in pbar:
        # compute nash
        R = gamescape(env, pi, evaluation_episodes)

        nash_p = solve_nash(R)

        # eval exploitability
        nash_pi = nash_p.reshape(-1,1,1)*pi  
        nash_pi = nash_pi.sum(0) # The current Nash equilibrium policy


        expl = exploitability_nash(env, nash_pi, pi, Ne=evaluation_episodes)
        div = (nash_p.reshape(1,-1)@np.maximum(R,0)@nash_p.reshape(-1,1))[0,0]


        # train a new agent by Q-learning, The opponent adopts the current Nash equilibrium policy
        Q = np.random.randn(env.observation_space.n,env.action_space.n)*1e-2  # reset Q   
        Q[-env.n_ternimal:] = 0 # terminal states to 0
        env.reset(opponent=Agent(nash_pi), train=True)
        """
        You may use tabular_Q() to get the Q-table of the policy trained, and then process this Q-table into the form of a policy we used. 
        The policy you got needs to be named as "beta".
        """
        ############################
        # YOUR IMPLEMENTATION HERE #
        ############################

        # Use tabular_Q to get the Q-table of the policy trained against the Nash equilibrium
        Q = tabular_Q(env, num_steps_per_iter, Q=Q, discount=0.9, epsilon=eps, alpha=alpha)

        # Convert Q-values to a policy (probabilities)
        # We only need to focus on the first row (ongoing state 0)
        beta = np.zeros_like(pi[0])  # Same shape as existing policies
        # For the ongoing state (state 0), use softmax to convert Q-values to probabilities
        q_vals = Q[0]
        exp_q = np.exp(q_vals - np.max(q_vals))  # Subtract max for numerical stability
        beta[0] = exp_q / np.sum(exp_q)
        # For terminal states, keep the same structure as other policies
        beta[1:] = pi[0][1:]
                

        # check criteria for early stopping
        stop=0
        for pi_i in pi:
            if (pi_i == beta).all():
                print("policy exhausted, early stopping")
                stop=1
                break
        if stop:
            break

        # append policy to policys pool
        pi = np.concatenate([pi,np.expand_dims(beta,0)],0)

        desc = f"expl={round(expl,4)}, div={round(div,4)}, nash={nash_pi[0]}| Iter"
        
        pbar.set_description(desc)
        pbar.refresh()

        # save data
        if niter%save_interval==0:
            expls.append(expl)
            divs.append(div)


    data = {
        "nash":nash_pi,
        "pi":pi,
        "R":R,
        "expl":expls,
        "div":divs
    }
    return data


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help="set seed", default=1)
    parser.add_argument('--model_file', type=str, help="filename of the model to be saved", default="Qh.npy")
    parser.add_argument('--num_iters', type=int, help="number of total training iterations", default=20)
    parser.add_argument('--num_steps_per_iter', type=int, help="number of training steps for each iteration", default=100)

    parser.add_argument('--step_size', type=int, help="learning rate alpha", default=0.1)
    parser.add_argument('--eps', type=float, help="hyperparameter epsilon for epsilon greedy policy", default=0.1)

    args = parser.parse_args()
    
    if not args.seed:
        args.seed = int(time.time())
    np.random.seed(args.seed)
    print("running with seed", args.seed)
    env = RPSEnv()

    print("args:",args)

    print("Training...")
    start = time.time()
    data = PSRO_Q(env, num_iters=args.num_iters, num_steps_per_iter = args.num_steps_per_iter, eps=args.eps, alpha=args.step_size)

    np.save(args.model_file, data)

    print("Final policy: ", data["nash"][0])
    print("Training complete, model saved at {}, elapsed {}s".format(args.model_file,round(time.time()-start,2)))