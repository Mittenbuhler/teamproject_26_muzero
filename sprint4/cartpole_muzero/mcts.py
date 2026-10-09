"""Vanilla latent MuZero MCTS: one recurrent inference per simulation."""
import math
import numpy as np


class MinMaxStats:
    def __init__(self): self.minimum, self.maximum = float("inf"), float("-inf")
    def update(self, value): self.minimum=min(self.minimum,value); self.maximum=max(self.maximum,value)
    def normalize(self, value):
        if self.maximum > self.minimum: return (value-self.minimum)/(self.maximum-self.minimum)
        return value


class MCTSNode:
    def __init__(self, prior=0.0, state=None, reward=0.0, parent=None, action=None):
        self.prior=float(prior); self.state=state; self.reward=float(reward)
        self.parent=parent; self.action=action; self.children={}; self.visit_count=0; self.value_sum=0.0
    @property
    def mean_value(self): return self.value_sum/self.visit_count if self.visit_count else 0.0
    # Paper notation, kept as properties so diagnostics can inspect P/N/W/Q/R.
    P=property(lambda self:self.prior)
    N=property(lambda self:self.visit_count)
    W=property(lambda self:self.value_sum)
    Q=property(lambda self:self.mean_value)
    R=property(lambda self:self.reward)
    latent_state=property(lambda self:self.state)
    def expanded(self): return bool(self.children)


class ModelBasedMCTS:
    def __init__(self, dynamics_model, prediction_network=None, action_dim=None, simulations=50,
                 discount=0.997, pb_c_base=19652, pb_c_init=1.25, root_dirichlet_alpha=0.25,
                 root_exploration_fraction=0.25, policy_network=None, value_network=None):
        if action_dim is None or int(action_dim) <= 0:
            raise ValueError("action_dim must be positive")
        if int(simulations) <= 0:
            raise ValueError("simulations must be positive")
        if not 0 <= float(discount) <= 1:
            raise ValueError("discount must be in [0, 1]")
        if float(pb_c_base) <= 0 or float(pb_c_init) <= 0:
            raise ValueError("PUCT parameters must be positive")
        if float(root_dirichlet_alpha) <= 0:
            raise ValueError("root_dirichlet_alpha must be positive")
        if not 0 <= float(root_exploration_fraction) <= 1:
            raise ValueError("root_exploration_fraction must be in [0, 1]")
        self.dynamics_model=dynamics_model
        self.prediction_network=prediction_network
        self.policy_network=policy_network; self.value_network=value_network
        self.action_dim=int(action_dim); self.simulations=int(simulations); self.discount=float(discount)
        self.pb_c_base=float(pb_c_base); self.pb_c_init=float(pb_c_init)
        self.root_dirichlet_alpha=float(root_dirichlet_alpha); self.root_exploration_fraction=float(root_exploration_fraction)
        self.expansions_last_search=0

    def _predict(self,state):
        if self.prediction_network is not None: return self.prediction_network.predict(state)
        return self.policy_network.action_probs(state), self.value_network.value(state)

    def _expand_priors(self,node,priors):
        priors=np.asarray(priors,dtype=np.float64); priors=priors/priors.sum()
        for action in range(self.action_dim): node.children[action]=MCTSNode(prior=priors[action],parent=node,action=action)

    def search(self, root_state, add_exploration_noise=False):
        priors, root_value=self._predict(root_state)
        root=MCTSNode(state=np.asarray(root_state,np.float32)); self._expand_priors(root,priors)
        if add_exploration_noise:
            noise=np.random.dirichlet([self.root_dirichlet_alpha]*self.action_dim)
            for a,child in root.children.items():
                child.prior=(1-self.root_exploration_fraction)*child.prior+self.root_exploration_fraction*noise[a]
        stats=MinMaxStats(); self.expansions_last_search=0
        for _ in range(self.simulations):
            node=root; path=[root]
            while node.expanded():
                _,node=self._select(node,stats); path.append(node)
                if node.state is None: break
            if node.state is None:
                node.state,node.reward=self.dynamics_model.predict(node.parent.state,node.action)
                leaf_priors,leaf_value=self._predict(node.state); self._expand_priors(node,leaf_priors)
                self.expansions_last_search+=1
            else: leaf_value=root_value
            self._backup(path,float(leaf_value),stats)
        return root

    def _select(self,node,stats):
        best=[]; best_score=-float("inf")
        for action,child in node.children.items():
            pb_c=(math.log((node.visit_count+self.pb_c_base+1)/self.pb_c_base)+self.pb_c_init)
            pb_c*=math.sqrt(max(node.visit_count,1))/(child.visit_count+1)
            q=child.reward+self.discount*child.mean_value if child.visit_count else 0.0
            score=stats.normalize(q)+pb_c*child.prior
            if score>best_score+1e-12: best=[(action,child)]; best_score=score
            elif abs(score-best_score)<=1e-12: best.append((action,child))
        return best[np.random.randint(len(best))]

    def _backup(self,path,value,stats):
        for node in reversed(path):
            node.value_sum+=value; node.visit_count+=1
            if node.parent is not None:
                value=node.reward+self.discount*value; stats.update(value)


def visit_count_policy(root,temperature=1.0):
    counts=np.asarray([root.children[a].visit_count for a in range(len(root.children))],np.float64)
    if counts.sum()==0: return np.ones(len(counts),np.float32)/len(counts)
    if temperature<=0:
        p=np.zeros(len(counts),np.float32); p[int(np.argmax(counts))]=1; return p
    counts=counts**(1/temperature); return (counts/counts.sum()).astype(np.float32)


def select_action(root,temperature=0.0):
    policy=visit_count_policy(root,temperature)
    action=int(np.argmax(policy)) if temperature<=0 else int(np.random.choice(len(policy),p=policy))
    return action,policy
