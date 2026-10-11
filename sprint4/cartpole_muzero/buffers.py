"""Episode replay and correctly aligned vanilla MuZero targets."""
from collections import deque
import random
import numpy as np
import torch


class LatentReplayBuffer:
    def __init__(self, capacity=20000):
        self.capacity, self.episodes, self.size = int(capacity), deque(), 0

    def add_episode(self, transitions):
        episode = []
        for item in transitions:
            episode.append({
                "observation": np.asarray(item["observation"], np.float32),
                "action": int(item["action"]), "reward": float(item["reward"]),
                "policy": np.asarray(item["policy"], np.float32), "value": float(item["value"]),
                "policy_valid": bool(item.get("policy_valid", True)),
                "terminated": bool(item.get("terminated", False)), "truncated": bool(item.get("truncated", False)),
            })
        if not episode: return
        self.episodes.append(tuple(episode)); self.size += len(episode)
        while self.size > self.capacity and len(self.episodes) > 1:
            self.size -= len(self.episodes.popleft())

    def sample(self, batch_size, unroll_steps, action_dim, device=None, starts=None):
        """Return root targets at t and recurrent targets for transitions t..t+K-1.

        rewards[:,k] is r_(t+k+1); policies/values[:,k+1] are pi/z_(t+k+1).
        A transition's policy_valid flag masks only policy supervision; value and
        reward targets remain active during model-learning warm-up.
        """
        if not self.episodes: raise ValueError("Cannot sample an empty replay buffer")
        episodes = random.choices(list(self.episodes), weights=[len(e) for e in self.episodes], k=batch_size)
        starts = starts or [random.randrange(len(e)) for e in episodes]
        obs=[]; actions=np.zeros((batch_size,unroll_steps),np.int64)
        policies=np.zeros((batch_size,unroll_steps+1,action_dim),np.float32)
        values=np.zeros((batch_size,unroll_steps+1,1),np.float32)
        rewards=np.zeros((batch_size,unroll_steps,1),np.float32)
        policy_masks=np.zeros((batch_size,unroll_steps+1,1),np.float32)
        value_masks=np.ones_like(policy_masks); reward_masks=np.ones((batch_size,unroll_steps,1),np.float32)
        for b,(ep,start) in enumerate(zip(episodes,starts)):
            obs.append(ep[start]["observation"]); alive=True
            for k in range(unroll_steps+1):
                i=start+k
                if alive and i < len(ep):
                    policies[b,k]=ep[i]["policy"]; values[b,k,0]=ep[i]["value"]
                    policy_masks[b,k,0]=float(ep[i].get("policy_valid", True))
                if k < unroll_steps:
                    if alive and i < len(ep):
                        actions[b,k]=ep[i]["action"]; rewards[b,k,0]=ep[i]["reward"]
                        if ep[i]["terminated"] or ep[i]["truncated"]: alive=False
                    else:
                        actions[b,k]=0  # absorbing dummy action; reward/value stay zero
        arrays=(np.asarray(obs),actions,policies,values,rewards,policy_masks,value_masks,reward_masks)
        tensors=tuple(torch.as_tensor(a,device=device) for a in arrays)
        return tensors

    def state_dict(self):
        """Serializable replay state used by resumable training checkpoints."""
        return {
            "capacity": self.capacity,
            "episodes": list(self.episodes),
            "size": self.size,
        }

    def load_state_dict(self, state):
        self.capacity=int(state["capacity"])
        self.episodes=deque(tuple(episode) for episode in state["episodes"])
        self.size=int(state.get("size",sum(len(episode) for episode in self.episodes)))

    def resize(self, capacity):
        """Apply a requested capacity when resuming an existing replay buffer."""
        capacity=int(capacity)
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity=capacity
        while self.size > self.capacity and len(self.episodes) > 1:
            self.size-=len(self.episodes.popleft())

    def __len__(self): return self.size
