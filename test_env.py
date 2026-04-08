import sys
from env.environment import DataCleaningEnv
from env.models import Action

def test():
    env = DataCleaningEnv("easy_cleaning", max_steps=3)
    obs = env.reset()
    print("Initial observation:", obs)
    
    actions = [
        Action(action_type="drop_nulls"),
        Action(action_type="remove_duplicates"),
        Action(action_type="drop_nulls")
    ]
    
    for i, act in enumerate(actions):
        obs, rew, done, info = env.step(act)
        print(f"Step {i+1} Reward: {rew:.2f} Done: {done} Info: {info}")

if __name__ == "__main__":
    test()
