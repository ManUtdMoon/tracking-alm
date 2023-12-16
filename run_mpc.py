from new_env.pyth_tracking_env import *
from new_env.pyth_tracking_mpc import *


seed = 0
env = PathTrackingEnv(pre_horizon=25, max_episode_steps=200)
policy = ModelPredictiveController(env, seed)

policy.tunable_para_unmapped = np.array([
    -128916, -85944, 1.06, 1.85, 1412, 1536.7, 
    1e-5, 1.0, 10., 30., 50., 60., 1.0]
)


# run the environment with the policy in a for loop
for i in range(1):
    state = env.reset()
    done = False
    while not done:
        action, _ = policy.get_action_alm(state)
        state, _, done, _ = env.step(action)
        env.render()

print("Done!")