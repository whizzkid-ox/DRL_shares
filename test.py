# Author: Ryo Segawa (whizznihil.kid@gmail.com)

# Load the model
model = SAC.load("sac_stock_trading")

# Evaluate the model
obs = env.reset()
for _ in range(10000):  # Number of test steps
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if dones:
        obs = env.reset()