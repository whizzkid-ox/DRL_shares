# Author: Ryo Segawa (whizznihil.kid@gmail.com)

import yfinance as yf
import numpy as np
import pandas as pd
import gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

class TradingEnvironment:
    def __init__(self, stock_data, initial_balance=10_000_000):
        self.stock_data = stock_data
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.current_step = 0
        self.n_stocks = stock_data.shape[1]
        self.shares_held = np.zeros(self.n_stocks)
        self.total_shares_sold = np.zeros(self.n_stocks)
        self.total_sales_value = 0
        self.total_profit = 0

    def reset(self):
        self.balance = self.initial_balance
        self.current_step = 0
        self.shares_held = np.zeros(self.n_stocks)
        self.total_shares_sold = np.zeros(self.n_stocks)
        self.total_sales_value = 0
        self.total_profit = 0
        return self._get_observation()

    def _get_observation(self):
        return np.concatenate(([self.balance], self.shares_held, self.stock_data[self.current_step]))

    def step(self, action):
        self._take_action(action)
        self.current_step += 1
        if self.current_step >= len(self.stock_data):
            self.current_step = 0
        reward = self.total_profit
        done = self.balance <= 0
        obs = self._get_observation()
        return obs, reward, done

    def _take_action(self, action):
        action_type = action[0]
        amount = action[1]
        if action_type < 0:
            self._buy_stock(amount)
        elif action_type > 0:
            self._sell_stock(amount)
        else:
            pass

    def _buy_stock(self, amount):
        for i in range(self.n_stocks):
            stock_price = self.stock_data[self.current_step, i]
            max_buyable = self.balance // stock_price
            shares_bought = min(amount, max_buyable)
            self.balance -= shares_bought * stock_price
            self.shares_held[i] += shares_bought

    def _sell_stock(self, amount):
        for i in range(self.n_stocks):
            stock_price = self.stock_data[self.current_step, i]
            shares_sold = min(amount, self.shares_held[i])
            self.balance += shares_sold * stock_price
            self.shares_held[i] -= shares_sold
            self.total_shares_sold[i] += shares_sold
            self.total_sales_value += shares_sold * stock_price
            self.total_profit = self.total_sales_value - self.initial_balance

class StockTradingEnv(gym.Env):
    def __init__(self, stock_data):
        super(StockTradingEnv, self).__init__()
        self.env = TradingEnvironment(stock_data)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.env.n_stocks*2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(self.env.n_stocks*2+1,), dtype=np.float32)
        
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        obs, reward, done = self.env.step(action)
        return obs, reward, done, {}
    
    def render(self, mode='human', close=False):
        pass

def get_historical_data(tickers, start_date, end_date):
    stock_data = []
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        stock_data.append(data['Close'].values)
    stock_data = np.array(stock_data).T
    return stock_data

# Data collection: Fetch historical data
tickers = ['7203.T', '6758.T', '9432.T']  # Example: Toyota, Sony, and NTT Docomo
start_date = '2020-01-01'
end_date = '2023-01-01'
stock_data = get_historical_data(tickers, start_date, end_date)

# Create and wrap the environment
env = DummyVecEnv([lambda: StockTradingEnv(stock_data)])

# Define and train the model
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Save the model
model.save("sac_stock_trading")

# Load the model
model = SAC.load("sac_stock_trading")

# Evaluate the model
obs = env.reset()
for _ in range(1000):  # Number of test steps
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if dones:
        obs = env.reset()