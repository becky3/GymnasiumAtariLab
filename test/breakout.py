import gymnasium as gym

# Breakout環境を作成
env = gym.make("Breakout-v4", render_mode="human")

# 環境をリセット
observation, info = env.reset()

# サンプルエピソードを実行
for _ in range(1000):
    action = env.action_space.sample()  # ランダムなアクションを選択
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

# 環境を閉じる
env.close()
