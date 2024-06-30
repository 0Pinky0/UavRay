from gymnasium.wrappers import HumanRendering

import boat_env
import gymnasium as gym
from tests.cvae import CVAE
import torch

from tests.deep_q_net import DeepQNet

render = True
env = gym.make('BoatEnv', render_mode='rgb_array')
if render:
    env = HumanRendering(env)
obs, _ = env.reset()
if render:
    env.render()
done = False

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
seq_len = 20
hidden_size = 256
latent_size = 8
pred_model = CVAE(
    latent_size=latent_size,
    device=device,
)
pred_model_hidden_states = torch.zeros(20,
                                       hidden_size,
                                       device=device)
dqn = DeepQNet(predict_dim=latent_size)
q_values = dqn(x=torch.from_numpy(obs))
action = int(torch.argmax(q_values))

while not done:
    # action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action)
    history = torch.from_numpy(info['history']).type(torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        predict = pred_model.encoder.predict(
            history,
            pred_model_hidden_states,
            torch.tensor([seq_len])
        )[0]
    q_values = dqn(x=torch.from_numpy(obs), predict=predict)
    action = int(torch.argmax(q_values))
    print(f'History: \n{history}')
    print(f'Predict: \n{predict}')
    print('=' * 40)
    if render:
        env.render()
