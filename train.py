import torch
import time
import random
from ultimate import UltimateTicTacToe
from DQN import DQN

device = torch.device('cuda')

def train(model, env, epochs, eps, update_time):
    for e in range(epochs):
        env.reset()
        state = env.get_state()
        while(not env.game_over):
            actions = env.get_legal_actions()
            if random.random() < eps:
                action = int(random.choice(actions).item())
            else:
                action = model.generate_action(state, actions)
            #env.print_board()
            next_state, reward, terminal = env.step(model, action)
            model.Learn(state, action, next_state, reward, terminal)
        print("Eval:", model.Net(env.get_state().to(device)))
        print("GAME COMPLETE, winner:", env.winner)
        print("Resolutions:", env.resolutions)
        print("Epoch:", e, "Complete")
        env.print_board()
        if e%update_time == 0:
            model.updateWeights()
            eps *= 0.999
            eps = min(eps, 0.35)
        if e%1000 == 0:
            torch.save(model.Net.state_dict(), "model2.pth")
    return model


def main():
    env = UltimateTicTacToe()
    d = DQN(91, 81, 128, 0.99, device)
    #d.Net.load_state_dict(torch.load("model.pth"))
    #d.Target.load_state_dict(torch.load("model.pth"))
    #env.debug_play_agent(d)

    EPOCHS = 200000
    EPS = 0.5
    UPDATE_TIME = 25
    d = train(d, env, EPOCHS, EPS, UPDATE_TIME)
    torch.save(d.Net.state_dict(), "model.pth")



if __name__ == "__main__":
    main()

