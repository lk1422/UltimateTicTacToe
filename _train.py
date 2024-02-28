import torch
import time
import random
from ultimate import UltimateTicTacToe
from DQN import DQN

def train(model, env, epochs, eps):
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
            eps *= 0.995
            eps = min(eps, 0.1)
        model.updateWeights()
        print("Eval:", model.Net(env.get_state()))
        print("GAME COMPLETE, winner:", env.winner)
        print("Resolutions:", env.resolutions)
        print("Epoch:", e, "Complete")
        env.print_board()
        torch.save(model.Net.state_dict(), "model2.pth")
    return model


def main():
    env = UltimateTicTacToe()
    d = DQN(91, 81, 64, 0.99)
    #d.Net.load_state_dict(torch.load("model.pth"))
    #d.Target.load_state_dict(torch.load("model.pth"))
    #env.debug_play_agent(d)

    EPOCHS = 100000
    EPS = 0.2
    d = train(d, env, EPOCHS, EPS)
    torch.save(d.Net.state_dict(), "model.pth")


if __name__ == "__main__":
    main()

