import torch
import torch.nn as nn
from typing import Tuple, List
from memory import MemoryBuffer


class DQN():
    
    def __init__(self, state_size: int, n_actions:int, batch_size: int , discount:float, device=None):

        #Initialize models
        self.Net = nn.Sequential(*[nn.Linear(state_size, state_size*2), nn.ReLU(), \
            nn.Linear(state_size*2, state_size*2), nn.ReLU(), \
            nn.Linear(state_size*2, state_size*2), nn.ReLU(), \
            nn.Linear(state_size*2, n_actions) ])

        self.Target = nn.Sequential(*[nn.Linear(state_size, state_size*2), nn.ReLU(), \
            nn.Linear(state_size*2, state_size*2), nn.ReLU(), \
            nn.Linear(state_size*2, state_size*2), nn.ReLU(), \
            nn.Linear(state_size*2, n_actions) ])
        """
        self.Net = nn.Sequential(*[nn.Linear(state_size, state_size*2), nn.ReLU(), \
            nn.Linear(state_size*2, n_actions)])
        self.Target = nn.Sequential(*[nn.Linear(state_size, state_size*2), nn.ReLU(), \
            nn.Linear(state_size*2, n_actions)])
        """

        #Initialize Loss & Optim
        self.optim = torch.optim.Adam(self.Net.parameters(), lr=3e-5)
        self.loss  = torch.nn.MSELoss()

        #Set up memory buff and step_count
        self.mem = MemoryBuffer(512, state_size, batch_size)

        #Save variables
        self.gamma = discount
        self.n_actions = n_actions 

        if device==None:
            self.device = torch.device('cpu')
        else:
            self.device = device

        #Move Networks to GPU
        self.Net    = self.Net.to(self.device)
        self.Target = self.Target.to(self.device)

    def Learn(self, state: torch.Tensor, action: int, state2: torch.Tensor, reward: int, terminal: int) -> None:

        #Move New Tensors to device

        S, A, S_, R, T = self.mem._getBatch()
        S = S.to(self.device)
        A = A.to(self.device)
        S_ = S_.to(self.device)
        R = R.to(self.device)
        T = T.to(self.device)
        self.mem.addMemory(state, action, state2, reward, terminal)
        if(S.shape[0] == 0): return
        #(State, Action, S_, R, T)
        #Set up models
        self.Net.train()
        self.Target.eval()
        non_terminal_mask = torch.abs(1-T)
        current_state_val = self.Net(S)
        next_state_val, _ = torch.max(self.Target(S_), dim=1)
        target_val = R + self.gamma*next_state_val*non_terminal_mask
        current_state_val = current_state_val[:, A][0]
        self.optim.zero_grad()
        l = self.loss(current_state_val , target_val)
        l.backward()
        self.optim.step()

        #Add new event to memories
        self.mem.addMemory(state, action, state2, reward, terminal)
        #print("BATCH LOSS:", total_loss/S.shape[0])

    def updateWeights(self):
            self.Target.load_state_dict(self.Net.state_dict())


    def generate_action(self, state:torch.Tensor, legal_actions : torch.Tensor, adv=False) -> int:
        mask = torch.zeros(self.n_actions).to(self.device)
        legal_actions = legal_actions.to(torch.long)
        mask[legal_actions] = 1
        state = state.to(self.device)
        self.Net.eval()
        vals = self.Net(state)
        self.Net.eval()
        negated_mask = torch.abs(1-mask)
        if not adv:
            vals = negated_mask*(vals*0 - float('inf')) + mask*vals
            return int(torch.argmax(vals, dim=0).item())
        else:
            vals = negated_mask*(vals*0 + float('inf')) + mask*vals
            return int(torch.argmin(vals, dim=0).item())
        
