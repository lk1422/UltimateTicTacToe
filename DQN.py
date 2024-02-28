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
        self.optim = torch.optim.Adam(self.Net.parameters(), lr=3e-6)
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

        batch:List[Tuple] = self.mem.getBatch()

        #add new tensors to batch
        batch.append( (state, action, state2, reward, terminal) )

        #Set up models
        self.Net.train()
        self.Target.eval()

        total_loss = 0
        for b in batch:
            #Calculate Q(s,a), max Q(s', a')
            s  = b[0].to(self.device)
            s1 = b[2].to(self.device)
            net_out = self.Net(s)
            targ_out = self.Target(s1)

            #get target action and target val
            target_val, _    = torch.max(targ_out, dim=0)
            target_val = torch.zeros((1,1)).to(self.device) \
                         if b[4] else target_val
            target_val    = b[3] + self.gamma * target_val
            
            #Get model output
            one_hot = torch.zeros((self.n_actions, 1)).to(self.device)
            one_hot[b[1] , 0] = 1
            out = torch.matmul(net_out, one_hot).unsqueeze(0)

            #Calculate Loss
            self.optim.zero_grad()
            target_val = target_val.reshape(1)
            out = out.reshape(1)
            l = self.loss(out, target_val)
            l.backward()
            total_loss+= l.item()
            #print(l.item())

            #Optimize
            self.optim.step()

        #Add new event to memories
        self.mem.addMemory(state, action, state2, reward, terminal)
        #print("BATCH LOSS:", total_loss/len(batch))

    def updateWeights(self):
            self.Target.load_state_dict(self.Net.state_dict())


    def generate_action(self, state:torch.Tensor, legal_actions : torch.Tensor) -> int:
        mask = torch.zeros(self.n_actions)
        legal_actions = legal_actions.to(torch.long)
        mask[legal_actions] = 1
        #self.Net.eval()
        state = state.to(self.device)
        vals = self.Target(state)
        negated_mask = torch.abs(1-mask)
        vals = negated_mask*(vals*0 - 100) + mask*vals
        a = int(torch.argmax(vals, dim=0).item())
        return a
        
