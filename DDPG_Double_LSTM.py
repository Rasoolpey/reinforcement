class Actor(nn.Module):
    def __init__(self, state_dim1, state_dim2, hidden_dim1, hidden_dim2, action_dim):
        super(Actor, self).__init__()
        self.lstm1 = nn.LSTM(state_dim1, hidden_dim1, batch_first=True)
        self.lstm2 = nn.LSTM(state_dim2, hidden_dim2, batch_first=True)
        self.fc = nn.Linear(hidden_dim1 + hidden_dim2, action_dim)

    def forward(self, state1, state2):
        _, (h_n1, _) = self.lstm1(state1)
        _, (h_n2, _) = self.lstm2(state2)
        x = torch.cat((h_n1[-1], h_n2[-1]), dim=1)
        x = self.fc(x)
        action = torch.sigmoid(x)  # Apply sigmoid activation
        return action

class Critic(nn.Module):
    def __init__(self, state_dim1, state_dim2, hidden_dim1, hidden_dim2):
        super(Critic, self).__init__()
        self.lstm1 = nn.LSTM(state_dim1, hidden_dim1, batch_first=True)
        self.lstm2 = nn.LSTM(state_dim2, hidden_dim2, batch_first=True)
        self.fc = nn.Linear(hidden_dim1 + hidden_dim2, 1)

    def forward(self, state1, state2):
        _, (h_n1, _) = self.lstm1(state1)
        _, (h_n2, _) = self.lstm2(state2)
        x = torch.cat((h_n1[-1], h_n2[-1]), dim=1)
        q_value = self.fc(x)
        return q_value