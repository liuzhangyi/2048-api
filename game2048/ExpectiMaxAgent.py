from game2048.agents import Agent

class OwnAgent(Agent):
    '''Agent Base.'''

    def step(self):
        direction = np.random.randint(0, 4)
        return direction

