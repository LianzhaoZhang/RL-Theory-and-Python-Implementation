class BeSpokerAgent():
    def __init__(self):
        pass

    def decide(self, observation):
        position,  volicity  =  observation
        lb = min( -0.09* (position+ 0.25)** 2+ 0.03,
                        0.3* (position + 0.9) ** 4 - 0.008)
        ub = -0.07*(position+0.38)**2+0.06
        if lb<volicity<ub:
            action=2
        else:
            action=0
        return action

    def learn(delf,*args):
        pass

