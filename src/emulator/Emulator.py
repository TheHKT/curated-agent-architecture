class Emulator:
    def __init__(self, actualEnv, hypothesisDb, useLlm=True):
        self.actualEnv = actualEnv
        self.hypothesisDb = hypothesisDb
        self.useLlm = useLlm
        
    #TODO: How to know which api to make? -> for each env one emulator would be the easiest