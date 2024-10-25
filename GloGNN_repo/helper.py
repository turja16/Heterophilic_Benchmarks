class NCDataset(object):
    def __init__(self, name):
        self.name = name
        self.graph = {}
        self.label = None