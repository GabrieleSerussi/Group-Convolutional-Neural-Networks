import torch

# Implement the 90-degree rotation group (cyclic group) over a plane
class CyclicGroup():
    def __init__(self) -> None:
        super().__init__()
        self.order = len(self.elements())
    def elements(self):
        '''Returns the elements of the group
        @:return elements: torch.tensor of group elements
        '''
        return torch.arange(0, 4)
    def product(self, g,f):
        '''Returns the group product of a and b
        @:param g: first group element
        @:param f: second group element
        @:return out: composition of g with f
        '''
        return torch.remainder(g + f, 4)
    def inverse(self, g):
        '''Returns the inverse of a
        @:param g: group element to be inverted
        @:return out: inverse element of g
        '''
        return torch.remainder(-g, 4)
    def action(self, g, x, cat=True):
        '''Returns the group action of g on x
        @param g: group elements (tensor of group elements)
        @param x: input (shape [channels, height, width])
        '''
        if cat:
            out = torch.empty((len(g), *x.shape))
            for i,elem in enumerate(g.tolist()):
                out[i] = torch.rot90(x, elem, [-2,-1])
        else:
            out = []
            for elem in g.tolist():
                out.append(torch.rot90(x, elem, [-2,-1]))
        return out