##### Loss function #####


class LossFunc(nn.Module):
    '''Cost function
    Args:
        sf_star: reference final range
    '''
    def __init__(self, sf_star):
        super().__init__()
        self.sf_star = sf_star
        
    def forward(self, sf_pred):
        """
        sf_pred: predicted final range
        """
        cost = 0.5 * (sf_pred - self.sf_star)**2
        
        return cost

