import torch
from torch import nn

__all__ = [
    'ImageTensor'
]
torch.tensor
class ImageTensor(torch.Tensor):
    torch.Tensor()
    def __init__(self, name='',*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__name__ = name
        self = self.fft()

    def fft(self):
        F1 = torch.fft.fft2(self)
        F2 = torch.fft.fftshift(F1)
        return F2
    
    def plot(self):
        import matplotlib.pyplot as plt
        plt.imshow(self.fourier.abs().log1p().numpy())
        plt.show()
    