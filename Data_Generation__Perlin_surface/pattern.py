from image import Image

import numpy as np


class Pattern(Image):
    """Image subclass for grayscale patterns"""
    
    def __init__(self, size_x: int, size_y: int, Xcoords = None, Ycoords = None, *args, **kwargs):
        """
        Grayscale pattern stored as a numpy array with additional methods.
        
        Parameters
        ----------
        size_x: int
            The size of the x dimension (height) of the pattern in pixels.
        size_y: int
            The size of the y dimension (width) of the pattern in pixels.
        """
        if not isinstance(size_x, int) or not isinstance(size_y, int):
            raise TypeError("Both input parameters should be integers.")
        
        super().__init__(size_x, size_y)
        x = np.arange(0, size_x)
        y = np.arange(0, size_y)
        X, Y = np.meshgrid(x, y, indexing='ij')
        self.X = Xcoords if Xcoords is not None else X
        self.Y = Ycoords if Ycoords is not None else Y
        self.compute_Z(*args, **kwargs)
        
    @property
    def size_x(self):
        """Alias for height: the height of the image in pixels"""
        return self.height
    
    @property
    def size_y(self):
        """Alias for width: the width of the image in pixels"""
        return self.width
    
    @property
    def Z(self):
        """Alias for array: the numpy array associated to this image."""
        return self.array
    
    @Z.setter
    def Z(self, array: np.ndarray):
        self.array = array
        
    def compute_Z(self, *args, **kwargs):
        self.Z = np.zeros(self.shape)       

       
class Lines(Pattern):
    
    def compute_Z(self, n_periods: float, phase_rad: float = 0, phase_deg: float = 0):
        # Set phase in radians
        phase_rad = phase_rad or phase_deg/180*np.pi
        
        self.Z = np.cos(2*np.pi*n_periods*self.Y/self.size_y + phase_rad)
        self.normalise()
 
       
class LinesSaw(Pattern):
    
    def compute_Z(self, n_periods: float):
        self.Z = np.arccos(np.cos(2*np.pi*n_periods*self.Y/self.size_y))
        self.normalise()

      
class Dots(Pattern):
    
    def compute_Z(self, n_periods: float):
        self.Z = np.cos(2*np.pi*n_periods*self.X/self.size_x) + np.cos(2*np.pi*n_periods*self.Y/self.size_y)
        self.normalise()
        
        
class DotsSaw(Pattern):
    
    def compute_Z(self, n_periods: float):

        self.Z = -np.arccos(np.cos(2*np.pi*n_periods*self.X/self.size_x)) + -np.arccos(np.cos(2*np.pi*n_periods*self.Y/self.size_y))
        self.normalise()


class Checkers(Pattern):
    
    def compute_Z(self, n_periods: float):

        self.Z = np.cos(2*np.pi*n_periods*self.X/self.size_x) * np.cos(2*np.pi*n_periods*self.Y/self.size_y)
        self.normalise()
  
        
class CheckersSaw(Pattern):
    
    def compute_Z(self, n_periods: float):

        self.Z = np.arccos(np.cos(2*np.pi*n_periods*self.X/self.size_x) * np.cos(2*np.pi*n_periods*self.Y/self.size_y))
        self.normalise()


class Grating(Pattern):
    
    def compute_Z(self, n_periods: float):

        self.Z = np.max([
            np.cos(2*np.pi*n_periods*(self.X/self.size_x + self.Y/self.size_y)),
            np.cos(2*np.pi*n_periods*(self.X/self.size_x - self.Y/self.size_y))
            ], axis=0)
        self.normalise()
   
        
class GratingSaw(Pattern):
    
    def compute_Z(self, n_periods: float):

        self.Z = np.max([
            -np.arccos(np.cos(2*np.pi*n_periods*(self.X/self.size_x + self.Y/self.size_y))),
            -np.arccos(np.cos(2*np.pi*n_periods*(self.X/self.size_x - self.Y/self.size_y)))
            ], axis=0)
        self.normalise()
   
        
class Flowers(Pattern):
    
    def compute_Z(self, n_periods: float):

        self.Z = np.max([
            np.cos(2*np.pi*n_periods*self.Y/self.size_y),
            np.cos(2*np.pi*n_periods*(np.sqrt(3)/2*self.X/self.size_x + 1/2*self.Y/self.size_y)),
            np.cos(2*np.pi*n_periods*(np.sqrt(3)/2*self.X/self.size_x - 1/2*self.Y/self.size_y))
            ], axis=0)
        self.normalise()
   
        
class FlowersSaw(Pattern):
    
    def compute_Z(self, n_periods: float):

        self.Z = np.max([
            -np.arccos(np.cos(2*np.pi*n_periods*self.Y/self.size_y)),
            -np.arccos(np.cos(2*np.pi*n_periods*(np.sqrt(3)/2*self.X/self.size_x + 1/2*self.Y/self.size_y))),
            -np.arccos(np.cos(2*np.pi*n_periods*(np.sqrt(3)/2*self.X/self.size_x - 1/2*self.Y/self.size_y)))
            ], axis=0)
        self.normalise()
   
        
class Honeycomb(Pattern):
    
    def compute_Z(self, n_periods: float):

        self.Z = 1-np.product([
            (np.cos(2*np.pi*n_periods*self.Y/self.size_y)),
            (np.cos(2*np.pi*n_periods*(np.sqrt(3)/2*self.X/self.size_x + 1/2*self.Y/self.size_y))),
            (np.cos(2*np.pi*n_periods*(np.sqrt(3)/2*self.X/self.size_x - 1/2*self.Y/self.size_y)))
            ], axis=0)
        self.normalise()
        

class HoneycombSaw(Pattern):
    
    def compute_Z(self, n_periods: float):

        self.Z = 1-np.product([
            np.arccos(np.cos(2*np.pi*n_periods*self.Y/self.size_y)),
            np.arccos(np.cos(2*np.pi*n_periods*(np.sqrt(3)/2*self.X/self.size_x + 1/2*self.Y/self.size_y))),
            np.arccos(np.cos(2*np.pi*n_periods*(np.sqrt(3)/2*self.X/self.size_x - 1/2*self.Y/self.size_y)))
            ], axis=0)
        self.normalise()