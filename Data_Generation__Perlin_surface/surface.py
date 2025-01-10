from image import Image

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.interpolate
from perlin_numpy import generate_perlin_noise_2d


class Surface(Image):
    """Image subclass for two-dimensional surfaces"""
    
    def __init__(self, size_x: int, size_y: int):
        """
        2D surface stored as a numpy array with additional methods.
        
        Parameters
        ----------
        size_x: int
            The size of the x dimension (height) of the surface in pixels.
        size_y: int
            The size of the y dimension (width) of the surface in pixels.
        """
        super().__init__(size_x, size_y)
        
        # Create meshgrid
        x = np.arange(0, size_x)
        y = np.arange(0, size_y)
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        
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
        """Alias for array: the numpy array associated to this surface."""
        return self.array
    
    @Z.setter
    def Z(self, array: np.ndarray):
        self.array = array
    
    def plot(self, z_max: int|float = 1, show: bool = True):
        """
        Create a surface plot of the surface
        
        Parameters
        ----------
        z_max: int or float, defaults to 1
            The upper limit of the z axis.
        show: bool, defaults to True
            Whether to show the plot
            
        Returns
        -------
            fig, ax : Figure, Axes (see `plt.subplots()`)
        """
        
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(self.X, self.Y, self.Z, cmap=cm.Blues)
        
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        
        ax.axis('equal')
        ax.set_zlim(0, z_max)
        
        # if type(self) == RandomInterpol:
        #     ax.scatter(self.x, self.y, self.z, marker="^", c="orange")
        
        if show:
            fig.show()
            
        return fig, ax
    
    def interpolate_sample(self, n_points: int = 30, method: str = "linear"):
        """Extract random points from the surface and create a new surface by linear interpolation"""
        # Pick random points from the surface
        sx, sy = self.size_x, self.size_y
        x = np.append(np.random.randint(sx//16, sx-sx//16, size=n_points-4), [0, 0, sx-1, sx-1])
        y = np.append(np.random.randint(sy//16, sy-sy//16, size=n_points-4), [0, sy-1, 0, sy-1])
        z = self.Z[x, y]
        # Apply linear interpolation between the chosen points
        new_Z = scipy.interpolate.griddata((x,y), z, (self.X, self.Y), method=method)
        return new_Z
            
            
class Slope(Surface):
    
    def __init__(self, size_x: int, size_y: int, amp_x: float = 1, amp_y: float = 1):
        super().__init__(size_x, size_y)
        self.Z = amp_x*(size_x/2 - self.X) + amp_y*(size_y/2 - self.Y)
        self.normalise()
        
            
class Paraboloid(Surface):
    
    def __init__(self, size_x: int, size_y: int):
        super().__init__(size_x, size_y)
        self.Z = -(size_x/2 - self.X)**2 - (size_y/2 - self.Y)**2
        self.normalise()
        
        
class Pringles(Surface):
    
    def __init__(self, size_x: int, size_y: int):
        super().__init__(size_x, size_y)
        self.Z = (size_x/2 - self.X)**2 - (size_y/2 - self.Y)**2
        self.normalise()
        
        
class Cone(Surface):
    
    def __init__(self, size_x: int, size_y: int):
        super().__init__(size_x, size_y)
        self.Z = -np.sqrt((size_x/2 - self.X)**2 + (size_y/2 - self.Y)**2)
        self.normalise()
        
        
class Sphere(Surface):
    
    def __init__(self, size_x: int, size_y: int):
        super().__init__(size_x, size_y)
        R_sq = (size_x/2)**2 + (size_y/2)**2
        self.Z = np.sqrt(R_sq - (size_x/2 - self.X)**2 - (size_y/2 - self.Y)**2)
        self.normalise()
        
        
class Pyramid(Surface):
    
    def __init__(self, size_x: int, size_y: int):
        super().__init__(size_x, size_y)
        self.Z = size_x/2 - np.abs(size_x/2 - self.X) + size_y/2 - np.abs(size_y/2 - self.Y)
        self.normalise()
        
        
class Waves(Surface):
    
    def __init__(self, size_x: int, size_y: int, n_periods: float = 1):
        super().__init__(size_x, size_y)
        self.Z = - np.cos(2*np.pi*n_periods*self.Y/size_y) - np.cos(2*np.pi*n_periods*self.X/size_x)
        self.normalise()
        
        
class Jagged(Surface):
    
    def __init__(self, size_x: int, size_y: int, n_periods: float = 1):
        super().__init__(size_x, size_y)
        self.Z = np.arccos(np.cos(2*np.pi*n_periods*self.X/size_x))
        self.normalise()

      
class Pyramids(Surface):
    
    def __init__(self, size_x: int, size_y: int, n_periods: float = 1):
        super().__init__(size_x, size_y)
        self.Z = np.arccos(np.cos(2*np.pi*n_periods*self.X/size_x)) + np.arccos(np.cos(2*np.pi*n_periods*self.Y/size_y))
        self.normalise()

class Frustum(Surface):
    
    def __init__(self, size_x: int, size_y):
        super().__init__(size_x, size_y)
        # Set the heigth of 12 points
        x = np.array([0, 0, 6, 6, 1, 1, 5, 5, 2, 2, 4, 4])*size_x//6
        y = np.array([0, 6, 0, 6, 1, 5, 1, 5, 2, 4, 2, 4])*size_y//6
        z = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
        # Apply linear interpolation
        self.Z = scipy.interpolate.griddata((x,y), z, (self.X, self.Y), method="linear")
        
class Perlin(Surface):
    """Create a randomised surface using Perlin noise"""
    
    def __init__(self, size_x: int, size_y: int, freq: int|tuple[int] = 2, linear_segments: bool = False):
        # Initialize this class with a meshgrid in self.X, self.Y
        super().__init__(size_x, size_y) 
        freq_x, freq_y = (freq, freq) if not isinstance(freq, tuple) else freq
        
        # Enlarge the grid and make sure the sizes are divisible by the frequencies
        shape_x = int((size_x*3/2)//freq_x)*freq_x
        shape_y = int((size_y*3/2)//freq_y)*freq_y
        
        # Generate perlin noise
        perlin_noise = generate_perlin_noise_2d((shape_x, shape_y), res = (freq_x, freq_y))
        
        # Crop the generated perlin noise randomly to fit the original grid
        x0 = np.random.randint(shape_x-size_x)
        y0 = np.random.randint(shape_y-size_y)
        self.array = perlin_noise[x0:x0+size_x, y0:y0+size_y]
        
        if linear_segments:
            # Copy, rotate 180° and invert (z --> -z) the surface
            arr2 = -self.array[::-1,::-1]
            # Extract random points of the copy and apply linear interpolation
            arr1 = self.interpolate_sample(n_points=np.random.randint(20, 60), method='linear')
            # Combine both surfaces into one
            self.array = np.maximum(arr1, arr2)
        
      
class RandomGauss(Surface):
    """Create a randomised surface using Gaussian curves"""
    
    def __init__(self, size_x, size_y, n_points=20, seed=None):
        super().__init__(size_x, size_y)
        
        # Set seed
        np.random.seed(seed)
        
        # Generate random coordinates
        x = np.random.random(n_points)*size_x
        y = np.random.random(n_points)*size_y
        
        # Generate random parameters
        a = np.random.random(n_points)*2-1
        b = (np.random.random(n_points)+1)*max(size_x, size_y)/10
        
        # Reshape coordinates and parameters to a 3d-array
        A = a[:, np.newaxis, np.newaxis]
        X = x[:, np.newaxis, np.newaxis]
        B = b[:, np.newaxis, np.newaxis]
        Y = y[:, np.newaxis, np.newaxis]
        
        # Calculate the surface
        self.Z = np.sum(A*np.exp(-((self.X-X)/B)**2 - ((self.Y-Y)/B)**2), axis=0)
        
        # Redistribute the surface between two random numbers
        new_min = np.random.random()/2
        new_max = np.random.random()/2+1/2
        self.redistribute(new_min, new_max, inplace=True)
        
        # Reset seed
        np.random.seed()
        
class RandomInterpol(Surface):
    """Create a randomised surface using interpolation of random points"""
    
    def __init__(self, size_x, size_y, n_points=12,
                 seed: int|float = None, method: str = None):
        super().__init__(size_x, size_y)
        
        # Set seed
        np.random.seed(seed)
        
        # Generate random point coordinates and include corners
        self.x = x = np.append(np.random.randint(size_x, size=n_points-4), [0, 0, size_x-1, size_x-1])
        self.y = y = np.append(np.random.randint(size_y, size=n_points-4), [0, size_y-1, 0, size_y-1])
        # self.x = x = np.append(np.random.choice(np.arange(5, size_x-5, 5), size=n_points-4), [0, 0, size_x-1, size_x-1])
        # self.y = y = np.append(np.random.choice(np.arange(5, size_y-5, 5), size=n_points-4), [0, size_y-1, 0, size_y-1])
        self.z = z = np.random.random(n_points)
        
        # Calculate the interpolated surface
        if method == "rbf":
            # Reshape the grid to 1D arrays
            grid_points = np.column_stack((self.X.flatten(), self.Y.flatten()))

            # Create RBFInterpolator
            rbf_interpolator = scipy.interpolate.RBFInterpolator(np.column_stack((x, y)), z)

            # Interpolate values on the output grid
            Z = rbf_interpolator(grid_points)

            # Reshape the result back to a 2D array
            self.Z = Z.reshape(self.X.shape)
            
        else:        
            self.Z = scipy.interpolate.griddata((x,y), z, (self.X, self.Y), method=method)
            # sigmoid = lambda z: 1/(1+np.exp(-z))
            # self.Z = sigmoid(self.Z)
        
        new_min=z.min()
        new_max=z.max()
        self.z = (z-self.Z.min())/(self.Z.max()-self.Z.min())*(new_max-new_min)+new_min
        self.redistribute(new_min, new_max)
        
        
        # Reset seed
        np.random.seed()


            
        