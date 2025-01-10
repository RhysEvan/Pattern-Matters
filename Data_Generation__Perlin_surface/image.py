import numpy as np
import cv2
import copy
from typing import Union


class Image:
    """Grayscale Image stored as a numpy array with additional methods."""

    def __init__(self, height: int, width: int):
        """
        Grayscale Image stored as a numpy array with additional methods.
        
        Parameters
        ----------
        height: int
            The height of the image in pixels.
        width: int
            The width of the image in pixels.
        """
        if not isinstance(height, int) or not isinstance(width, int):
            raise TypeError("Both the height and width values should be integers.")
        
        self.array = np.zeros((height, width), dtype=np.float32)

    @property
    def height(self) -> int:
        """The height of the image in pixels."""
        return self.array.shape[0]

    @property
    def width(self) -> int:
        """The width of the image in pixels."""
        return self.array.shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        """The shape of the numpy array associated to this image."""
        return (self.height, self.width)

    @property
    def array(self) -> np.ndarray:
        """The numpy array associated to this image."""
        return self._array

    @array.setter
    def array(self, array: np.ndarray):
        assert isinstance(array, np.ndarray)
        self._array = array
    
    @classmethod
    def from_file(cls, filename: str):
        """Create a new Image instance from a file. Supports all formats supported by cv2."""
        if not isinstance(filename, str):
            raise TypeError("The filename must be a string.")
        array = cv2.imread(filename, flags=cv2.IMREAD_GRAYSCALE)
        img = cls(*array.shape)
        img.array = array
        return img

    @classmethod
    def from_array(cls, array: np.ndarray):
        """Create a new Image instance from a numpy array. The array must be twodimensional."""
        if not isinstance(array, np.ndarray):
            raise TypeError("The array must be a numpy array.")
        assert array.ndim == 2, "The array must be 2-dimensional."
        img = cls(*array.shape)
        img.array = array
        return img
     
    def normalise(self) -> None:
        """Normalise the image to an interval between 0 and 1."""
        min = self.array.min()
        max = self.array.max()
        self.array = (self.array-min)/(max-min)
        
    def redistribute(self, new_min: int|float, new_max: int|float) -> None:
        """Redistribute the image to an interval between new_min and new_max."""
        min = self.array.min()
        max = self.array.max()
        self.array = (self.array-min)/(max-min)*(new_max-new_min)+new_min
        
    def resize(self, height: int = 512, width: int = 512):
        """Returns a new image resized to the given dimensions"""
        new = Image(height, width)
        new.array = cv2.resize(self.array, (width, height), interpolation=cv2.INTER_NEAREST)
        return new
    
    def copy(self):
        """Returns a deep copy of the image"""
        return copy.deepcopy(self)
    
    def imshow(self, name: str, size: int = None):
        """Display the image with cv2"""
        f = size//max(self.height, self.width) if size is not None else 1
        img = self.resize(self.height*f, self.width*f)
        cv2.imshow(name, img.array)
        
    def imwrite(self, name: str, size: int = None):
        """Save the image to a file with cv2"""
        f = size//max(self.height, self.width) if size is not None else 1
        img = self.resize(self.height*f, self.width*f)
        cv2.imwrite(name, img.array*255)
        
    
    def __neg__(self):
        new = self.copy()
        new.array = -self.array
        return new
    
    def __add__(self, val: Union[int, float, "Image"]):
        if isinstance(val, (int, float)):
            return self.add_int(val)
        elif isinstance(val, Image):
            return self.add_image(val)
        else:
            raise TypeError("You can only add an integer or an image.")
    
    def add_int(self, val: int|float):
        """Adds a constant to value to every pixel of the image"""
        new = self.copy()
        new.array += val
        return new
    
    def add_image(self, img: "Image"):
        """Adds the pixel values of both images together"""
        if self.shape != img.shape:
            raise ValueError("You can only add images of the same shape.")
        new = self.copy()
        new.array += img.array
        return new
    
    def __radd__(self, val: Union[int, float, "Image"]):
        return self + val
    
    def __sub__(self, val: Union[int, float, "Image"]):
        return self + -val
    
    def __rsub__(self, val: Union[int, float, "Image"]):
        return -self + val
    
    def __mul__(self, val: Union[int, float]):
        if isinstance(val, (int, float)):
            return self.mul_int(val)
        else:
            raise TypeError("You can only multiply with an integer or float")
        
    def mul_int(self, val: int|float):
        """Multiplies each pixel of the image with a constant."""
        new = self.copy()
        new.array *= val
        return new
    
    def __rmul__(self, val: int|float):
        return self * val
        
    def __floordiv__(self, val: int|float):
        new = self.copy()
        new.array //= val
        return new
    
    def __truediv__(self, val: int|float):
        new = self.copy()
        new.array /= val
        return new
