import numpy as np
class AppendArray():
    """
    A wrapper around a numpy array that supports the append operation, but does it
    without making copies. It does this by doubling the size of the array every time
    it fills up, amortizing the cost of appending.

    This is a utility class, and does not need to be used by users of the simulator.
    """

    def __init__(self, init_x, dt):
        """
        Parameters
        ----------

        init_x:
            The initial length of the array

        y:
            The width of the array (number of data per row)

        dt:
            The numpy data type for a row
        """

        self.data = np.zeros(init_x, dtype=dt)
        self.full = 0
        self.dt   = dt

    def append(self, val):
        """
        Append a row to the array.

        Parameters
        ----------

        val:
            The row (of type dt) to append to the array. The array will be expanded if needed.
        """

        val = np.array(tuple(val), dtype=self.dt)

        if self.full < len(self.data):
            self.data[self.full] = val
            self.full += 1
        else:
            self.data = np.lib.pad(self.data, (0,self.data.shape[0]),
                                   'constant', constant_values=(0))
            self.data[self.full] = val
            self.full += 1

    def shrink(self):
        """
        Shrink the array to the size of however many elements are in it.
        """

        self.data = self.data[:self.full]

    def get(self):
        """
        Get a copy of they numpy array associated with this _AppendArray
        """

        return self.data[:self.full].copy()

    def reset(self):
        """
        Reset the contents of this array
        """

        self.full = 0
