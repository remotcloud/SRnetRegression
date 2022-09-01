"""The functions used to create programs.

"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import numpy as np
from sklearn.externals import six

__all__ = ['_func_map','_range_map']


class _Function(object):

    """A representation of a mathematical relationship, a node in a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting vector based on a mathematical relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(x1, *args) that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the ``function`` takes.

    """

    def __init__(self, function, name, arity):
        self.function = function
        self.name = name
        self.arity = arity

    def __call__(self, *args):
        return self.function(*args)


def make_function(function, name, arity):
    """Make a function node, a representation of a mathematical relationship.

    This factory function creates a function node, one of the core nodes in any
    program. The resulting object is able to be called with NumPy vectorized
    arguments and return a resulting vector based on a mathematical
    relationship.

    Parameters
    ----------
    function : callable
        A function with signature `function(x1, *args)` that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the `function` takes.

    """
    if not isinstance(arity, int):
        raise ValueError('arity must be an int, got %s' % type(arity))
    if not isinstance(name, six.string_types):
        raise ValueError('name must be a string, got %s' % type(name))

    # Check output shape
    args = [np.ones(10) for _ in range(arity)]
    try:
        h=function(*args)
    except ValueError:
        raise ValueError('supplied function %s does not support arity of %d.'
                         % (name, arity))
    if not hasattr(function(*args), 'shape'):
        raise ValueError('supplied function %s does not return a numpy array.'
                         % name)
    if function(*args).shape != (10,):
        raise ValueError('supplied function %s does not return same shape as '
                         'input vectors.' % name)

    # Check closure for zero & negative input arguments
    args = [np.zeros(10) for _ in range(arity)]
    '''if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'zeros in argument vectors.' % name)'''
    '''args = [-1 * np.ones(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'negatives in argument vectors.' % name)'''

    return _Function(function, name, arity)


def _koza1(x1):
    return np.sin(x1)+np.sin(x1+x1**2)
def _koza2(x1,x2):
    return 2*np.sin(x1)*np.cos(x2)
def _korns1(x1):
    return 3+2.13*np.log(np.abs(x1))
def _korns2(x1,x2):
    with np.errstate(divide='ignore', invalid='ignore'):
        return 1/(1+np.power(x1,-4))+1/(1+np.power(x2,-4))
def _keijzer1(x1,x2,x3):
    return 30*x1*x2/((x1-10)*x3**2)
def _keijzer2(x1,x2):
    return x1*x2+np.sin((x1-1)*(x2-1))

def _feynman7(x1,x2,x3,y1,y2,y3):
    return x1*y1+x2*y2+x3*y3
k0 = make_function(function=_koza1, name='kkk0',arity=1)
k1 = make_function(function=_koza2, name='kkk1',arity=2)
k2 = make_function(function=_korns1, name='kkk2',arity=1)
k3 = make_function(function=_korns2, name='kkk3',arity=2)
k4 = make_function(function=_keijzer1, name='kkk4',arity=3)
k5 = make_function(function=_keijzer2, name='kkk5',arity=2)
feynman7 = make_function(function=_feynman7,name='feynman7',arity=6)
_func_map = {'kkk0': k0,
            'kkk1': k1,
            'kkk2': k2,
            'kkk3': k3,
            'kkk4': k4,
            'kkk5': k5,
            'feynman7': feynman7
}

_range_map = {
    'kkk0': [-1,1,200],
                 'kkk1': [-1,1,200],
                 'kkk2': [-50,50,200],
                 'kkk3': [-5,5,25],
                 'kkk4': [-1,1,1000],
                 'kkk5': [-3,3,20],
                 'feynman7': [1,5,2000]
}
if __name__ == '__main__':
    x = _range_map['kkk2']
    feynman7 = _func_map['feynman7']

    print(x)