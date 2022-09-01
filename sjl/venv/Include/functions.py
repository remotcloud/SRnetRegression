import torch
import sympy as sym
import eventlet


def trans_expression(expression, simplify=False):
    '''
        transform string initial expression f(x0, x1...) to a enable reading form.
        return a symbol form when simplify == False
        return a simple easy form when simplify == True
        example:
             Input initial form: 'cos(mul(add(x0,x1),div(x0,x0)))'
             ->symbol form: 'cos((x0/x0)*(x0+x1))'
             ->simple form: 'cos(x0+x1)'
    '''
    exp = expression
    last_index = 0  # last position of '(' , ')' or ','
    func_stack = [] # where _Functions object store
    str_stack = []  # where string expression store
    for i in range(len(exp)):
        if exp[i] == '(':
            fname = exp[last_index:i]
            fname = fname.replace(" ", "")
            if fname not in function_map:
                raise ValueError('invalid function name:', fname)
            function = function_map[fname]
            func_stack.append(function)
            last_index = i + 1

        elif exp[i] == ',':
            if exp[i-1] != ')':
                oper = exp[last_index:i]
                str_stack.append(oper)
            last_index = i + 1

        elif exp[i] == ')':
            opers = []
            if exp[i-1] != ')':
                opers.insert(0, exp[last_index:i])
            else:
                opers.insert(0, str_stack.pop())
            function = func_stack.pop()
            for i in range(function.arity - 1):
                opers.insert(0, str_stack.pop())
            str_stack.append(_get_a_expression(function, opers))
            last_index = i + 1

    if len(str_stack) == 0:
        str_stack.append(exp)

    symbol_form = str_stack.pop()
    if not simplify:
         return symbol_form

    eventlet.monkey_patch()
    with eventlet.Timeout(10, True):
        simple_form = sym.simplify(symbol_form)
        return simple_form

    print('simplify time out: ', symbol_form)
    return symbol_form


def _get_a_expression(function, opers):
    return function_symbol_map[function.__name__].format(*opers)


class _Function(object):
    def __init__(self, function, name, arity):
        self.function = function
        self.__name__ = name
        self.arity = arity

    def __call__(self, *args):
        return self.function(*args)

    def __name__(self):
        return self.__name__


def _protected_division(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    one = torch.tensor(1.0)
    return torch.where(torch.abs(x2)>0.001, torch.divide(x1, x2), one)


def _protected_sqrt(x1):
    """Closure of square root for negative arguments."""
    return torch.sqrt(torch.abs(x1))


def _protected_log(x1):
    """Closure of log for zero arguments."""
    zero = torch.tensor(0.0)
    return torch.where(torch.abs(x1)>0.001, torch.log(torch.abs(x1)), zero)


def _protected_inverse(x1):
    """Closure of log for zero arguments."""
    zero = torch.tensor(0.0)
    return torch.where(torch.abs(x1)>0.001, 1. / x1, zero)


def _square(x):
    return torch.pow(x, 2)


def _cube(x):
    return torch.pow(x, 3)


add2 = _Function(function=torch.add, name='add', arity=2)
sub2 = _Function(function=torch.subtract, name='sub', arity=2)
mul2 = _Function(function=torch.multiply, name='mul', arity=2)
div2 = _Function(function=_protected_division, name='div', arity=2)
sqrt1 = _Function(function=_protected_sqrt, name='sqrt', arity=1)
log1 = _Function(function=_protected_log, name='log', arity=1)
neg1 = _Function(function=torch.negative, name='neg', arity=1)
inv1 = _Function(function=_protected_inverse, name='inv', arity=1)
abs1 = _Function(function=torch.abs, name='abs', arity=1)
max2 = _Function(function=torch.maximum, name='max', arity=2)
min2 = _Function(function=torch.minimum, name='min', arity=2)
sin1 = _Function(function=torch.sin, name='sin', arity=1)
cos1 = _Function(function=torch.cos, name='cos', arity=1)
tan1 = _Function(function=torch.tan, name='tan', arity=1)

function_names = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs',
                  'neg', 'inv', 'max', 'min', 'sin', 'cos', 'tan']

function_symbol_map = {'add': '({}+{})',
                     'sub': '({}-{})',
                     'mul': '({}*{})',
                     'div': '({}/{})',
                     'sqrt': '({}**0.5)',
                     'log': 'log({})',
                     'abs': '|{}|',
                     'neg': '(-{})',
                     'inv': '(1/{})',
                     'max': 'max({},{})',
                     'min': 'min({},{})',
                     'sin': 'sin({})',
                     'cos': 'cos({})',
                     'tan': 'tan({})'}

function_map = {'add': add2,
                 'sub': sub2,
                 'mul': mul2,
                 'div': div2,
                 'sqrt': sqrt1,
                 'log': log1,
                 'abs': abs1,
                 'neg': neg1,
                 'inv': inv1,
                 'max': max2,
                 'min': min2,
                 'sin': sin1,
                 'cos': cos1,
                 'tan': tan1}

if __name__ == '__main__':
    '''
    旧版前缀转中缀
    '''
    opers1 = ['x0']
    opers2 = ['sin(x0)', 'x1']
    layers = [['sub(cos(sub(cos(x0),sin(cos(div(x0,x0))))),mul(sub(cos(cos(div(x0,x0))),log(div(log(div(x0,x0)),div(div(x0,div(x0,x0)),div(x0,x0))))),mul(div(x0,div(x0,x0)),div(x0,x0))))']]
    for i in range(len(layers)):
        print(':::::::: Layer', i, '\n')
        for j in range(len(layers[i])):
            print('F_%d%d' % (i, j))
            b = layers[i][j]
            print('original form:', b)
            easy_expression = trans_expression(b, simplify=False)
            print('->readable form:', easy_expression)
            f = sym.simplify(easy_expression)
            print('->simple form:', f)