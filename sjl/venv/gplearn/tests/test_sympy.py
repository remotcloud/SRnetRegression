from sympy import sympify
from sympy import *

x = symbols('x')
class sigmoid(Function):
    @ classmethod
    def eval(cls, x):
        if x.is_Number:
            return 1 / (1 + exp(-abs(x)))
    def _eval_is_real(self):
        return self.args[0].is_real
variables = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
values = range(1, len(variables) + 1)
sig = sigmoid(1)
expression = sympify('sigmoid(x)',locals={'sigmoid':sigmoid})
sub_table = [('x',x+1)]
expression = expression.subs(sub_table)
sub_table = [('x',x+1)]
expression = expression.subs(sub_table)
sub_table = [('x',1)]
expression = expression.subs(sub_table)
print (values)
# [1, 2, 3, 4, 5, 6, 7]
print( expression)
# a*b*c*d*e*f*g
print (sub_table)
# [('a', 1), ('b', 2), ('c', 3), ('d', 4), ('e', 5), ('f', 6), ('g', 7)]
# x= sympify('3.34725515744247*sin(0.0408984742947494*(0.546671791972115 - 0.546074501787973*sin(X0))*(-0.298850048138004*(1 - 0.998907406248295*sin(X0))**2 - 1.63822350536392*sin(X0) + 1.64001537591635)*sin(0.298850048138004*(1 - 0.998907406248295*sin(X0))**2 - 0.723456719097708) + 0.154669861924483*sin(X0) + 0.312631283063277) + 2.38229476561238*sin(sin(0.227088611910744*(0.546671791972115 - 0.546074501787973*sin(X0))*(-0.298850048138004*(1 - 0.998907406248295*sin(X0))**2 - 1.63822350536392*sin(X0) + 1.64001537591635)*sin(0.298850048138004*(1 - 0.998907406248295*sin(X0))**2 - 0.723456719097708) + 0.460400085528707*sin(X0) + 0.327475656597629)) + 2.38229476561238*cos(cos(0.328053698786161*(0.546671791972115 - 0.546074501787973*sin(X0))*(-0.298850048138004*(1 - 0.998907406248295*sin(X0))**2 - 1.63822350536392*sin(X0) + 1.64001537591635)*sin(0.298850048138004*(1 - 0.998907406248295*sin(X0))**2 - 0.723456719097708) + 0.665096984425274*sin(X0) + 0.473073482220691)) - 3.00789175631004')

print(x)