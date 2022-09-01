import sys
import sympy
from sympy import *

import traceback
import _pickle as pickle

sys.path.append('..')
from gplearn._program import _Program


def test_combine():
    #image_compare()
    log_dir = f'log/kkk0_program.pkl'
    with open(log_dir, 'rb') as f:
        individual_Set = pickle.load(f)
    individual_Set = individual_Set[0]
    for run_num in range(4):
        n_layer = len(individual_Set[run_num])
        former_exp = []
        for hide in range(n_layer):
            individual = individual_Set[run_num][hide]
            y_features = len(individual[2])
            x_features = len(individual[0])
            y_pre = []
            variables_set = []
            for m in range(y_features):
                weight = individual[1]
                exp = sympy.sympify('0')
                for k in range(x_features):
                    x_n,symbol_set = individual[0][k].get_expression()
                    weight_m_k = weight[m][0][k]
                    exp = x_n*weight_m_k +exp
                    if m==0:
                        variables_set.append(symbol_set)
                exp = exp + individual[2][m][0]
                exp = exp.xreplace({n : round(n, 3) for n in exp.atoms(Number)})
                y_pre.append(exp)
            if former_exp:
                for inda,i in zip(y_pre,range(len(y_pre))):
                    try:
                        variables = variables_set
                        sub_table= []
                        for j in range(x_features):
                            sub_table.append((variables[j][0],former_exp[j]))

                        print(len(str(inda)))
                        inda = inda.subs(sub_table)
                        print(len(str(inda)))
                        y_pre[i] = inda
                        # print('individual:',inda)
                        # print('x:',exp)
                        # if hide!=0:
                        #     for symbol in symbol_set:
                        #         exp = exp.subs(symbol[0],former_exp[symbol[1]])
                        # former_exp.append(exp)
                    except:
                        print('traceback.print_exc():', traceback.print_exc())
                        print('traceback.format_exc():\n%s' % traceback.format_exc())
                        pass
            former_exp = y_pre
    #X0,X1  = sympy.symbols('X0')
    expy = sympy.sympify('sin((X0 +- 0.203)) + ((X254536 + X0) - sin(-0.576))')
if __name__ == '__main__':

    test_combine()