
Symbol = str
Number = {int, float}
List = list
Env = dict  

import math
import operator as op
        # An environment is a mapping of {variable: value}

def standard_env():
    env = Env()
    env.update(vars(math)) # sin, cos, sqrt, pi, ...
    env.update({
        '+':op.add, '-':op.sub, '*':op.mul, '/':op.div, 
        '>':op.gt, '<':op.lt, '>=':op.ge, '<=':op.le, '=':op.eq, 
        'abs':     abs,
        'append':  op.add,  
        'apply':   apply,
        'begin':   lambda *x: x[-1],
        'car':     lambda x: x[0],
        'cdr':     lambda x: x[1:], 
        'cons':    lambda x,y: [x] + y,
        'eq?':     op.is_, 
        'equal?':  op.eq, 
        'length':  len, 
        'list':    lambda *x: list(x), 
        'list?':   lambda x: isinstance(x,list), 
        'map':     map,
        'max':     max,
        'min':     min,
        'not':     op.not_,
        'null?':   lambda x: x == [], 
        'number?': lambda x: isinstance(x, Number),   
        'procedure?': callable,
        'round':   round,
        'symbol?': lambda x: isinstance(x, Symbol),
    })
    return env

class Env(dict):
    def __init__(self, parms=(), args=(), outer=None):
        self.update(zip(parms, args))
        self.outer = outer
    def find(self, var):
        return self if (var in self) else self.outer.find(var)

global_env = standard_env()


def eval(x, env=global_env):
    if isinstance(x, Symbol):      
        return env.find(x)[x]
    elif not isinstance(x, List):  
        return x                
    elif x[0] == 'quote':          
        (_, exp) = x
        return exp
    elif x[0] == 'if':             
        (_, test, conseq, alt) = x
        exp = (conseq if eval(test, env) else alt)
        return eval(exp, env)
    elif x[0] == 'define':         
        (_, var, exp) = x
        env[var] = eval(exp, env)
    elif x[0] == 'set!':           
        (_, var, exp) = x
        env.find(var)[var] = eval(exp, env)
    elif x[0] == 'lambda':         
        (_, parms, body) = x
        return Procedure(parms, body, env)
    else:                         
        proc = eval(x[0], env)
        args = [eval(exp, env) for exp in x[1:]]
        return proc(*args)

class Procedure(object):
    def __init__(self, parms, body, env):
        self.parms, self.body, self.env = parms, body, env
    def __call__(self, *args): 
        return eval(self.body, Env(self.parms, args, self.env))

def tokenize(program):
	return program.replace('(', ' ( ').replace(')', ' ) ').split()


def parse(program):
	return read_from_tokens(tokenize(program))



def read_from_tokens(tokens):
	if len(tokens) == 0:
		raise SyntaxError('unexpected EOF while reading')

	token = tokens.pop(0)

	if token == '(':
		L = []
		while tokens[0] != ')':
			L.append(read_from_tokens(tokens))
		tokens.pop(0)
		return L
	elif token == ')':
		raise SyntaxError('unexpected')
	else:
	    return atom(token)

def atom(token):
	try: return int(token)
	except ValueError:
		try: return float(token)
		except ValueError:
			return Symbol(token)

#A REPL

def repl(prompt='schemePY.py> '):
    while True:
        val = eval(parse(raw_input(prompt)))
        if val is not None: 
            print(lispstr(val))

def lispstr(exp):
    if isinstance(exp, List):
        return '(' + ' '.join(map(lispstr, exp)) + ')' 
    else:
        return str(exp)

repl()

