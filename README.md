# schemePY
a scheme subset implemented in Python

从0用python实现一个scheme的一个子集. 

scheme采用前缀表示法, 这点与之前接触的语言都不相同, 含高阶函数, 并且语法非常简单.

<center><img src="https://ooo.0o0.ooo//2017//03//17//58cb52fec79b9.png" width="400"></center>

那么解释器和编译器有什么区别呢, 解释器（Interpreter）是一种程序，能够读入程序并直接输出结果，如上图。相对于编译器（Compiler），解释器并不会生成目标机器代码，而是直接运行源程序，简单来说：

> 解释器是运行程序的程序。


一个解释器主要包含

1. 解析, 包含词法分析和语法分析, 生成抽象的语法树.
2. 求值, 对生成的语法树进行求值.

## 解析

### 词法分析

词法分析负责把源程序解析成一个个词法单元（Lex），以便之后的处理。这个子集的词法单元只包括括号, 符号, 数值. 简单的利用python的split函数就可以完成工作

```python
def tokenize(chars):
    return chars.replace('(', ' ( ').replace(')', ' ) ').split()
```

让我们试试效果, 先定义一个g

```python
In [18]: program = "(def gcd (func (a b) (if (= b 0) a (func (b (% a b))))))"

In [19]: tokenize(program)
Out[19]:
['(',
 'def',
 'gcd',
 '(',
 'func',
 '(',
 'a',
 'b',
 ')',
 '(',
 'if',
 '(',
 '=',
 'b',
 '0',
 ')',
 'a',
 '(',
 'func',
 '(',
 'b',
 '(',
 '%',
 'a',
 'b',
 ')',
 ')',
 ')',
 ')',
 ')',
 ')']
 ```

### 语法分析

接下来我们需要从我们得到的词法单元构造语法树. 构造词法树的函数是递归形式的, 首先我们判断是否遇到'(', 表明一个根节点开始, 将其从词法单元中取出, 对剩余的词法单元进行递归处理, 并推到结果去, 直到遇到')', 表示结束, 从队列中pop掉')'. 

```python
def read_from_tokens(tokens):
    if len(tokens) == 0:
        raise SyntaxError('unexpected EOF while reading')
    token = tokens.pop(0)
    if '(' == token:
        L = []
        while tokens[0] != ')':
            L.append(read_from_tokens(tokens))
        tokens.pop(0) # pop off ')'
        return L
    elif ')' == token:
        raise SyntaxError('unexpected )')
    else:
        return atom(token)
```

这里atom函数用来对数值和符号进行判断, 直接利用python的强制类型转换.

```python
def atom(token):
    try: return int(token)
    except ValueError:
        try: return float(token)
        except ValueError:
            return Symbol(token)
```

接下来需要定义Scheme中一些对象, 我们直接可以从python中调用相应的实现

```python
Symbol = str
List = list
Number = {float,int}
```

把词法分析和语法分析连接在一起, 我们就有一个解析器.

```python
def parse(program):
    return read_from_tokens(tokenize(program))
```

下面测试一下

```python
In [20]: parse(program)
Out[20]:
['def',
 'gcd',
 ['func',
  ['a', 'b'],
  ['if', ['=', 'b', 0], 'a', ['func', ['b', ['%', 'a', 'b']]]]]]
```

## 求值

### 作用域

当要对一个语法树进行求值的时候, 我们首先要确定求值的作用域, 用来确定一个变量对应的值是多少. 比如不同函数的同一个变量可能有相同的名字, 但是求值的结果是不相同的. 默认使用全局作用域, 可以使用python的字典来实现.

```python
import math
import operator as op

Env = dict          

def standard_env():
    "An environment with some Scheme standard procedures."
    env = Env()
    env.update(vars(math)) 
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

global_env = standard_env()
```

### 求值

先看看scheme的求值形式.

<p align="center"><img src="https://ooo.0o0.ooo//2017//03//17//58cb3ec84e92b.png" width="580"></p>

按此我们对语法树进行求值

```
def eval(x, env=global_env):
    "Evaluate an expression in an environment."
    if isinstance(x, Symbol):     
        return env[x]
    elif not isinstance(x, List):  
        return x                
    elif x[0] == 'if':            
        (_, test, conseq, alt) = x
        exp = (conseq if eval(test, env) else alt)
        return eval(exp, env)
    elif x[0] == 'define':         
        (_, var, exp) = x
        env[var] = eval(exp, env)
    else:                         
        proc = eval(x[0], env)
        args = [eval(arg, env) for arg in x[1:]]
        return proc(*args)
```

加上一个repl


```python
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
```

框架完成, 只需要添加lambda和一些环境即可

```python

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
```

## 测试

```python
Memoir@Purelove ~/desktop> python schemePY.py                                master-!?
schemePY.py> (define circle-area (lambda (r) (* pi (* r r))))
schemePY.py> (circle-area 3)
28.2743338823
schemePY.py> (define fact (lambda (n) (if (<= n 1) 1 (* n (fact (- n 1))))))
schemePY.py> (fact 10)
3628800
schemePY.py> (fact 100)
93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000
schemePY.py> (circle-area (fact 10))
4.13690872058e+13
schemePY.py> (define first car)
schemePY.py> (define rest cdr)
schemePY.py> (define count (lambda (item L) (if L (+ (equal? item (first L)) (count item (rest L))) 0)))
schemePY.py> (count 0 (list 0 1 2 3 0 0))
3
schemePY.py> (count (quote the) (quote (the more the merrier the bigger the better)))
4
schemePY.py> (define twice (lambda (x) (* 2 x)))
schemePY.py> (twice 5)
10
schemePY.py> (define repeat (lambda (f) (lambda (x) (f (f x)))))
schemePY.py>  ((repeat twice) 10)
40
schemePY.py> ((repeat (repeat twice)) 10)
160
schemePY.py> ((repeat (repeat (repeat twice))) 10)
2560
schemePY.py> ((repeat (repeat (repeat (repeat twice)))) 10)
655360
schemePY.py> (pow 2 16)
65536.0
schemePY.py> (define fib (lambda (n) (if (< n 2) 1 (+ (fib (- n 1)) (fib (- n 2))))))
schemePY.py> (define range (lambda (a b) (if (= a b) (quote ()) (cons a (range (+ a 1) b)))))
schemePY.py> (range 0 10)
(0 1 2 3 4 5 6 7 8 9)
schemePY.py> (map fib (range 0 10))
(1 1 2 3 5 8 13 21 34 55)
schemePY.py> (map fib (range 0 20))
(1 1 2 3 5 8 13 21 34 55 89 144 233 377 610 987 1597 2584 4181 6765)
```



