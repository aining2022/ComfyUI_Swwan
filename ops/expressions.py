# SPDX-License-Identifier: MIT
"""Scalar expression evaluator; no Python eval or arbitrary attribute calls."""
import ast
import math
import operator
import random

BINARY={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,
        ast.FloorDiv:operator.floordiv,ast.Mod:operator.mod,ast.Pow:operator.pow,ast.BitXor:operator.xor,
        ast.BitAnd:operator.and_,ast.BitOr:operator.or_,ast.LShift:operator.lshift,ast.RShift:operator.rshift}
UNARY={ast.USub:operator.neg,ast.UAdd:operator.pos,ast.Not:operator.not_,ast.Invert:operator.invert}
COMPARE={ast.Eq:operator.eq,ast.NotEq:operator.ne,ast.Lt:operator.lt,ast.LtE:operator.le,ast.Gt:operator.gt,ast.GtE:operator.ge}
FUNCTIONS={name:getattr(math,name) for name in ['sin','cos','tan','asin','acos','atan','atan2','pow','sqrt','hypot','log','log10','exp','sinh','cosh','tanh','asinh','acosh','atanh','radians','degrees','fabs','ceil','floor','copysign','fmod']}
FUNCTIONS.update(round=round,min=min,max=max,abs=abs,int=int,
                 randomint=random.randint,randomchoice=lambda *args:random.choice(args),
                 iif=lambda a,b,c:b if a else c,clamp=lambda x,lo,hi:max(min(x,hi),lo),
                 lerp=lambda a,b,c:a+(b-a)*c,sign=lambda a:(a>0)-(a<0))


def evaluate_expression(expression, values, attribute=None, legacy_ast=False):
    tree=ast.parse(expression.replace('\n',' ').replace('\r','').replace('÷','/'),mode='eval').body
    def visit(n):
        if isinstance(n,ast.Constant) and isinstance(n.value,(int,float,complex,bool)):return n.value
        if isinstance(n,ast.Name) and n.id in values:
            value=values[n.id]
            if not isinstance(value,(int,float,complex,bool)):raise TypeError('Use image.width or image.height for tensor inputs.')
            return value
        if isinstance(n,ast.BinOp) and type(n.op) in BINARY:return BINARY[type(n.op)](visit(n.left),visit(n.right))
        if isinstance(n,ast.UnaryOp) and type(n.op) in UNARY:return UNARY[type(n.op)](visit(n.operand))
        if isinstance(n,ast.BoolOp):
            items=n.values[:2] if legacy_ast else n.values
            result=visit(items[0])
            for item in items[1:]:
                if isinstance(n.op,ast.And):result=visit(item) if result else result
                else:result=result if result else visit(item)
            return int(bool(result)) if legacy_ast else result
        if isinstance(n,ast.Compare):
            left=visit(n.left)
            pairs=list(zip(n.ops,n.comparators))[:1] if legacy_ast else list(zip(n.ops,n.comparators))
            for op,right_node in pairs:
                if type(op) not in COMPARE:raise ValueError('Unsupported comparison.')
                right=visit(right_node)
                if not COMPARE[type(op)](left,right):return 0 if legacy_ast else False
                left=right
            return 1 if legacy_ast else True
        if isinstance(n,ast.IfExp):return visit(n.body if visit(n.test) else n.orelse)
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in FUNCTIONS and not n.keywords:
            return FUNCTIONS[n.func.id](*(visit(a) for a in n.args))
        if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and attribute and not n.attr.startswith('_'):
            return attribute(n.value.id,n.attr)
        raise ValueError('Unsupported expression syntax or name: '+ast.dump(n,include_attributes=False))
    return visit(tree)
