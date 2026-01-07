from trace_graph import draw_dot
import math

class Value:
    def __init__(self, data, _children = (), _op='', label=''):
        self.data = data
        self._op = _op
        self._prev = _children
        self.grad = 0.0
        self.label = label
        self._backward = lambda:None
    
    def __repr__(self) -> str:
        return f'Value({self.data} | grad({self.grad}))'
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other),'+')
        
        def _backward():
            self.grad += (1.0 * out.grad)
            other.grad += (1.0 * out.grad)
        out._backward = _backward  
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other),'*')
        def _backward():
            self.grad += (other.data * out.grad)
            other.grad += (self.data * out.grad)
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        return self * (other ** -1)
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self, ), f'**{other}')

        def _backward():
            self.grad+=other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -other

    def exp(self):
        x  = self.data
        out = Value(math.exp(x), (self,), 'exp')
        
        def _backward():
            self.grad += (out.data * out.grad)

        out._backward = _backward
        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += ((1 - t**2) * out.grad)
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def backward(self):
        self.grad=1.0
        topo = []
        visied = set()
        def build_topo(v):
            if v not in visied:
                visied.add(v)
                for n in v._prev:
                    build_topo(n)
                topo.append(v)
        build_topo(self) 
        for n in reversed(topo):
            n._backward()



# inputs x1, x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1, w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias of the neuron
b = Value(6.8813735870195432, label='b')

# x1*w1 + x2*w2 + b
x1w1 = x1*w1;x1w1.label='x1*w1'
x2w2 = x2*w2;x2w2.label='x2*w2'
x1w1x2w2 = x1w1 + x2w2;x1w1x2w2.label='x1*w1 + x2*w2'
n = x1w1x2w2 + b;n.label='n'
e = (2*n).exp();e.label='e'
o = (e-1) / (e+1);o.label = 'o'

o.backward()




dot = draw_dot(o)
dot.render('round-table.gv')
