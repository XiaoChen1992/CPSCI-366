"""
@ Description: Compare which PyTorch nodes retain gradients automatically or not.
@ Author: Prof. Chen
Function:
    z = (x^2 * y + sin(x)) * exp(exp(y + x*y))
"""

import torch

# Input values
x = torch.tensor(0.3, requires_grad=True)
y = torch.tensor(1.2, requires_grad=True)

# Intermediate calculations
a = x ** 2                   # a = x^2
a.retain_grad()             # Let's retain a's grad for demo

b = a * y                   # b = a * y
# not retaining b.grad

c = torch.sin(x)            # c = sin(x)
# not retaining c.grad

d = b + c                   # d = b + c
d.retain_grad()             # Retain d's grad

f = x * y                   # f = x * y
f.retain_grad()             # Retain f's grad

g = y + f                   # g = y + x*y
# not retaining g.grad

h = torch.exp(torch.exp(g)) # h = exp(exp(g))
# not retaining h.grad

z = d * h                   # z = d * h
z.retain_grad()             # Retain final output

# Backward pass
z.backward()

# Display which gradients are available
def show_grad(name, tensor):
    grad = tensor.grad
    print(f"{name:<2} | grad: {grad if grad is not None else 'None'}")

print("Gradients after backward():")
show_grad("x", x)     # always available
show_grad("y", y)     # always available
show_grad("a", a)     # retained
show_grad("b", b)     # not retained
show_grad("c", c)     # not retained
show_grad("d", d)     # retained
show_grad("f", f)     # retained
show_grad("g", g)     # not retained
show_grad("h", h)     # not retained
show_grad("z", z)     # retained (final output)