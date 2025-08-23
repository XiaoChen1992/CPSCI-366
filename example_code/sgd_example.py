import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2 - 4*x + 3

def df(x):
    return 2*x - 4

# Parameters
x = -4       # initial guess
eta = 1.2   # learning rate
iterations = 20  # fewer iterations for clearer visualization

# Collect data points for the plot
x_values = [x]
y_values = [f(x)]

# SGD optimization
for i in range(iterations):
    grad = df(x)
    x = x - eta * grad
    x_values.append(x)
    y_values.append(f(x))

# Generate values for plotting the function
x_plot = np.linspace(-10, 10, 1000)
y_plot = f(x_plot)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot, label='f(x) = $x^2 - 4x + 3$')
plt.scatter(x_values, y_values, color='red', zorder=5)
plt.plot(x_values, y_values, 'ro--', label='SGD Steps', markersize=8)

# Annotating the initial and final points
plt.annotate('Start', xy=(x_values[0], y_values[0]), xytext=(x_values[0]-0.5, y_values[0]+3),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate('End', xy=(x_values[-1], y_values[-1]), xytext=(x_values[-1], y_values[-1]+3),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.title('SGD Optimization Trajectory on f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
