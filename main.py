import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


def gradient_f(func):
    x, y = sp.symbols('x y')
    
    df_dx = sp.diff(func, x)
    df_dy = sp.diff(func, y)
    
    grad_f_x = sp.lambdify((x, y), -df_dx, 'numpy')
    grad_f_y = sp.lambdify((x, y), -df_dy, 'numpy')
    
    return grad_f_x, grad_f_y

def lambdify_f(func):
    x, y = sp.symbols('x y')
    return sp.lambdify((x, y), func, 'numpy')


x, y = sp.symbols('x y')
# func = (x - 1)**2+(y-3)**2-x*y/2
func = 100*(y-x**3)**2+(1-x)**2

grad_f_x, grad_f_y = gradient_f(func)
# x_vals = np.linspace(-5, 6, 20)
# y_vals = np.linspace(-5, 6, 20)
x_vals = np.linspace(-1, 1.5, 20)
y_vals = np.linspace(-1, 1.5, 20)
X1, X2 = np.meshgrid(x_vals, y_vals)

func_numeric = lambdify_f(func)
Z = func_numeric(X1, X2)

# 1. Виведення графіка функції
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='none')
ax.set_title('3D Plot of the Function')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
plt.savefig("graph.png")
# plt.show()

df_dx = grad_f_x(X1, X2)
df_dy = grad_f_y(X1, X2)

plt.figure(figsize=(8, 6))
contour = plt.contour(X1, X2, Z, levels=[0.01, 0.1, 1.0, 10.0, 59.0, 100.0, 150.0, 200.0, 250.0, 300.0], cmap='viridis')
plt.clabel(contour, inline=True, fontsize=8)
plt.title('Contour Plot of the Function')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(contour)
plt.savefig("contour.png")
# plt.show()

# Візуалізація 
plt.figure(figsize=(8, 6))
plt.quiver(X1, X2, df_dx, df_dy)
plt.title('Gradient Vector Field')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("gradient.png")
# plt.show()