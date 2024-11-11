import unittest
import numpy as np
import sympy as sp

from main import gradient_f, lambdify_f

class UnitTest(unittest.TestCase):
    def setUp(self):
        self.x, self.y = sp.symbols('x y')
        self.func = 100*(self.y - self.x**3)**2 + (1 - self.x)**2

    def test_gradient_f(self):
        grad_f_x, grad_f_y = gradient_f(self.func)
        
        x_val, y_val = 0, 0
        expected_grad_x = 2
        expected_grad_y = 0

        np.testing.assert_almost_equal(grad_f_x(x_val, y_val), expected_grad_x, decimal=5)
        np.testing.assert_almost_equal(grad_f_y(x_val, y_val), expected_grad_y, decimal=5)

    def test_lambdify_f(self):
        func_lambdified = lambdify_f(self.func)
        
        x_val, y_val = 1, 1
        expected_value = 100 * (1 - 1**3)**2 + (1 - 1)**2

        np.testing.assert_almost_equal(func_lambdified(x_val, y_val), expected_value, decimal=5)

if __name__ == '__main__':
    unittest.main()