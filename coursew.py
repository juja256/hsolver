import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from hsolver import HyperbolicSolver


def one_dimensional_problem():
	delta_x = 0.01
	delta_t = 0.01
	grid_s = [np.arange(0, 1+delta_x, delta_x),]
	grid_t = np.arange(0, 10, delta_t)
	initial_cond = [lambda x: 0.5*x*(x-1), lambda x: 0]
	F = lambda x, t: -np.cos(np.pi*t)*(0.5*(np.pi**2)*x**2 - 0.5*(np.pi**2)*x + 1)
	edges_cond = [lambda t: 0, lambda t: 0]
	s1 = HyperbolicSolver(grid_s, grid_t, initial_cond, edges_cond, F, [0.3, 0.3])
	s1.solve()
	true_solution = lambda x, t: 0.5*x*(x-1)*np.cos(np.pi*t)
	s1.animate(true_solution)
	s1.saveIntoImg([0, 0.25, 0.5, 0.75, 1, 9, 9.25, 9.5, 9.75, 9.99], true_solution)
	s1.saveErrorIntoImg([0, 0.25, 0.5, 0.75, 1, 9, 9.25, 9.5, 9.75, 9.99], true_solution)


def two_dimensional_problem():
	delta_x = 0.1
	delta_t = 0.01
	grid_s = [np.arange(0, 1+delta_x, delta_x), np.arange(0, 1+delta_x, delta_x)]
	grid_t = np.arange(0, 10, delta_t)
	initial_cond = [lambda x, y: 4*x*(1-x)*(1-y)*y, lambda x, y: 0]
	F = lambda x, y, t: 0
	edges_cond = [lambda t: 0, lambda t: 0]
	s2 = HyperbolicSolver(grid_s, grid_t, initial_cond, edges_cond, F, [0.3, 0.3])
	s2.solve()
	s2.animate(None)
	s2.saveIntoImg([0, 0.25, 0.5, 0.75, 1, 9, 9.25, 9.5, 9.75, 9.99], None)


if __name__ == '__main__':
	one_dimensional_problem()
	#two_dimensional_problem()
	
	
