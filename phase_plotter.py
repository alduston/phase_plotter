import numpy as np
import matplotlib.pyplot as plt
import itertools
import random


def iterate_forward(f, x, delta):
    dx = delta * f(x)
    return x + dx


def forward_trajectory(f ,x_0 = [], R= 5, delta = .1, lim_iter=1000):
    if not len(x_0):
        x_0 = np.asarray([random.randrange(-R, R), random.randrange(-R, R)])
    trajectory = [x_0]

    finished = False
    k = 0
    while not finished:
        x = iterate_forward(f, trajectory[-1], delta)
        if  max(np.abs(x)) <= R and v_cycled(x,trajectory) and k < lim_iter:
            k += 1
            trajectory.append(x)
        else:
            return trajectory + [x]


def iterate_backward(f, x, delta):
    dx = delta * f(x)
    return x - dx


def backward_trajectory(f ,x_0 = [], R= 5, delta = .1, lim_iter=1000):
    if not len(x_0):
        x_0 = np.asarray([random.randrange(-R, R), random.randrange(-R, R)])
    trajectory = [x_0]

    finished = False
    k = 0
    while not finished:
        x = iterate_backward(f, trajectory[0], delta)
        if max(np.abs(x)) <= R and v_cycled(x,trajectory) and k < lim_iter:
            k += 1
            trajectory = [x] + trajectory
        else:
            return [x] + trajectory


def create_trajectory(f , x_0 = [], R= 10, delta = .01, lim_iter=1000):
    forward = forward_trajectory(f ,x_0 , R, delta, lim_iter)
    backward = backward_trajectory(f ,x_0, R, delta , lim_iter)[:-1]
    return backward + forward


def v_cycled(v, V, tol = 10 ** -7):
    return np.linalg.norm(v-V[0]) > tol


def plot_axes(R = 5):
    axis = np.arange(-R -1, R + 1, 1)
    plt.plot(axis, np.zeros(len(axis)), color = 'black')
    plt.plot(np.zeros(len(axis)), axis, color = 'black')
    plt.xlim(-R , R )
    plt.ylim(-R , R )


def colored_plot(x,y):
    x_segments = [[x[i],x[i+1]] for i in range(0,len(x)-1)]
    y_segments = [[y[i], y[i + 1]] for i in range(0, len(x) - 1)]

    L = 0
    for (x, y) in (zip(x_segments, y_segments)):
        L += (x[1]-x[0])**2 + (y[1]-y[0])**2

    L_i = 0
    for i, (x,y) in enumerate(zip(x_segments, y_segments)):
        L_i += (x[1]-x[0])**2 + (y[1]-y[0])**2
        blue = 1 - L_i/L
        red = L_i/L

        segment_color = (red, 0, blue)
        plt.plot(x,y,color = segment_color)
    return True


def list_product(l_1,l_2):
    L = []
    for x in l_1:
        for y in l_2:
            L.append((x,y))
    return L


def unique_list(l):
    return list(set(l))


def get_trajectory_vals(trajectory, void_eps):
    return unique_list([str(np.asarray(np.rint((np.asarray(cords / void_eps))),int)).replace("  ", " ") for cords in trajectory])


def plot_trajectories_grid(f, R = 5, colored = False, cords = [np.zeros(2)], void_eps = .5):
    plot_axes(R)
    plt.xlim(-R,R)
    plt.ylim(-R, R)
    trajectories = []

    for x_0 in cords:
        trajectories.append(create_trajectory(f, x_0, R))

    trajectorie_val_set = [get_trajectory_vals(trajectory, void_eps) for trajectory in trajectories]
    trajectorie_val_set = unique_list(list(itertools.chain.from_iterable(trajectorie_val_set)))


    big_cords = np.asarray(np.arange(round(-R/void_eps),round(R/void_eps),1), int)
    big_cords = [np.asarray(cords, int)  for cords in list_product(big_cords, big_cords)]
    random.shuffle(big_cords)
    big_cords_set = [str(cord).replace("  ", " ") for cord in big_cords]


    for i,cord in enumerate(big_cords_set):
        if cord not in trajectorie_val_set:
            x_0 = big_cords[i] * void_eps
            trajectory = create_trajectory(f, x_0, R)
            trajectories.append(trajectory)
            trajectorie_val_set = unique_list(trajectorie_val_set + get_trajectory_vals(trajectory, void_eps))

    for trajectory in trajectories:
        x = [v[0] for v in trajectory]
        y = [v[1] for v in trajectory]
        if colored:
            colored_plot(x,y)
        else:
            plt.plot(x,y, color = 'blue')

    plt.show()
    return True

def run():
    # Define a function taking 2 by 1 vector as input, ex:
    def f(z):
        (x, y) = z
        x_dot = np.sin(x * y)
        y_dot = np.sin(x * y)
        return np.asarray([x_dot, y_dot])


    # R: float valued, gives range over which you are plotting, ex : R = 10 will plot in [-10,10]x[-10x10]
    # colored: bool, if true, will color plot according to direction of movement,
    # values start at blue, go to red
    # cords: list of 2 vectors. Gives points at which phase plane trajectories are intialiazed
    # voids_eps: float, governs density of plot. smaller void eps -> greater plot density, more trajectory lines

    plot_trajectories_grid(f, R=5, colored=False, cords=[np.zeros(2)], void_eps=.5)


    pass

if __name__ == '__main__':
    run()
