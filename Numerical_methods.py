import numpy as np

def euler_method(f, state, dt, time_points):
    states = [state]
    for t in time_points[:-1]:
        state = state + dt * f(state)
        states.append(state)
    return np.array(states)


def midpoint_method(f, state, dt, time_points):
    states = [state]
    for t in time_points[:-1]:
        k1 = f(state)
        k2 = f(state + dt * k1 / 2)
        state = state + dt * k2
        states.append(state)
    return np.array(states)


def rk2(f, state, dt, time_points):
    states = [state]
    for t in time_points[:-1]:
        k1 = f(state)
        k2 = f(state + dt * k1)
        state = state + dt * (k1 + k2) / 2
        states.append(state)
    return np.array(states)


def rk4(f, state, dt, time_points):
    states = [state]
    for t in time_points[:-1]:
        k1 = f(state)
        k2 = f(state + dt * k1 / 2)
        k3 = f(state + dt * k2 / 2)
        k4 = f(state + dt * k3)
        state = state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        states.append(state)
    return np.array(states)


def adams_bashforth(f, state, dt, time_points):
    states = [state]
    # Start with RK4 for the first step
    state = rk4(f, state, dt, [0, dt])[-1]
    states.append(state)

    for t in time_points[2:]:
        state = states[-1] + dt / 2 * (3 * f(states[-1]) - f(states[-2]))
        states.append(state)
    return np.array(states)

def adams_moulton(f, state, dt, time_points):
    states = [state]
    # Start with RK4 for the first step
    state = rk4(f, state, dt, [0, dt])[-1]
    states.append(state)

    for t in time_points[2:]:
        pred = states[-1] + dt / 2 * (3 * f(states[-1]) - f(states[-2]))
        state = states[-1] + dt / 2 * (f(pred) + f(states[-1]))
        states.append(state)
    return np.array(states)


