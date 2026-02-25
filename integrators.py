#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numerical Integration Schemes for ODEs

Provides explicit time-stepping methods for solving ordinary differential equations.
"""

import numpy as np


class ExplicitEuler:
    """Forward Euler scheme for numerical ODE integration.

    The simplest explicit method:
        y_{n+1} = y_n + dt * f(y_n, t_n)

    First-order accurate. Can be unstable for stiff problems.
    """

    def __init__(self, f):
        """Initialize with the ODE right-hand side function.

        Args:
            f: Callable f(y, t) returning dy/dt
        """
        self.f = f

    def iterate(self, y, t, dt):
        """Perform one Euler step.

        Args:
            y: Current state
            t: Current time
            dt: Time step

        Returns:
            State at time t + dt
        """
        return y + dt * self.f(y, t)


class RK2:
    """Second-order Runge-Kutta (midpoint method).

    Uses a midpoint evaluation for better accuracy:
        k1 = f(y_n, t_n)
        k2 = f(y_n + dt/2 * k1, t_n + dt/2)
        y_{n+1} = y_n + dt * k2

    Second-order accurate.
    """

    def __init__(self, f):
        """Initialize with the ODE right-hand side function.

        Args:
            f: Callable f(y, t) returning dy/dt
        """
        self.f = f

    def iterate(self, y, t, dt):
        """Perform one RK2 step.

        Args:
            y: Current state
            t: Current time
            dt: Time step

        Returns:
            State at time t + dt
        """
        k1 = self.f(y, t)
        k2 = self.f(y + dt / 2 * k1, t + dt / 2)
        return y + dt * k2


class RK4:
    """Fourth-order Runge-Kutta method.

    The classic RK4 scheme:
        k1 = f(y_n, t_n)
        k2 = f(y_n + dt/2 * k1, t_n + dt/2)
        k3 = f(y_n + dt/2 * k2, t_n + dt/2)
        k4 = f(y_n + dt * k3, t_n + dt)
        y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    Fourth-order accurate. Excellent balance of accuracy and efficiency.
    """

    def __init__(self, f):
        """Initialize with the ODE right-hand side function.

        Args:
            f: Callable f(y, t) returning dy/dt
        """
        self.f = f

    def iterate(self, y, t, dt):
        """Perform one RK4 step.

        Args:
            y: Current state
            t: Current time
            dt: Time step

        Returns:
            State at time t + dt
        """
        k1 = self.f(y, t)
        k2 = self.f(y + dt / 2 * k1, t + dt / 2)
        k3 = self.f(y + dt / 2 * k2, t + dt / 2)
        k4 = self.f(y + dt * k3, t + dt)
        return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class Integrator:
    """Generic ODE integrator wrapper.

    Integrates a differential equation from t_min to t_max using a
    specified time-stepping method.

    Example:
        >>> from sir_model import SIR
        >>> sir = SIR(alpha=0.3, recovery_rate=0.1)
        >>> method = RK4(sir)
        >>> integrator = Integrator(method, y0=[999, 1, 0], t_min=0, t_max=100, n_steps=1000)
        >>> dynamics = integrator.integrate()
        >>> time = integrator.get_time()
    """

    def __init__(self, method, y0, t_min, t_max, n_steps):
        """Initialize the integrator.

        Args:
            method: Time-stepping method with iterate(y, t, dt) method
            y0: Initial state (scalar or array)
            t_min: Start time
            t_max: End time
            n_steps: Number of time steps
        """
        self.method = method
        self.y0 = np.asarray(y0)
        self.t_min = t_min
        self.t_max = t_max
        self.n_steps = n_steps
        self.dt = (t_max - t_min) / (n_steps - 1)

    def get_time(self):
        """Return the time array.

        Returns:
            Array of time points from t_min to t_max
        """
        return np.linspace(self.t_min, self.t_max, self.n_steps)

    def integrate(self):
        """Integrate the ODE.

        Returns:
            Array of states at each time point. Shape is (n_steps,) for
            scalar ODEs or (n_steps, n_components) for systems.
        """
        # Initialize storage
        if self.y0.ndim == 0:
            # Scalar ODE
            result = np.zeros(self.n_steps)
        else:
            # System of ODEs
            result = np.zeros((self.n_steps, len(self.y0)))

        result[0] = self.y0
        y = self.y0.copy()

        # Time-stepping loop
        t = self.t_min
        for i in range(1, self.n_steps):
            y = self.method.iterate(y, t, self.dt)
            result[i] = y
            t += self.dt

        return result
