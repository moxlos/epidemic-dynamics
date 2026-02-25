#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Economic Cost Analysis for Epidemic Control

Provides functions to compute and optimize vaccination strategies based on
economic cost trade-offs.
"""

import numpy as np
from scipy.optimize import minimize_scalar

from sir_model import SIR
from integrators import Integrator, RK4


def compute_epidemic_cost(dynamics, vaccination_rate, vaccine_cost, health_cost, dt):
    """Compute total economic cost of an epidemic.

    Total cost = vaccination costs + health costs from infection

    Args:
        dynamics: Array of shape (n_steps, 3) with [S, I, R] at each time
        vaccination_rate: Daily vaccination rate (fraction of susceptible)
        vaccine_cost: Cost per vaccine administered
        health_cost: Daily cost per infected person
        dt: Time step size

    Returns:
        Total economic cost over the simulation period
    """
    s = dynamics[:, 0]
    i = dynamics[:, 1]

    # Vaccination cost: vaccinations occur at rate v*S
    # Number vaccinated per time step ≈ v * S * dt
    vaccination_cost = np.sum(vaccination_rate * s * vaccine_cost) * dt

    # Health cost: I people incur cost per day
    infection_cost = np.sum(i * health_cost) * dt

    return vaccination_cost + infection_cost


def simulate_epidemic(sir_params, initial_state, t_max, n_steps):
    """Run an epidemic simulation.

    Args:
        sir_params: Dictionary with SIR model parameters
            (alpha, recovery_rate, vaccination_rate, death_rate)
        initial_state: Initial [S, I, R] values
        t_max: Simulation end time
        n_steps: Number of time steps

    Returns:
        Tuple of (dynamics array, time array, dt)
    """
    sir = SIR(**sir_params)
    method = RK4(sir)
    integrator = Integrator(method, initial_state, 0, t_max, n_steps)
    dynamics = integrator.integrate()
    time = integrator.get_time()
    dt = integrator.dt
    return dynamics, time, dt


def optimize_vaccination_rate(base_sir_params, initial_state, t_max, n_steps,
                              vaccine_cost, health_cost,
                              v_min=0, v_max=0.1):
    """Find the optimal vaccination rate that minimizes total cost.

    Performs a bounded optimization to find the vaccination rate that
    minimizes the combined cost of vaccination and infection.

    Args:
        base_sir_params: Dictionary with SIR parameters (without vaccination_rate)
        initial_state: Initial [S, I, R] values
        t_max: Simulation end time
        n_steps: Number of time steps
        vaccine_cost: Cost per vaccine administered
        health_cost: Daily cost per infected person
        v_min: Minimum vaccination rate to consider
        v_max: Maximum vaccination rate to consider

    Returns:
        Dictionary with:
            - optimal_rate: Optimal vaccination rate
            - min_cost: Minimum total cost
            - result: Full scipy optimization result
    """

    def cost_function(v):
        params = base_sir_params.copy()
        params['vaccination_rate'] = v
        dynamics, _, dt = simulate_epidemic(params, initial_state, t_max, n_steps)
        return compute_epidemic_cost(dynamics, v, vaccine_cost, health_cost, dt)

    result = minimize_scalar(cost_function, bounds=(v_min, v_max), method='bounded')

    return {
        'optimal_rate': result.x,
        'min_cost': result.fun,
        'result': result
    }


def cost_sensitivity_analysis(base_sir_params, initial_state, t_max, n_steps,
                              vaccine_cost, health_cost, vaccination_rates):
    """Analyze how costs vary with vaccination rate.

    Args:
        base_sir_params: Dictionary with SIR parameters (without vaccination_rate)
        initial_state: Initial [S, I, R] values
        t_max: Simulation end time
        n_steps: Number of time steps
        vaccine_cost: Cost per vaccine administered
        health_cost: Daily cost per infected person
        vaccination_rates: Array of vaccination rates to evaluate

    Returns:
        Dictionary with:
            - rates: Input vaccination rates
            - total_costs: Total cost at each rate
            - vaccine_costs: Vaccination cost component at each rate
            - health_costs: Health cost component at each rate
    """
    total_costs = []
    vaccine_costs = []
    health_costs = []

    for v in vaccination_rates:
        params = base_sir_params.copy()
        params['vaccination_rate'] = v
        dynamics, _, dt = simulate_epidemic(params, initial_state, t_max, n_steps)

        s = dynamics[:, 0]
        i = dynamics[:, 1]

        v_cost = np.sum(v * s * vaccine_cost) * dt
        h_cost = np.sum(i * health_cost) * dt

        vaccine_costs.append(v_cost)
        health_costs.append(h_cost)
        total_costs.append(v_cost + h_cost)

    return {
        'rates': np.array(vaccination_rates),
        'total_costs': np.array(total_costs),
        'vaccine_costs': np.array(vaccine_costs),
        'health_costs': np.array(health_costs)
    }
