#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SIR Model for Epidemic Dynamics

A compartmental model for disease spread with vaccination and demographics.
"""

import numpy as np


class SIR:
    """SIR model with vaccination and demographics.

    The model describes the flow of individuals through three compartments:
    - S (Susceptible): can become infected
    - I (Infected): currently infected and infectious
    - R (Recovered): recovered and immune

    Differential equations:
        dS/dt = -αSI/N - vS - dS + dN
        dI/dt = αSI/N - (r+d)I
        dR/dt = rI - dR + vS

    where:
        α = infection rate (alpha)
        r = recovery rate
        v = vaccination rate
        d = death rate (assumed equal to birth rate for constant population)
        N = S + I + R = total population

    Attributes:
        alpha: Infection rate (transmission coefficient)
        recovery_rate: Rate at which infected individuals recover
        vaccination_rate: Rate at which susceptible individuals are vaccinated
        death_rate: Natural death rate (equals birth rate)
    """

    def __init__(self, alpha, recovery_rate, vaccination_rate=0, death_rate=0):
        """Initialize SIR model parameters.

        Args:
            alpha: Infection rate (transmission coefficient)
            recovery_rate: Rate at which infected individuals recover
            vaccination_rate: Rate at which susceptible individuals are vaccinated
            death_rate: Natural death rate (equals birth rate for constant N)
        """
        self.alpha = alpha
        self.recovery_rate = recovery_rate
        self.vaccination_rate = vaccination_rate
        self.death_rate = death_rate

    def __call__(self, y, t):
        """Compute derivatives for integration.

        Args:
            y: State vector [S, I, R]
            t: Time (unused, included for integrator compatibility)

        Returns:
            Derivative vector [dS/dt, dI/dt, dR/dt]
        """
        s, i, r = y
        n = s + i + r

        ds = (-(self.alpha * s * i) / n
              - self.vaccination_rate * s
              - self.death_rate * s
              + self.death_rate * n)
        di = (self.alpha * s * i) / n - (self.recovery_rate + self.death_rate) * i
        dr = self.recovery_rate * i - self.death_rate * r + self.vaccination_rate * s

        return np.array([ds, di, dr])

    def basic_reproduction_number(self):
        """Compute the basic reproduction number R₀.

        R₀ represents the average number of secondary infections caused by
        a single infected individual in a fully susceptible population.

        Returns:
            R₀ = α / (r + d)

        Note:
            - R₀ > 1: epidemic will spread
            - R₀ < 1: epidemic will die out
            - R₀ = 1: endemic equilibrium threshold
        """
        return self.alpha / (self.recovery_rate + self.death_rate)

    def herd_immunity_threshold(self):
        """Compute the herd immunity threshold.

        The fraction of the population that needs to be immune to prevent
        epidemic spread.

        Returns:
            HIT = 1 - 1/R₀

        Raises:
            ValueError: If R₀ <= 1 (no herd immunity needed)
        """
        r0 = self.basic_reproduction_number()
        if r0 <= 1:
            raise ValueError(
                f"R₀ = {r0:.3f} <= 1: disease cannot spread, "
                "no herd immunity threshold exists"
            )
        return 1 - 1 / r0

    def disease_free_equilibrium(self, N):
        """Compute the disease-free equilibrium state.

        When I = 0, the population reaches a stable state with no infection.

        Args:
            N: Total population

        Returns:
            Tuple (S*, I*, R*) at disease-free equilibrium
        """
        if self.vaccination_rate == 0:
            return (N, 0, 0)
        else:
            # With vaccination, some move directly to R
            # At equilibrium: dS/dt = 0 with I=0
            # -vS - dS + dN = 0 => S = dN/(v+d)
            s_star = self.death_rate * N / (self.vaccination_rate + self.death_rate)
            r_star = N - s_star
            return (s_star, 0, r_star)

    def endemic_equilibrium(self, N):
        """Compute the endemic equilibrium state.

        When R₀ > 1, the disease persists at a stable non-zero level.

        Args:
            N: Total population

        Returns:
            Tuple (S*, I*, R*) at endemic equilibrium

        Raises:
            ValueError: If R₀ <= 1 (no endemic equilibrium exists)
        """
        r0 = self.basic_reproduction_number()
        if r0 <= 1:
            raise ValueError(
                f"R₀ = {r0:.3f} <= 1: no endemic equilibrium exists, "
                "disease will die out"
            )

        # At endemic equilibrium with I > 0:
        # From dI/dt = 0: αS*/N = r + d => S* = N(r+d)/α = N/R₀
        s_star = N / r0

        # From dS/dt = 0: -αS*I*/N - vS* - dS* + dN = 0
        # -(r+d)I* - vS* - dS* + dN = 0
        # I* = (dN - vS* - dS*) / (r+d)
        # I* = (d(N - S*) - vS*) / (r+d)
        numerator = (self.death_rate * (N - s_star)
                     - self.vaccination_rate * s_star)
        i_star = numerator / (self.recovery_rate + self.death_rate)

        # If vaccination is too high, I* can become negative
        if i_star < 0:
            raise ValueError(
                "Vaccination rate is high enough to prevent endemic state"
            )

        r_star = N - s_star - i_star

        return (s_star, i_star, r_star)

    def effective_reproduction_number(self, S, N):
        """Compute the effective reproduction number Rₑ.

        Rₑ accounts for the current proportion of susceptibles.

        Args:
            S: Current number of susceptible individuals
            N: Total population

        Returns:
            Rₑ = R₀ × (S/N)
        """
        return self.basic_reproduction_number() * (S / N)

    def __repr__(self):
        return (
            f"SIR(alpha={self.alpha}, recovery_rate={self.recovery_rate}, "
            f"vaccination_rate={self.vaccination_rate}, "
            f"death_rate={self.death_rate})"
        )
