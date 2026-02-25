# Epidemic Dynamics: SIR Model

A compartmental model for disease spread with epidemiological analysis, numerical simulation, and economic cost optimization.

## Mathematical Formulation

### The SIR Model

The population is divided into three compartments:

- **S (Susceptible)**: individuals who can become infected
- **I (Infected)**: individuals who are currently infected and infectious
- **R (Recovered)**: individuals who have recovered and are immune

The dynamics follow the system of ODEs:

$$
\begin{aligned}
\frac{dS}{dt} &= -\frac{\alpha S I}{N} - v S - d S + d N \\
\frac{dI}{dt} &= \frac{\alpha S I}{N} - (r + d) I \\
\frac{dR}{dt} &= r I - d R + v S
\end{aligned}
$$

where:

- $\alpha$ = infection rate (transmission coefficient)
- $r$ = recovery rate
- $v$ = vaccination rate
- $d$ = death rate (assumed equal to birth rate for constant population)
- $N = S + I + R$ = total population

### Basic Reproduction Number

The **basic reproduction number** $R_0$ is the average number of secondary infections caused by a single infected individual in a fully susceptible population:

$$R_0 = \frac{\alpha}{r + d}$$

- $R_0 > 1$: epidemic spreads
- $R_0 < 1$: epidemic dies out

### Herd Immunity Threshold

The fraction of the population that needs to be immune to prevent spread:

$$\text{HIT} = 1 - \frac{1}{R_0}$$

## Usage

### Running the Notebook

```bash
cd epidemic-dynamics
pip install -r requirements.txt
jupyter lab disease_dynamics.ipynb
```

### Using the Modules

```python
from sir_model import SIR
from integrators import RK4, Integrator
import numpy as np

# Create model
sir = SIR(alpha=0.3, recovery_rate=0.1, vaccination_rate=0, death_rate=0.01)

# Epidemiological analysis
print(f"R₀ = {sir.basic_reproduction_number()}")  # 2.73
print(f"Herd immunity threshold = {sir.herd_immunity_threshold():.1%}")  # 63.4%

# Simulate epidemic
y0 = np.array([999, 1, 0])  # Initial state: 999 susceptible, 1 infected
method = RK4(sir)
integrator = Integrator(method, y0, t_min=0, t_max=365, n_steps=2000)
dynamics = integrator.integrate()
time = integrator.get_time()

# dynamics[:, 0] = susceptible over time
# dynamics[:, 1] = infected over time
# dynamics[:, 2] = recovered over time
```

### Cost Optimization

```python
from cost_analysis import optimize_vaccination_rate

base_params = {'alpha': 0.3, 'recovery_rate': 0.1, 'death_rate': 0.01}
result = optimize_vaccination_rate(
    base_params, y0, t_max=365, n_steps=2000,
    vaccine_cost=40, health_cost=10,
    v_min=0, v_max=0.05
)
print(f"Optimal vaccination rate: {result['optimal_rate']:.3%}")
```

## Modules

- **sir_model.py**: SIR model class with epidemiological methods
  - `basic_reproduction_number()`: Compute $R_0$
  - `herd_immunity_threshold()`: Compute HIT
  - `disease_free_equilibrium(N)`: Compute DFE state
  - `endemic_equilibrium(N)`: Compute endemic state

- **integrators.py**: Numerical integration schemes
  - `ExplicitEuler`: First-order forward Euler
  - `RK2`: Second-order Runge-Kutta (midpoint)
  - `RK4`: Fourth-order Runge-Kutta
  - `Integrator`: Generic wrapper for time-stepping

- **cost_analysis.py**: Economic cost optimization
  - `compute_epidemic_cost()`: Total cost calculation
  - `optimize_vaccination_rate()`: Find optimal vaccination strategy
  - `cost_sensitivity_analysis()`: Cost breakdown across strategies

## Dependencies

- numpy
- scipy
- matplotlib
- jupyter

## Author

Ntovoris Eleftherios
