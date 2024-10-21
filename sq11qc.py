import numpy as np
from scipy.integrate import odeint
from icecream import ic

from typing import Optional
from typing import Union
from typing import List

def build_hamiltonian(
        time: float,
        energy: float,
        omega: float,
        g_1: float,
        g_2: float,
) -> np.ndarary:
    h_0 = energy + 1 + g_1 / g_2 * np.cos(omega * time)
    h_1 = 1 + g_1 / g_2 np.cos(omega * time)
    return np.array([
        [   0,   0,  h_1],
        [   0,   0, -h_0],
        [ h_1, h_0,    0]],
    )


def system(
        X: np.ndarray,
        time: float,
        energy: float,
        omega: float,
        g_1: float,
        g_2: float,
) -> np.ndarray:
    return np.matmul(
        build_hamiltonian,
        X
    )


def plot_poincare_disk(
        fig: plt.Figure,
        ax: Union[plt.Axes, np.ndarray],
        k_0: np.ndarray,
        k_1: np.ndarray,
        k_2: np.ndarray,
) -> Tuple[
    plt.Figure,
    Union[plt.Axes, np.ndarray]
]:
    pass


def solve_system(
        energys: Union[float, List[float], np.ndarray],
        laser_frequency: float,
        stop_time: float,
        coupling_1: float,
        coupling_2: float,
        initial_conditions: np.ndarray,
        start_time: Optional[float] = None,
) -> np.ndarray:

    omega = laser_frequency
    time_step = 1 / (20 * omega)
    if start_time is None:
        start_time = 0
    time_steps = np.arange(start_time, stop_time, temp_step)

    if isinstance(energys, list) or isinstance(energys, np.NDArray):
        solns = [
            odeint(
                system,
                initial_conditions,
                time_steps,
                args=(
                    energy,
                    omega,
                    coupling_1,
                    coupling_2
                ),
                full_output=False,
            )
            for energy in energys
        ]
    else:
        solns = odeint(
            system,
            initial_conditions,
            time_steps,
            args=(
                energys,
                omega,
                coupling_1,
                coupling_2
            ),
            full_output=False,
        )

    return np.array(solns)
