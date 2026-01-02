import os
import pandas as pd
from mindspore import Tensor, dtype as mstype
from mindspore.nn import Cell
from pvlib import spectrum
from mindspore import ops

class SolarPerformanceEvaluator(Cell):
    """
    Evaluates the photovoltaic performance of Quantum Dot Solar Cells (QDSCs).
    
    This engine calculates the Short-Circuit Current (J_sc), Open-Circuit Voltage (V_oc),
    and Power Conversion Efficiency (PCE) for a batch of solar cell designs.
    It implements a multi-objective fitness function that maximizes efficiency while
    penalizing current mismatch in tandem structures.
    """

    def __init__(self, kappa: float = 0.5):
        """
        Initializes the physical constants and loads the solar spectrum.

        Args:
            kappa (float): Penalty coefficient for current mismatch in the fitness function.
                           Controls how strictly the algorithm enforces current matching.
        """
        super(SolarPerformanceEvaluator, self).__init__()

        # --- Physical Constants & Control Parameters ---
        self.ENERGY_CONST = 1.9864e-16      # hc in J*m
        self.ELECTRON_CHARGE = 1.60218e-19  # q in Coulombs
        self.FF = 0.75                      # Fill Factor (estimated standard)
        self.kappa = kappa                  # Optimization penalty weight

        # --- Load AM1.5G Standard Spectrum ---
        # We load the global irradiance data to simulate real-world conditions
        am15 = spectrum.get_reference_spectra()
        am15['global'] = am15['global'].astype('float32')

        self.global_irradiance = Tensor(am15['global'].values, mstype.float32)
        self.wavelengths = Tensor(am15.index.values, mstype.float32)

        # --- Pre-calculate Photon Flux ---
        # Convert Irradiance (W/m^2/nm) to Photon Flux (photons/s/m^2/nm)
        # Formula: Phi = Irradiance / (hc / lambda)
        energy = self.ENERGY_CONST / self.wavelengths
        self.photon_flux = self.global_irradiance / energy

        # Calculate spectral step size (delta lambda) for Riemann sum integration
        self.delta = Tensor(am15.index[1] - am15.index[0], mstype.float32)

        # Calculate total incident solar power (P_sun) ~ 100 mW/cm^2
        self.p_sun = (self.global_irradiance * self.delta).sum()

    def construct(self, absorption_coefficient: Tensor, e_qd: Tensor) -> Tensor:
        """
        Computes the fitness score for a batch of solar cell individuals simultaneously.

        Args:
            absorption_coefficient (Tensor): Shape (Batch, Layers, Wavelengths). 
                                             The spectral absorption profile for each layer.
            e_qd (Tensor): Shape (Batch, Layers). 
                           Quantum Dot Bandgaps in eV for each layer.

        Returns:
            Tensor: Shape (Batch,). The calculated fitness score for each individual in the population.
        """
        
        # 1. Calculate Short-Circuit Current (J_sc) for every layer
        # We integrate over the wavelengths (axis=-1) to get the total current per layer.
        # Result shape: (Batch, Layers)
        j_layers = self.ELECTRON_CHARGE * (self.photon_flux * self.delta * absorption_coefficient).sum(axis=-1)

        # 2. Calculate Open-Circuit Voltage (V_oc)
        # Estimation: V_oc approx (E_g / q) - 0.4V loss.
        # For tandem cells in series, voltages add up. We sum across layers (axis=1).
        v_layers = e_qd - 0.4
        v_oc_total = v_layers.sum(axis=1)

        # 3. Apply Current Matching Condition
        # In a series connection, the total current is limited by the layer generating the least current.
        # We take the minimum across layers (axis=1).
        j_sc_limit = j_layers.min(axis=1)

        # 4. Calculate Power Conversion Efficiency (PCE)
        # Eta = (J_sc * V_oc * FF) / P_in
        efficiency = (j_sc_limit * v_oc_total * self.FF) / self.p_sun

        # 5. Calculate Current Mismatch Penalty
        # We penalize designs where layers generate vastly different currents.
        # Uses vector slicing to compute sum(|J_i - J_{i+1}|) across layers.
        diff_j = ops.abs(j_layers[:, 1:] - j_layers[:, :-1]).sum(axis=1)
        
        # Final Fitness Calculation
        # Fitness = Efficiency - (Penalty * Mismatch)
        fitness = efficiency - self.kappa * diff_j

        return fitness