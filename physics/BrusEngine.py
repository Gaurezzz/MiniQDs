import mindspore as ms
from mindspore.nn import Cell
from mindspore import ops

class BrusEngine(Cell):
    """
    BrusEngine: A physical modeling engine for Quantum Dot (QD) semiconductors.
    
    This class implements the Brus Equation to calculate the size-dependent energy 
    gap of nanoparticles, incorporating Varshni's Law for temperature correction 
    and a Gaussian distribution for spectral absorption profiles.
    
    Architecture: Modular / Decoupled
    Framework: MindSpore (Optimized for GPU/NPU acceleration)
    """

    def __init__(self, 
                 bandgap: float, 
                 alpha: float, 
                 beta: float, 
                 me_eff: float, 
                 mh_eff: float, 
                 eps_r: float, 
                 max_absorption_coefficient: float = 1e7):
        """
        Initializes the material-specific intrinsic properties.

        Args:
            bandgap (float): Bulk energy gap (E0) at 0 Kelvin [eV].
            alpha (float): Varshni thermal coefficient [eV/K].
            beta (float): Varshni constant related to Debye temperature [K].
            me_eff (float): Effective mass of the electron [relative to m0].
            mh_eff (float): Effective mass of the hole [relative to m0].
            eps_r (float): Relative dielectric constant of the material [dimensionless].
            max_absorption_coefficient (float): Peak absorption value [m^-1]. Default is 1e7.
        """
        super(BrusEngine, self).__init__()
        
        # Material constants
        self.bandgap = bandgap
        self.alpha = alpha
        self.beta = beta
        self.eps_r = eps_r
        self.max_absorption_coefficient = max_absorption_coefficient
        
        # Physical Constants (Standard International Units)
        self.m0 = 9.109e-31                  # Electron rest mass [kg]
        self._reduced_plank = 1.05457e-34    # Reduced Planck's constant (hbar) [J·s]
        self.elementary_charge = 1.602e-19   # Elementary charge (q) [C]
        self.vacuum_permittivity = 8.854e-12 # Epsilon 0 [F/m]
        self.pi = ms.numpy.pi                # Pi constant
        self.plank_constant = 6.626e-34      # Planck's constant (h) [J·s]
        self.light_speed = 2.9979e8          # Speed of light (c) [m/s]
        
        # Derived material properties
        self.me_eff = me_eff * self.m0       # Effective electron mass [kg]
        self.mh_eff = mh_eff * self.m0       # Effective hole mass [kg]
        self.sigma = 10                      # Standard deviation for Gaussian broadening [nm]

    def construct(self, temperature: float, radius: float, wavelengths: ms.Tensor) -> ms.Tensor:
        """
        Computes the absorption coefficient spectrum based on the QD size and temperature.

        Args:
            temperature (float): Operating temperature [K].
            radius (float): Radius of the Quantum Dot [nm].
            wavelengths (ms.Tensor): A 1D tensor of wavelengths to evaluate [nm].

        Returns:
            ms.Tensor: Absorption coefficient profile [m^-1] for the given wavelength range.
        """
        
        # 1. Varshni's Law: Adjust bulk bandgap based on temperature
        # Formula: Eg(T) = E0 - (alpha * T^2) / (T + beta)
        e_bulk = self.bandgap - (self.alpha * ops.pow(temperature, 2)) / (temperature + self.beta)

        # 2. Brus Equation: Add Quantum Confinement and Coulomb interaction
        # Note: radius is converted from [nm] to [m] using 1e-9.
        # The Brus terms (Joules) are divided by elementary_charge to convert to [eV].
        confinement_term = (ops.pow(self._reduced_plank * self.pi, 2)) / \
                           (2 * ops.pow(radius * 1e-9, 2)) * (1/self.me_eff + 1/self.mh_eff)
        
        coulomb_term = (1.786 * ops.pow(self.elementary_charge, 2)) / \
                       (4 * self.pi * self.vacuum_permittivity * self.eps_r * radius * 1e-9)
        
        # Total Energy Gap of the Quantum Dot [eV]
        e_qd = e_bulk + (confinement_term - coulomb_term) / self.elementary_charge

        # 3. Spectral Conversion: Find peak resonance wavelength [nm]
        # Formula: lambda = (h * c) / Energy
        wavelength_peak = (self.plank_constant * self.light_speed) / \
                          (e_qd * self.elementary_charge) * 1e9 

        # 4. Gaussian Absorption Profile: Model the polydispersity of QDs
        # Computes the absorption coefficient for the entire wavelength tensor.
        absorption_coefficient = self.max_absorption_coefficient * ms.numpy.exp(
            -ops.pow((wavelengths - wavelength_peak), 2) / (2 * ops.pow(self.sigma, 2))
        )

        return absorption_coefficient