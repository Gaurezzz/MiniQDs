import pytest
import os
from pvlib import spectrum
from physics.SolarLoader import SolarLoader

class TestSolarLoader:
    def test_solar_loader(self):
        """Test the SolarLoader class for correct data loading and processing."""
        solar_loader = SolarLoader()
        flux_data = solar_loader.photon_flux.asnumpy()

        assert flux_data.dtype == 'float32', "Photon flux data type should be float32"
        assert flux_data.size > 0, "Photon flux data should not be empty"
        assert (flux_data >= 0).all(), "Photon flux should be greater than zero" 
        assert flux_data.mean() > 0, "Mean photon flux should be greater than zero"
        

