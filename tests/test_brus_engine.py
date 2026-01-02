import pytest
import pandas as pd
import numpy as np
import mindspore as ms
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from physics.BrusEngine import BrusEngine

def load_material_data():
    """Load material properties from CSV file."""
    csv_path = os.path.join(os.path.dirname(__file__), 'materiales.csv')
    df = pd.read_csv(csv_path)
    return df.to_dict(orient='records')

class TestBrusEngineMaterials:
    
    @pytest.mark.parametrize("material_props", load_material_data())
    def test_material_physics(self, material_props):
        """
        Validates that the engine correctly processes each material from the CSV.
        """
        name = material_props['Material']
        
        engine = BrusEngine(
            bandgap=material_props['Eg_0K_eV'],
            alpha=material_props['Alpha_evK'],
            beta=material_props['Beta_K'],
            me_eff=material_props['me_eff'],
            mh_eff=material_props['mh_eff'],
            eps_r=material_props['epsilon_r']
        )

        temp = ms.Tensor([300.0], ms.float32)
        radius = ms.Tensor([3.0], ms.float32) 
        wavelengths = ms.Tensor(np.arange(200, 3000, 1), ms.float32)

        absorption = engine(temp, radius, wavelengths)

        assert isinstance(absorption, ms.Tensor), f"Error in {name}: Output is not a Tensor"
        
        abs_np = absorption.asnumpy()
        assert not np.isnan(abs_np).any(), f"Error in {name}: NaN values detected"
        
        assert np.max(abs_np) > 0, f"Error in {name}: Absorption is zero across the entire range"

        print(f"Material {name} validated successfully.")

    def test_csv_structure(self):
        """Verifies that the CSV file has all required columns."""
        csv_path = os.path.join(os.path.dirname(__file__), 'materiales.csv')
        df = pd.read_csv(csv_path)
        required_columns = ['Material', 'Eg_0K_eV', 'Alpha_evK', 'Beta_K', 'me_eff', 'mh_eff', 'epsilon_r']
        for col in required_columns:
            assert col in df.columns, f"Missing critical column: {col}"