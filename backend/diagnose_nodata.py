import os
from main import feature_data_arrays, _get_pixel, feature_names
import numpy as np

lat = 27.891538339750078
lng = 85.78948974609376

with open("diag_out.txt", "w", encoding="utf-8") as f:
    f.write(f"Testing Lat: {lat}, Lng: {lng}\n")
    try:
        row, col = _get_pixel(lat, lng)
        f.write(f"Row: {row}, Col: {col}\n")
        
        has_nan = False
        for fname in feature_names:
            val = float(feature_data_arrays[fname][row, col])
            is_nan = np.isnan(val)
            f.write(f"  {fname}: {val} (NaN: {is_nan})\n")
            if is_nan:
                has_nan = True
                
        if has_nan:
            f.write("\nCONCLUSION: 'No data' error is caused by NaN in one or more features at this pixel.\n")
        else:
            f.write("\nCONCLUSION: No NaN found. The point should be predictable.\n")
            
    except Exception as e:
        f.write(f"Error: {e}\n")
