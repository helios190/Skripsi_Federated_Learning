# Federated Learning Differential Privacy with Server & Client Adaptive & Fixed Clipping

## Overview

This project demonstrates the implementation of **Federated Learning (FL)** with **Differential Privacy (DP)** using clipping strategies applied at the **server** and **client** sides. Experiments are conducted with both **adaptive** and **fixed** clipping mechanisms. The focus is on ensuring privacy through controlled gradients during FL while leveraging **LSTM models** for the analysis.

## Directory Structure

### Project Directories

1. **`clients/`**  
   - Contains client implementations for Federated Learning experiments.  
   - Subfolders:
     - `FedDFClientAdapClip`: Client code for adaptive clipping.
     - `FedDFClientFixedClip`: Client code for fixed clipping.
     - `FedDFServerAdapClip`: Contains client-specific experiments related to server-side adaptive clipping.
     - `FedDFServerFixedClip`: Contains client-specific experiments related to server-side fixed clipping.

2. **`server/`** 
   - Contains server-side logic for FL experiments.  
   - Subfolders:
     - `FedDFServerAdapClip`: Implements adaptive clipping on the server.
     - `FedDFServerFixedClip`: Implements fixed clipping on the server.

3. **`utils/`**  
   - Includes utility functions:
     - Splitting datasets.
     - LSTM implementation and preprocessing.
     - Data Balancing Process - Random Oversampling

4. **`data/`**  
   - Directory for storing input data for the experiments.

5. **`results/`**  
   - Saves the output from the FL experiments, including logs and aggregated models.

6. **`visualizer/`**  
   - Tools for visualizing results (e.g., performance metrics, graphs).

7. **`analysis.ipynb`**  
   - Jupyter notebook for analyzing the FL results.

8. **`README.md`**  
   - This file.

9. **`requirements.txt`**  
   - List of required Python dependencies.

### Workflow

1. **Server Initialization**  
   - Navigate to the appropriate folder under `server/` and run the server-side script.
   - Example:  
     ```bash
     python FedDFServerAdapClip/server.py
     ```

2. **Client Execution**  
   - Navigate to the corresponding folder under `clients/` and run the client scripts.
   - Example:  
     ```bash
     python FedDFClientAdapClip/client1.py
     python FedDFClientAdapClip/client2.py
     ```

3. **Results**  
   - After running both server and client scripts, results will be automatically saved under the `results/` directory.

## Clipping Mechanisms

### Fixed Clipping
- Clipping gradients with a constant threshold.

### Adaptive Clipping
- Dynamically adjusts the clipping threshold based on observed gradients.

## Running the Project

### Prerequisites
1. Python >= 3.8.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Execution
1. Start the **server** (adaptive or fixed).
2. Run **clients** in parallel.
3. Analyze results using `analysis.ipynb`.

## References

- Federated Learning with Differential Privacy: [McMahan et al. (2017)](https://arxiv.org/abs/1702.06087)  
- Adaptive Clipping: [Sun et al. (2021)](https://arxiv.org/abs/2106.02965)  

Feel free to reach out for further queries or clarifications!
