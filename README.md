# PINN-based Online PDE Calculator

A web-based application built with Dash and Physics-Informed Neural Networks (PINNs) for solving partial differential equations (PDEs) with high precision. This tool allows you to configure problem parameters, launch training, and visualize results all within an interactive UI.

---

## Features

* **Problem Setup**: Define custom PDEs in symbolic form and specify boundary/initial conditions.
* **Adaptive Sampling**: Configure collocation, boundary, and additional sample points.
* **Network Configuration**: Set network depth, width, and training epochs (Adam + L-BFGS).
* **Visualization**: Interactive tabs for:

  * Collocation points
  * Solution vs. residuals
  * Error heatmaps
  * Loss curves (total/data/equation)
  * Boundary loss
  * Frequency spectrum
* **Real-time Logging**: Training logs streamed to the UI with auto-scroll.
* **Modular Codebase**: Clear separation of layout, callbacks, figures, and utilities for maintainability.

---

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Cc1-Yy/PINN-based-online-PDE-calculator.git
   cd PINN-based-online-PDE-calculator
   ```
2. **Create a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # Unix/Mac
   venv\Scripts\activate    # Windows
   ```
3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. **Prepare data directory**: Ensure the `data/` folder exists and is writable for `.npz` outputs and logs.
2. **Run the application**:

   ```bash
   # From project root
   python -m pinn_app.app
   ```
3. **Access in browser**: Open `http://127.0.0.1:8050` (or the URL shown in console).
4. **Configure and train**:

   * Enter PDE and boundary conditions.
   * Adjust sample points and network settings.
   * Click **Start Training** to begin.
   * Monitor training logs and view updated plots.

---

## Project Structure

```
PINN-based-online-PDE-calculator/
├── data/                  # Output data and logs
├── pinn_app/              # Main application package
│   ├── __init__.py        # App factory and exports
│   ├── app.py             # Entry-point script
│   ├── constants.py       # Global constants
│   ├── layout.py          # Dash layout and clientside callbacks
│   ├── figures.py         # Figure-generation functions
│   ├── logger.py          # Logging handlers and redirection
│   ├── utils.py           # Utility functions (e.g., get_fig)
│   └── callbacks/         # Dash callbacks modules
│       ├── __init__.py    # Registers all callbacks
│       ├── input_validation.py
│       ├── bd_groups.py
│       ├── training.py
│       └── result_graph.py
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```
