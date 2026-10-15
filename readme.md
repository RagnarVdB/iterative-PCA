# Iterative PCA detrending for exoplanet spectra at high-spectral resolution

This is a Python implementation of the iterative PCA-based detrending algorithm for removing stellar and telluric features from spectroscopic exoplanet by [Van den Broeck et al. (2026)](). The method can be used as a substitute for SysRem or classical PCA detrending and should provide a cleaner separation of the planet signal, with the downsides of increased noise and more parameters to tune.

We provide two interfaces, a low-level one that operates directly on a 2D data matrix that has already been resampled to a log-equidistant wavelength grid, and a higher level one that operates on a 3D data cube containing all echelle orders and handles interpolating to the required wavelength grid and splitting along nodding positions.

### Example

```python
from iterative_PCA import IterativePCADetrender
import numpy as np
from matplotlib import pyplot as plt

# 3D data cube with normalized flux values
dataset = np.load("W127b_CRIRES+_transit.npy")
wavelengths = np.load("CRIRES+_wavelengths.npy")
phases = np.load("W127b_phases.npy")
barycentric = np.load("W127b_barycentric.npy")
# Array of A-B nodding positions
nodding = np.load("W127b_nodding.npy")
# Boolean array with wavelengths that need to be masked
masks = np.load("W127b_masks.npy")

### Iterative-PCA
expected_Kp = 133
planet_velocities = expected_Kp * np.sin(phases) + barycentric
itPCA = IterativePCADetrender(
    wavelengths,
    planet_velocities,
    transit_start=24, # first in-transit exposure
    transit_end=76, # last in-transit + 1
    n_components=4,
    max_iter=60,
    delta_wl=7e-7, # regrid with this spacing in log10(lambda)
    nodding=nodding,
    masks=masks,
)
processed_dataset, info = itPCA(dataset)
# The dataset has been regridded to a log-equidistant wavelength_grid itPCA.output_wavelengths.

fig, (ax1, ax2) = plt.subplots(1, 2)
order = 0
ax1.pcolormesh(itPCA.input_wavelengths, phases, dataset[:, order, :])
ax2.pcolormesh(itPCA.output_wavelengths, phases, processed_dataset[:, order, :])

```

### High level interface

```python
class IterativePCADetrender(
    input_wavelengths: np.ndarray | list[np.ndarray],
    planet_velocities: np.ndarray,
    transit_start: int,
    transit_end: int,
    n_components: int = 4,
    max_iter: int = 100,
    delta_wl: float = 7e-7,
    nodding: np.ndarray | None = None,
    masks: list[np.ndarray] | None = None,
    intermediate_saves: list[int] | np.ndarray = [],
    convergence_quantile: float | None = None,
    convergence_threshold: float | None = None
)
```

Initialize the IterativePCADetrender.

- `wavelengths` : np.ndarray  
   2D array or list of wavelengths (n_orders, n_wavelengths).
- `planet_velocities` : np.ndarray  
   1D array of planet velocities.
- `transit_start` : int  
   Index of the start of the transit (first in-transit exposure).
- `transit_end` : int  
   Index of the end of the transit (first out-of-transit exposure).
- `n_components` : int, optional  
   Number of principal components to use.
- `max_iter` : int, optional  
   Maximum number of iterations.
- `delta_wl` : float, optional  
   Wavelength step size. Input data matrix will be interpolated to this step size.
- `nodding` : np.ndarray, optional  
   Array of nodding values A|B. Data will be split into two groups based on this
  array and PCA will be performed separately on each group.
- `masks` : list[np.ndarray], optional  
   Boolean masks for every order.
- `intermediate_saves` : list[int] | np.ndarray, optional  
   List of intermediate save points.
- `convergence_quantile` : float, optional  
   Quantile for convergence checking
- `convergence_threshold` : float, optional  
   Threshold for convergence checking.

```python
def iterativePCA.__call__(
    self,
    data: np.ndarray,
    guess: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    progress: bool=False,
)
```

### Low level interface

```python
class IterativePCA(
    n_components: int,
    max_iter: int = 100,
    intermediate_saves: list[int] | np.ndarray | None = None,
    convergence_quantile: float | None = None,
    convergence_threshold: float | None = None,
)
```
Apply the iterative PCA detrending to the input data.


Initialize the IterativePCA class.

- `n_components` : int  
  Number of principal components fitted. As with classical PCA-detrending, typical values are 3-9
- `max_iter` : int, optional  
   Maximum number of iterations. Typically 40-100. Visual inspection of the CCF after different numbers of iterations is recommended to check convergence.
- `intermediate_saves` : list[int] | np.ndarray, optional  
   List of iteration numbers at which to save intermediate results.
- `convergence_quantile` : float, optional  
   The quantile to use for the convergence check. The algorithm will stop once a fraction `convergence_quantile` of all pixels change by less than `convergence_threshold` compared to the previous iteration.
- `convergence_threshold` : float, optional  
   The threshold to use for the convergence check.

```python
IterativePCA.__call__(
    self,
    X: np.ndarray,
    pixel_shifts: np.ndarray,
    mask: np.ndarray | None = None,
    transit_start: int = 0,
    transit_end: int = -1,
    guess: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
)
```

Perform iterative PCA.

- `X` : np.ndarray  
   Input data matrix of shape (n_timesteps, n_wavelengths).
- `pixel_shifts` : np.ndarray
  Number of pixels each spectrum must be shifted to move to planet rest frame.
- `mask` : np.ndarray, optional  
   Boolean mask for which wavelengths to include in the PCA.
- `transit_start` : int, optional  
   Index of the start of the transit.
- `transit_end` : int, optional  
   Index of the end of the transit (first out-of-transit exposure).
- `guess` : tuple[np.ndarray, np.ndarray, np.ndarray], optional
