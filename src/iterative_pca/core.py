import numpy as np
from numba import njit
from sklearn.decomposition import PCA
from tqdm import trange


@njit
def update_pca(Xwork, W, H):
    n_samples = Xwork.shape[0]
    n_features = Xwork.shape[1]
    n_components = H.shape[0]

    HHt = H @ H.T
    HXt = H @ Xwork.T
    for i in range(n_samples):
        for t in range(n_components):
            grad = -HXt[t, i]
            for r in range(n_components):
                grad += W[i, r] * HHt[r, t]
            hess = HHt[t, t]
            W[i, t] -= grad / hess

    WtW = W.T @ W
    WtX = W.T @ Xwork
    for j in range(n_features):
        for t in range(n_components):
            grad = -WtX[t, j]
            for r in range(n_components):
                grad += H[r, j] * WtW[r, t]
            hess = WtW[t, t]
            H[t, j] -= grad / hess


@njit
def make_planet_matrix2(
    n_samples,
    n_features,
    residual,
    pixel_shifts,
    mask,
    transit_start,
    transit_end,
):
    """
    Take the PCA subtracted data matrix, average the in-transit planet signal along the
    expected Doppler shift and produce the 2D matrix returned to Earth frame
    """
    pixel_shifts_in_transit = pixel_shifts[transit_start:transit_end]
    planet_spec_1D = np.zeros(
        n_features
        + int(
            np.floor(pixel_shifts_in_transit.max())
            - np.floor(pixel_shifts_in_transit.min())
        )
        + 1
    )
    offset = int(-pixel_shifts_in_transit.min()) + 1
    counts = np.zeros_like(planet_spec_1D)
    for i in range(transit_start, transit_end):
        first = offset + int(np.floor(pixel_shifts[i])) + 1
        last = first + n_features - 1
        w = pixel_shifts[i] - np.floor(pixel_shifts[i])
        wi = 1 - w
        for j in range(first, last):
            if mask[j - first] and mask[j - first + 1]:
                x = w * residual[i, j - first] + wi * residual[i, j - first + 1]
                if np.isnan(x):
                    print(
                        "nan1",
                        w,
                        residual[i, j - first],
                        residual[i, j - first + 1],
                    )
                planet_spec_1D[j] += x
                counts[j] += 1

    planet_spec_1D /= counts
    planet_spec_1D[counts == 0] = 0

    planet_matrix = np.zeros((n_samples, n_features))
    for i in range(transit_start, transit_end):
        sf = int(np.floor(pixel_shifts[i]))
        w = pixel_shifts[i] - np.floor(pixel_shifts[i])
        wi = 1 - w
        for j in range(n_features):
            if mask[j]:
                p1 = planet_spec_1D[j + sf + offset]
                p2 = planet_spec_1D[j + sf + offset + 1]
                if not np.isnan(p1) and not np.isnan(p2):
                    planet_matrix[i, j] = wi * p1 + w * p2
                elif not np.isnan(p1):
                    planet_matrix[i, j] = p1
                elif not np.isnan(p2):
                    planet_matrix[i, j] = p2
                else:
                    planet_matrix[i, j] = 0

    return planet_matrix, planet_spec_1D


@njit
def exec_pca(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    pixel_shifts: np.ndarray,
    max_iter: int,
    checkpoints: list[int] | np.ndarray,
    mask: np.ndarray | None = None,
    transit_start: int = 0,
    transit_end: int = -1,
    guess_tell_matrix: np.ndarray | None = None,
    convergence_quantile: float | None = None,
    convergence_threshold: float | None = None,
):
    n_samples = X.shape[0]
    n_features = X.shape[1]
    if mask is None:
        mask = np.ones(n_features, dtype=np.bool_)
    if transit_end == -1:
        transit_end = n_samples
    residual_vec = np.zeros((len(checkpoints), X.shape[0], X.shape[1]))
    planets = np.zeros((len(checkpoints), X.shape[0], X.shape[1]))
    shifts_transit = pixel_shifts[transit_start:transit_end]
    planets_flat = np.zeros(
        (
            len(checkpoints),
            n_features
            + int(np.floor(shifts_transit.max()) - np.floor(shifts_transit.min()))
            + 1,
        )
    )
    pca_reconstructions = np.zeros((len(checkpoints), X.shape[0], X[:, mask].shape[1]))
    full_resid_vec = np.zeros((len(checkpoints), X.shape[0], X[:, mask].shape[1]))

    nrng = 0
    Xwork = np.copy(X)

    if guess_tell_matrix is not None:
        residual = np.zeros_like(X)
        residual[:, mask] = guess_tell_matrix - X[:, mask]
        planet, planet_flat = make_planet_matrix2(
            n_samples,
            n_features,
            residual,
            pixel_shifts,
            mask,
            transit_start,
            transit_end,
        )
        Xwork = X + planet

    Xwork_cut = np.copy(Xwork[:, mask])
    mean = np.sum(Xwork_cut, axis=0) / n_samples
    converged = False
    for it in range(max_iter):
        if converged:
            break
        mean = np.sum(Xwork_cut, axis=0) / n_samples
        Xwork_cut -= mean
        update_pca(Xwork_cut, W, H)
        Xwork_cut += mean

        pca_reconstruction = W @ H + mean
        residual = np.zeros_like(X)
        residual[:, mask] = pca_reconstruction - X[:, mask]
        if np.any(np.isnan(residual)):
            print("nans in residuals")
            print(np.where(np.isnan(W)))
            print(np.where(np.isnan(H)))
            break
        else:
            planet, planet_flat = make_planet_matrix2(
                n_samples,
                n_features,
                residual,
                pixel_shifts,
                mask,
                transit_start,
                transit_end,
            )

        Xwork = X + planet
        # Convergence check
        if convergence_quantile is not None and convergence_threshold is not None:
            diff = np.abs(Xwork[:, mask] - Xwork_cut)
            threshold_value = np.quantile(diff, convergence_quantile)
            if threshold_value < convergence_threshold:
                converged = True

        Xwork_cut = np.copy(Xwork[:, mask])

        if it in checkpoints:
            i = np.searchsorted(checkpoints, it)
            residual_vec[i] = residual
            planets[i] = planet
            pca_reconstructions[i] = pca_reconstruction
            full_resid_vec[i] = pca_reconstruction - Xwork_cut
            planets_flat[i] = planet_flat
            nrng += 1

    return (
        H,
        W,
        residual_vec[: nrng + 1],
        planets[: nrng + 1],
        planets_flat[: nrng + 1],
        pca_reconstructions,
        full_resid_vec,
        mean,
        it,
    )


class IterativePCA:
    def __init__(
        self,
        n_components: int,
        max_iter: int = 100,
        intermediate_saves: list[int] | np.ndarray | None = None,
        convergence_quantile: float | None = None,
        convergence_threshold: float | None = None,
    ):
        """
        Initialize the IterativePCA class.

        Parameters
        ----------
        n_components : int
            Number of principal components fitted.
        max_iter : int, optional
            Maximum number of iterations.
        intermediate_saves : list[int] | np.ndarray | None, optional
            List of iteration numbers at which to save intermediate results.
        convergence_quantile : float | None, optional
            The quantile to use for the convergence check.
        convergence_threshold : float | None, optional
            The threshold to use for the convergence check.
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.intermediate_saves = intermediate_saves
        self.convergence_quantile = convergence_quantile
        self.convergence_threshold = convergence_threshold
        if bool(self.convergence_quantile) ^ bool(self.convergence_threshold):
            raise ValueError(
                "Both convergence_quantile and convergence_threshold must be provided together."
            )

    def __call__(
        self,
        X: np.ndarray,
        pixel_shifts: np.ndarray,
        mask: np.ndarray | None = None,
        transit_start: int = 0,
        transit_end: int = -1,
        guess: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ):
        """
        Perform iterative PCA on the input data.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix of shape (n_timesteps, n_wavelengths).
        pixel_shifts : np.ndarray
            Number of pixels each spectrum must be shifted to move to planet rest frame.
        mask : np.ndarray, optional
            Boolean mask for which wavelengths to include in the PCA.
        transit_start : int, optional
            Index of the start of the transit.
        transit_end : int, optional
            Index of the end of the transit (first out-of-transit exposure).
        guess : tuple[np.ndarray, np.ndarray, np.ndarray], optional

        """
        assert X.ndim == 2, "Input data X must be a 2D array."
        assert pixel_shifts.ndim == 1, "Pixel shifts must be a 1D array."
        assert X.shape[0] == pixel_shifts.shape[0], (
            "Number of samples in X must match length of pixel_shifts."
        )
        if mask is None:
            mask = np.ones(X.shape[1], dtype=bool)
        X = np.ascontiguousarray(X)
        if guess is None:
            Xwork_cut = X[:, mask]
            P = PCA(self.n_components)
            W = np.ascontiguousarray(P.fit_transform(Xwork_cut))
            H = np.ascontiguousarray(P.components_)
            guess_matrix = None
        else:
            W, H, mu = guess
            guess_matrix = W @ H + mu

        (
            H,
            W,
            residual_vec,
            planets,
            planets_flat,
            pca_reconstructions,
            full_resid_vec,
            mean,
            it,
        ) = exec_pca(
            X,
            W,
            H,
            pixel_shifts,
            self.max_iter,
            [] if self.intermediate_saves is None else self.intermediate_saves,
            mask,
            transit_start,
            transit_end,
            guess_matrix,
            self.convergence_quantile,
            self.convergence_threshold,
        )

        reconstructed = W @ H + mean
        corrected = X - reconstructed
        extra_info = {}
        if self.intermediate_saves is not None:
            extra_info["iterations"] = residual_vec
        return corrected, extra_info


@njit
def resample(old_wl, flux, new_wl):
    assert old_wl.shape == flux.shape
    out = []
    i = 0
    for nwl in new_wl:
        while old_wl[i] < nwl:
            i += 1

        weight = (old_wl[i] - nwl) / (old_wl[i] - old_wl[i - 1])
        out.append(flux[i - 1] * (weight) + flux[i] * (1 - weight))
    return np.array(out)


@njit()
def resample_mask(old_wl, mask, new_wl):
    out = []
    i = 0
    for nwl in new_wl:
        while old_wl[i] < nwl:
            i += 1

        out.append(mask[i - 1] and mask[i])
    return np.array(out)


class IterativePCADetrender:
    def __init__(
        self,
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
        convergence_threshold: float | None = None,
    ):
        """
        Initialize the IterativePCADetrender.

        Parameters
        ----------
        wavelengths : np.ndarray
            2D array or list of wavelengths (n_orders, n_wavelengths).
        planet_velocities : np.ndarray
            1D array of planet velocities.
        transit_start : int
            Index of the start of the transit (first in-transit exposure).
        transit_end : int
            Index of the end of the transit (first out-of-transit exposure).
        n_components : int, optional
            Number of principal components to use.
        max_iter : int, optional
            Maximum number of iterations.
        delta_wl : float, optional
            Wavelength step size. Input data matrix will be interpolated to this step size.
        nodding : np.ndarray | None, optional
            Array of nodding values A|B. Data will be split into two groups based on this
            array and PCA will be performed separately on each group.
        masks : list[np.ndarray] | None, optional
            Boolean masks for every order.
        intermediate_saves : list[int] | np.ndarray | None, optional
            List of intermediate save points.
        convergence_quantile : float | None, optional
            Quantile for convergence checking.
        convergence_threshold : float | None, optional
            Threshold for convergence checking.
        """
        if masks is not None:
            assert len(masks) == len(input_wavelengths), (
                "Number of masks must match number of wavelength arrays."
            )
            assert all(
                mask.shape[0] == wl.shape[0]
                for mask, wl in zip(masks, input_wavelengths)
            ), (
                "Each mask must have the same length as the corresponding wavelength array."
            )
            assert all(mask.dtype == bool for mask in masks), (
                "Masks must be boolean arrays."
            )
        if nodding is not None:
            assert len(nodding) == len(planet_velocities), (
                "Length of nodding array must match length of planet_velocities (number of exposures)."
            )

        self.processor = IterativePCA(
            n_components,
            max_iter,
            intermediate_saves,
            convergence_quantile,
            convergence_threshold,
        )

        self.input_wavelengths = input_wavelengths
        self.planet_velocities = planet_velocities
        self.transit_start = transit_start
        self.transit_end = transit_end
        self.n_components = n_components
        self.delta_wl = delta_wl
        self.nodding = nodding
        self.masks = masks
        self.max_iter = max_iter
        self.intermediate_saves = intermediate_saves
        self.convergence_quantile = convergence_quantile
        self.convergence_threshold = convergence_threshold

        self.output_wavelengths = [
            10
            ** np.arange(
                np.log10(wli[0]),
                np.log10(wli[-1]),
                self.delta_wl,
            )
            for wli in input_wavelengths
        ]
        if masks is not None:
            self.masks_resampled = [
                resample_mask(
                    input_wavelengths[i], masks[i], self.output_wavelengths[i]
                )
                for i in range(len(input_wavelengths))
            ]
        else:
            self.masks_resampled = [
                np.ones_like(self.output_wavelengths[i], dtype=bool)
                for i in range(len(input_wavelengths))
            ]

    def __call__(self, data: np.ndarray, guess=None, progress=False):
        processed = []
        iterations = []
        iteration_counts = []
        for i in trange(data.shape[1], leave=False, disable=not progress):
            target_wl = self.output_wavelengths[i]
            d = data[:, i, :]
            data_resampled = np.zeros((d.shape[0], target_wl.size))
            for l in range(d.shape[0]):
                data_resampled[l, :] = resample(
                    self.input_wavelengths[i], d[l, :], target_wl
                )
            mask_resampled = np.copy(self.masks_resampled[i])
            data_resampled[:, ~mask_resampled] = 0
            out = np.zeros((data.shape[0], np.sum(mask_resampled)))
            its = np.zeros(
                (len(self.intermediate_saves), data.shape[0], np.sum(mask_resampled))
            )

            for n, nod in enumerate(["A", "B"]):
                nmsk = self.nodding == nod
                transit_start = np.sum(nmsk[: self.transit_start])
                transit_end = np.sum(nmsk[: self.transit_end])
                pixel_shifts = (
                    -np.log10(1 + self.planet_velocities[nmsk] / 299792.458)
                    / self.delta_wl
                )

                (
                    H,
                    W,
                    resids,
                    planet,
                    planet_flat,
                    pca_rec,
                    full_resids,
                    mean,
                    it_count,
                ) = self.processor(
                    data_resampled[nmsk],
                    pixel_shifts,
                    mask=mask_resampled,
                    transit_start=transit_start,
                    transit_end=transit_end,
                    guess=None if guess is None else guess[i][n],
                )
                iteration_counts.append(it_count)
                reconstructed = W @ H + mean
                out[nmsk] = data_resampled[nmsk, :][:, mask_resampled] - reconstructed
                if do_rng:
                    for k in range(len(resids)):
                        its[k, nmsk] = (-resids[k][..., mask_resampled],)

            processed.append(out)
            iterations.append(its)
        return (
            processed,
            iterations if do_rng else None,
            {"iteration_counts": iteration_counts},
        )
