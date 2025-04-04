from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Self

from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from xarray import DataArray


@dataclass
class BaseFilter(BaseEstimator, TransformerMixin, ABC):
    """Abstract base class for all filters in the cell recruitment pipeline.

    This class implements a scikit-learn style transformer interface for filtering
    cell candidates. It provides a common interface for all filters and handles
    basic validation and preprocessing.

    Methods
    -------
    fit(X, y, **fit_params)
        Fit the filter to the data.

    transform(X, y)
        Apply the filter to the data.

    fit_transform(X, y=None, **fit_params)
        Fit the filter and apply it to the data.

    fit_kernel(X, seeds)
        Abstract method to be implemented by subclasses for fitting logic.

    transform_kernel(X, seeds)
        Abstract method to be implemented by subclasses for transform logic.

    fit_transform_shared_preprocessing(X, seeds)
        Abstract method for preprocessing steps shared between fit and transform.

    Notes
    -----
    This class follows the scikit-learn transformer interface with additional
    functionality specific to cell candidate filtering:

    - Axes validation for xarray DataArrays
    - Support for stateless filters
    - Tracking of fit state
    - Shared preprocessing between fit and transform steps
    - Parallelization across a specified iteration axis

    Subclasses should implement the abstract methods:
    `fit_kernel`, `transform_kernel`, and `fit_transform_shared_preprocessing`.

    See Also
    --------
    sklearn.base.TransformerMixin : Parent class providing fit_transform interface
    sklearn.base.BaseEstimator : Parent class providing get/set parameters interface
    """

    core_axes: list[str] = field(default_factory=lambda: ["width", "height"])
    """The axes in which filters will be applied against."""
    iter_axis: str = "frames"
    """The axis in which the filtering will be parallelized against."""
    spatial_axis: str = "spatial"
    """The multiplexed axis that encompasses the entire visual space of the movie."""
    new_data_in_transform: bool = True
    """Set this to True when transform() will be called on a different dataset than what
    was used in fit(). This ensures fit_transform_shared_preprocessing() runs again during
    transform() to recompute preprocessing attributes for the new data. Set to False if
    transforming the same dataset that was used in fit()."""
    _stateless: ClassVar[bool] = field(default=False, init=False)
    """True if the filter is stateless."""

    def _validate_axes(self, X: DataArray) -> None:
        """Validate that all required axes exist in the input DataArray.

        This method checks that both the core axes (typically spatial dimensions)
        and the iteration axis are present in the input data. It raises a ValueError
        if any required axes are missing.

        Parameters
        ----------
        X : xarray.DataArray
            The input data array to validate. Must contain dimensions matching
            both self.core_axes and self.iter_axis.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If any required axes are missing from the input DataArray.
            The error message will list all missing dimensions and the
            available dimensions in the input.

        Examples
        --------
        >>> filter = MyFilter(core_axes=['height', 'width'], iter_axis='frames')
        >>> data = xr.DataArray(np.random.rand(10, 32, 32),
        ...                     dims=['time', 'y', 'x'])
        >>> filter._validate_axes(data)  # raises ValueError - wrong dimension names

        See Also
        --------
        xarray.DataArray.dims : Dimension names of a DataArray
        """
        missing_axes = []

        # Check core axes
        for axis in self.core_axes:
            if axis not in X.dims:
                missing_axes.append(axis)

        # Check iteration axis
        if self.iter_axis not in X.dims:
            missing_axes.append(self.iter_axis)

        if missing_axes:
            raise ValueError(
                f"DataArray is missing dimensions: {missing_axes}. "
                f"Available dimensions are: {list(X.dims)}"
            )

    @abstractmethod
    def fit_kernel(self, X: DataArray, seeds: DataFrame) -> Any:
        """Core fitting logic to be implemented by each filter subclass.

        This abstract method defines the interface for the fitting step of a filter.
        Subclasses must implement this method to learn any parameters needed for
        filtering from the input data and seeds.

        For stateless filters (_stateless=True), this method can simply pass as
        no fitting is required.

        Parameters
        ----------
        X : xarray.DataArray
            The input calcium imaging video data. Must have dimensions matching
            both self.core_axes and self.iter_axis. Typically has shape
            (frames, height, width).
        seeds : pandas.DataFrame
            DataFrame containing cell candidate locations. Must have columns
            matching self.core_axes (typically 'height' and 'width').

        Returns
        -------
        None
            This method modifies the filter instance in-place by setting
            learned parameters as instance attributes.

        Notes
        -----
        This method is called by the public `fit` method after axes validation
        and shared preprocessing. Implementations should:

        - Learn parameters specific to the filtering strategy
        - Set those parameters as instance attributes
        - Not modify the input data or seeds
        - Not perform validation (handled by parent class)
        - Not return anything (modifications should be in-place)

        For stateless filters, this method can be implemented as:
        >>> def fit_kernel(self, X, seeds):
        ...     pass

        See Also
        --------
        transform_kernel : Corresponding method for applying the learned parameters
        fit_transform_shared_preprocessing : Method for shared preprocessing steps
        """
        pass

    @abstractmethod
    def transform_kernel(self, X: DataArray, seeds: DataFrame) -> Any:
        """Core transformation logic to be implemented by each filter subclass.

        This abstract method defines the interface for the transformation step
        of a filter. Subclasses must implement this method to apply the learned
        parameters to filter cell candidates.

        Parameters
        ----------
        X : xarray.DataArray
            The input calcium imaging video data. Must have dimensions matching
            both self.core_axes and self.iter_axis. Typically has shape
            (frames, height, width).
        seeds : pandas.DataFrame
            DataFrame containing cell candidate locations. Must have columns
            matching self.core_axes (typically 'height' and 'width').

        Returns
        -------
        pandas.DataFrame
            The filtered seeds DataFrame with an additional boolean mask column
            indicating which candidates passed the filter. The mask column
            should be named 'mask_{filter_name}' where filter_name is unique
            to each filter implementation.

        Notes
        -----
        This method is called by the public `transform` method after axes validation
        and optional shared preprocessing. Implementations should:

        - Apply the filtering strategy using parameters learned in fit_kernel
        - Not modify the input data or seeds
        - Not perform validation (handled by parent class)
        - Return a new DataFrame or a copy with the additional mask column
        - Use consistent naming for the mask column across all calls

        For stateless filters (_stateless=True), this method may be called
        without a preceding call to fit_kernel.

        See Also
        --------
        fit_kernel : Corresponding method for learning filter parameters
        fit_transform_shared_preprocessing : Method for shared preprocessing steps
        """
        pass

    def fit(self, X: DataArray, y: DataFrame, **fit_params: dict) -> Self:
        """Fit the filter to the input data.

        This method handles the common fitting pipeline for all filters:
        1. Validates input data dimensions
        2. Performs shared preprocessing steps
        3. Calls the filter-specific fitting logic

        WARNING: Filter developers should NOT override this method. Instead,
        implement the abstract methods fit_kernel and fit_transform_shared_preprocessing.
        This ensures consistent behavior across all filters.

        Parameters
        ----------
        X : xarray.DataArray
            The input calcium imaging video data. Must have dimensions matching
            both self.core_axes and self.iter_axis. Typically has shape
            (frames, height, width).
        y : pandas.DataFrame
            DataFrame containing cell candidate locations. Must have columns
            matching self.core_axes (typically 'height' and 'width').
        **fit_params : dict
            Additional parameters passed to the fit method.
            Currently unused, maintained for scikit-learn compatibility.

        Returns
        -------
        self : BaseFilter
            The fitted filter instance.

        Raises
        ------
        ValueError
            If the input data is missing required dimensions.

        Notes
        -----
        This method is decorated with @track_calls which sets _has_been_fitted
        to True after successful completion. For stateless filters, this still
        sets the flag even though no actual fitting occurs.

        The fitting process consists of three steps:
        1. _validate_axes: Ensures input data has correct dimensions
        2. fit_transform_shared_preprocessing: Performs any needed preprocessing
        3. fit_kernel: Learns filter-specific parameters

        Developer Note
        -------------
        This is a template method that should NOT be modified in filter implementations.
        To customize fitting behavior:
        - Implement fit_kernel for filter-specific parameter learning
        - Implement fit_transform_shared_preprocessing for custom preprocessing
        - Set _stateless=True for filters that don't require fitting

        See Also
        --------
        transform : Apply the fitted filter to new data
        fit_transform : Convenience method to fit and transform in one step
        fit_kernel : Method to implement filter-specific fitting logic
        """
        self._validate_axes(X)
        self.fit_transform_shared_preprocessing(X=X, seeds=y)
        self.fit_kernel(X=X, seeds=y)
        self._is_fitted = True
        return self

    def transform(self, X: DataArray, y: DataFrame) -> Any:
        """Apply the fitted filter to new data.

        This method handles the common transformation pipeline for all filters:
        1. Validates input data dimensions
        2. Checks if the filter has been fitted
        3. Optionally performs shared preprocessing if reusing fit
        4. Calls the filter-specific transformation logic

        WARNING: Filter developers should NOT override this method. Instead,
        implement the abstract method transform_kernel. This ensures consistent
        behavior across all filters.

        Parameters
        ----------
        X : xarray.DataArray
            The input calcium imaging video data. Must have dimensions matching
            both self.core_axes and self.iter_axis. Typically has shape
            (frames, height, width).
        y : pandas.DataFrame
            DataFrame containing cell candidate locations. Must have columns
            matching self.core_axes (typically 'height' and 'width').

        Returns
        -------
        pandas.DataFrame
            The filtered seeds DataFrame with an additional boolean mask column
            indicating which candidates passed the filter.

        Raises
        ------
        ValueError
            If the input data is missing required dimensions.
        NotFittedError
            If the filter requires fitting and has not been fitted yet.
            Does not apply to stateless filters.

        Notes
        -----
        The transformation process consists of these steps:
        1. _validate_axes: Ensures input data has correct dimensions
        2. Check fitted state: Ensures fit() was called if needed
        3. fit_transform_shared_preprocessing: Optionally rerun if new_data_in_transform=True
        4. transform_kernel: Apply filter-specific transformation

        If new_data_in_transform=True (default), shared preprocessing will be rerun on the
        new data. Set new_data_in_transform=False if the transform data is the same as
        the fit data to avoid redundant preprocessing.

        Developer Note
        -------------
        This is a template method that should NOT be modified in filter implementations.
        To customize transformation behavior:
        - Implement transform_kernel for filter-specific logic
        - Implement fit_transform_shared_preprocessing for custom preprocessing
        - Set _stateless=True for filters that don't require fitting

        See Also
        --------
        fit : Fit the filter to training data
        fit_transform : Convenience method to fit and transform in one step
        transform_kernel : Method to implement filter-specific transformation logic
        """
        self._validate_axes(X)
        check_is_fitted(self)

        if self.new_data_in_transform:
            self.fit_transform_shared_preprocessing(X=X, seeds=y)

        return self.transform_kernel(X=X, seeds=y)

    def fit_transform(self, X: DataArray, y: DataFrame = None, **fit_params: dict) -> Any:
        """Fit the filter to the data and apply transformation in one step.

        This is a convenience method that chains fit() and transform() together.

        Parameters
        ----------
        X : DataArray
            Input data array containing fluorescence values with dimensions matching
            core_axes and iter_axis.
        y : DataFrame, optional
            DataFrame containing seed coordinates in columns matching core_axes.
        **fit_params : dict
            Additional parameters passed to fit().

        Returns
        -------
        pandas.DataFrame
            Seeds DataFrame with an additional boolean mask column indicating valid seeds.

        See Also
        --------
        fit : Fit the filter to training data
        transform : Apply the fitted filter to new data
        """
        return self.fit(X, y, **fit_params).transform(X, y)

    @abstractmethod
    def fit_transform_shared_preprocessing(self, X: DataArray, seeds: DataFrame) -> Any:
        """Perform shared preprocessing steps for both fit and transform operations.

        This method is called during both fit() and transform() to handle any preprocessing
        steps that need to be consistent between training and transformation. Override this
        method to implement custom preprocessing logic.

        Parameters
        ----------
        X : DataArray
            Input data array containing fluorescence values with dimensions matching
            core_axes and iter_axis.
        seeds : DataFrame
            DataFrame containing seed coordinates in columns matching core_axes.

        Notes
        -----
        This method is called before transform_kernel() during transform() operations,
        and before fit_kernel() during fit() operations.
        """
        pass

    def __sklearn_is_fitted__(self) -> bool:
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted
