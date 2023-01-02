# denoising.py

Code for FFT denoiser function

# Denoising_demo.py

Demonstrates smoothing methods on one week of data for a consumer (consumption data before and after smoothing)

Methods tested:

- Kalman Filter
- FFT denoising (not clear how to choose the filtering threshold...)
- Rolling average
- Gaussian filter with std 3 and 7

# test_dtw.py

Demonstrates Kmeans with dynamic time warping metrics

# Dimension_Reduction.py

Analyses from Dimension_Reduction_intro.ipynb in .py script

# Dimension_Reduction_intro.ipynb

Demonstrations of the effect of different dimensionality reduction (DR) methods on clustering results via Kmeans.

Combinations tested:

- No DR + Kmeans
- PCA + Kmeans
- MDS + Kmeans
- FFT + Kmeans
- Continuous Wavelet Transform + Kmeans
- Piecewise Constant Approximation

# Outlier_detection.py

Demonstration of outlier detection using Isolation Forest. Plot of first two principal components with outliers highlighted in blue. 

# Seasonality_Investigation.ipynb

Visualization of mean evolution (weekly, monthly, seasonally ie 3 months, whole dataset).

Investigation on the signal means depending on the basis for the mean (weekly, monthly, etc) to study the general trend in the population.

# SOM

Demonstration of SOM clustering with plots of CH and Silhouette index results.

# SSA

Code started for attempting Singular Spectral Analysis (a sort of SVD for time series). Could not run it because of the large dimensionality of the dataset...

# Wavelet_Analysis.ipynb

Demonstrates results for clustering after applying pca, wavelet and autocorrelation transforms. Includes the removal of outliers based on Isolation Forest and series maximum / std values.

Includes:

Outlier removal

Autocorrelation with 1-month lag + Kmeans and Spectral clustering

Coif5, sym10 and db8 wavelets (for different choices of the hyperparameter "level") + Kmeans

PCA + Coif5, sym10 and db8 wavelets (for different choices of the hyperparameter "level") + Kmeans
