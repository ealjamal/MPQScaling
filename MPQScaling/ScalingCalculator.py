import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
import kllr as kl
import itertools as it


# Set the units of scatter to be natural log.
kl.set_params(scatter_factor = np.log(10)) 

class ScalingCalculator:
    '''
        Initialize the ScalingCalculator as a context for KLLR where we
        take inputs for the KLLR settings and define the default attributes
        that result from the KLLR analysis.
        
        --------
        Params
        --------
        
        properties: (list of str)
            The list of properties that are columns in the data frame that will
            be used in the analysis.
            
        scale_var: (str)
            The column name in the data that will be used as the scale variable
            (the independent variable in the KLLR analysis.)
            
        bins: (int)
            The number of bins that KLLR will break the range of scale_var in
            order to do weighted linear regression in each bin. Default is
            10 bins.

        xrange: ((2,) list, tuple or numpy array)
            The minimum and maximum range in the scale_var that will be used
            in the KLLR analysis. Default is None which means that the minimum
            and maximum range in the scale_var data will be used.
        
        nBootstrap: (int)
            The number of bootstrap realizations to determine the statistical
            error. Default is 100 realizations.

        percentile: ((2,) list, tuple or numpy array)
            The percentiles that will be used in calculating the bootstrap 
            statistical error. Default is 1sigma error [16., 84.].

        kernel_type: (str)
            The kernel type that will be used to determine the weights in the
            weighted linear regression. Possible choices are
            ['gaussian', 'tophat]. Default is 'gaussian'.

        kernel_width: (float)
            The width of the kernel, default is 0.2.
        
        seed: (int)
            The random seed that will be used in the realization of MC sampling
            of the covariance to calculate the mpq statistical error.

        verbose: (boolean)
            Controls the verbosity of the analysis output. Default is True.       
       
       '''
    
    
    def __init__(self, properties, scale_var, bins = 10, xrange = None,
                 nBootstrap = 100, percentile = [16., 84.],
                 kernel_type = "gaussian", kernel_width = 0.2,
                 seed = None, verbose = True):
        
        self.properties = properties
        self.p = len(self.properties) # number of properties used in the analysis
        # KLLR settings.
        self.scale_var = scale_var # scale variable
        self.bins = bins # number of bins 
        self.xrange = xrange
        self.nBootstrap = nBootstrap # number of Bootstrap samples
        self.percentile = percentile # percentile range for bootstrap uncertainty
        self.kernel_type = kernel_type # kernel type, 'gaussian' or 'top hat'
        self.kernel_width = kernel_width # kernel width
        self.seed = seed # random generator seed for monte carlo
        self.verbose = verbose # verbosity control

        # The following attributes will be initialized to zero and filled by the
        # KLLR analysis.

        # Binned KLLR parameters of normalization, slope, and scatter (natural log)
        self.scale = np.zeros(shape = (1, self.bins)) # the binned scale variable.
        self.norms = np.zeros(shape = (self.p, self.bins)) # the KLLR normalizations
        self.norms_minus = np.zeros(shape = (self.p, self.bins)) # KLLR normalization minus uncertainties
        self.norms_plus = np.zeros(shape = (self.p, self.bins)) # KLLR normalization plus uncertainties
        self.slopes = np.zeros(shape = (self.p, self.bins)) # KLLR slopes
        self.slopes_minus = np.zeros(shape = (self.p, self.bins)) # KLLR slope minus uncertainties
        self.slopes_plus = np.zeros(shape = (self.p, self.bins)) # KLLR slope plus uncertainties
        self.scatters = np.zeros(shape = (self.p, self.bins)) # KLLR scatters
        self.scatters_minus = np.zeros(shape = (self.p, self.bins)) # KLLR scatter minus uncertainities
        self.scatters_plus = np.zeros(shape = (self.p, self.bins)) # KLLR scatter plus uncertainties

        # Binned covariance matrices.
        self.x = np.zeros(shape = (1, self.bins)) # the binned scale that results from covariance calculations, should be the same as self.scale
        self.C = np.zeros(shape = (self.p, self.p, self.bins)) # KLLR covariances between properties for each bin
        self.C_minus = np.zeros(shape = (self.p, self.p, self.bins))# KLLR covariance minus uncertainties
        self.C_plus = np.zeros(shape = (self.p, self.p, self.bins)) # KLLR covariance plus uncertainties

        # Binned correlation matrix.
        self.x_corr = np.zeros(shape = (1, self.bins)) # the binned scale that results from correlation calculations, should be the same as self.scale
        self.corr = np.zeros(shape = (self.p, self.p, self.bins)) # KLLR correlation between properties for each bin
        self.corr_minus = np.zeros(shape = (self.p, self.p, self.bins))# KLLR correlation minus uncertainties
        self.corr_plus = np.zeros(shape = (self.p, self.p, self.bins)) # KLLR correlation plus uncertainties
        
        
    def calculate_kllr_parameters(self, data):
        '''
        Performs KLLR for each property to calculate the scaling parameters.
        
        --------
        Params
        --------

        data: (pd.DataFrame)
            Data Frame containing all properties of interest.

        '''

        # If xrange is None, then set the bounds to be min and max of x variable.
        if self.xrange is None:
            self.xrange = (np.min(data[f"{self.scale_var}"]),
                           np.max(data[f"{self.scale_var}"]))


        for k, prop in enumerate(self.properties):
            if self.verbose:
                print(f"\n\033[33mLoading KLLR parameters data for property {k}: {prop}...\033[0m")
            fit_data, _ = kl.Plot_Fit_Summary(df = data, xlabel = self.scale_var, ylabel = prop,
                                              bins = self.bins, xrange = self.xrange, nBootstrap = self.nBootstrap,
                                              percentile = self.percentile, kernel_type = self.kernel_type,
                                              kernel_width = self.kernel_width, verbose = self.verbose)
            plt.close()

            # Store KLLR results in attributes.
            self.scale = fit_data["x"]
            self.norms[k, :] = fit_data["y"]
            self.norms_minus[k, :] = fit_data["y-"]
            self.norms_plus[k, :] = fit_data["y+"]
            self.slopes[k, :] = fit_data["slope"]
            self.slopes_minus[k, :] = fit_data["slope-"]
            self.slopes_plus[k, :] = fit_data["slope+"]
            self.scatters[k, :] = fit_data["scatter"]
            self.scatters_minus[k, :] = fit_data["scatter-"]
            self.scatters_plus[k, :] = fit_data["scatter+"]

    
    def calculate_covariance_matrix(self, data):
        '''
        Calculate the binned covariance matrix of the self.p properties.

        --------
        Params
        --------

        data: (pd.DataFrame)
            Data frame containing all properties of interest.
            
        '''


        # If xrange is None, then set the bounds to be min and max of x variable.
        if self.xrange is None:
            self.xrange = (np.min(data[f"{self.scale_var}"]),
                           np.max(data[f"{self.scale_var}"]))

        if self.verbose:
            print(f"\n\033[36mCalculating covariances of {self.p} properties...\033[0m")

        pairs = list(it.combinations(range(self.p), 2)) # Define possible pairs of indices in covariance matrix.
        for i, j in pairs:
            prop1, prop2 = self.properties[i], self.properties[j]

            cov, _ = kl.Plot_Cov_Corr_Matrix(data, xlabel = self.scale_var, ylabels = [prop1, prop2], 
                                             bins = self.bins, xrange = self.xrange, nBootstrap = self.nBootstrap, 
                                             Output_mode = "Covariance", kernel_type = self.kernel_type, kernel_width = self.kernel_width,
                                             percentile = self.percentile, verbose = self.verbose)
            plt.close()
            
            # Store the covariance matrices in attributes.
            # Nominal covariance values, remembering covariance is symmetric.
            self.C[i, i, :] = cov["cov_" + prop1 + "_" + prop1]
            self.C[i, j, :] = self.C[j, i, :] = cov["cov_" + prop1 + "_" + prop2]
            self.C[j, j, :] = cov["cov_" + prop2 + "_" + prop2]

            # Covariance values for the minus percentiles.
            self.C_minus[i, i, :] = cov["cov_" + prop1 + "_" + prop1 + "-"]
            self.C_minus[i, j, :] = self.C_minus[j, i, :] = cov["cov_" + prop1 + "_" + prop2 + "-"]
            self.C_minus[j, j, :] = cov["cov_" + prop2 + "_" + prop2 + "-"]

            # Covariance values for the plus percentile.
            self.C_plus[i, i, :] = cov["cov_" + prop1 + "_" + prop1 + "+"]
            self.C_plus[i, j, :] = self.C_plus[j, i, :] = cov["cov_" + prop1 + "_" + prop2 + "+"]
            self.C_plus[j, j, :] = cov["cov_" + prop2 + "_" + prop2 + "+"]
        
        # This is the scale that results from KLLR covariance calculation,
        # It should be the same as self.scale because the xrange and number of bins is the same.
        self.x = cov["x"]

    
    def calculate_correlation_matrix(self, data):
        '''
        Calculate the binned correlation matrix of the self.p properties.

        --------
        Params
        --------

        data: (pd.DataFrame)
            Data frame containing all properties of interest.
            
        '''


        # If xrange is None, then set the bounds to be min and max of x variable.
        if self.xrange is None:
            self.xrange = (np.min(data[f"{self.scale_var}"]),
                           np.max(data[f"{self.scale_var}"]))

        if self.verbose:
            print(f"\n\033[36mCalculating correlations of {self.p} properties...\033[0m")

        pairs = list(it.combinations(range(self.p), 2)) # Define possible pairs of indices in correlation matrix.
        for i, j in pairs:
            prop1, prop2 = self.properties[i], self.properties[j]

            corr, _ = kl.Plot_Cov_Corr_Matrix(data, xlabel = self.scale_var, ylabels = [prop1, prop2], 
                                              bins = self.bins, xrange = self.xrange, nBootstrap = self.nBootstrap, 
                                              Output_mode = "Correlation", kernel_type = self.kernel_type, kernel_width = self.kernel_width,
                                              percentile = self.percentile, verbose = self.verbose)
            plt.close()
            
            # Store the covariance matrices in attributes.
            # Nominal covariance values, remembering covariance is symmetric.
            self.corr[i, i, :] = 1
            self.corr[i, j, :] = self.corr[j, i, :] = corr["corr_" + prop1 + "_" + prop2]
            self.corr[j, j, :] = 1

            # Covariance values for the minus percentiles.
            self.corr_minus[i, i, :] = 1
            self.corr_minus[i, j, :] = self.corr_minus[j, i, :] = corr["corr_" + prop1 + "_" + prop2 + "-"]
            self.corr_minus[j, j, :] = 1

            # Covariance values for the plus percentile.
            self.corr_plus[i, i, :] = 1
            self.corr_plus[i, j, :] = self.corr_plus[j, i, :] = corr["corr_" + prop1 + "_" + prop2 + "+"]
            self.corr_plus[j, j, :] = 1
        
        # This is the scale that results from KLLR covariance calculation,
        # It should be the same as self.scale because the xrange and number of bins is the same.
        self.x_corr = corr["x"]

    
    @staticmethod
    def _mpq(slopes, inverse_C):
        '''
            Convenience method that calculates the mpq from the slope and inverse
            covariance matrix according to the equation
            mpq = (slopes^T * C^{-1}* slopes)^{-1/2}.

            --------
            Params
            --------

            slopes (numpy array):
                Array of KLLR slopes of shape (self.p, self.b) that result
                from the KLLR analysis.
            
            inverse_C:
                The binned inverse covariances of shape (self.p, self.p, self.b)
                the inverse is taken with respect to the first to dimensions of
                the covariance matrices from KLLR.
            
            --------
            Output
            --------

            The mass proxy quality as defined by the above equation. If x variable
            is other than mass then it is no longer interpreted as the mass proxy
            quality.
           
        '''


        inv_C_slope_prod = np.einsum('ptb,pb ->tb', inverse_C, slopes) # calculate C^{-1} * slopes, result shape: (p, b).
        mass_variance = 1/(np.einsum('pb,pb->b', slopes, inv_C_slope_prod)) # calculate slopes * (C^{-1} * slopes), result shape: (b, )
        mpq = np.sqrt(mass_variance) # calculate mpq from mass variance

        return mpq
    
    
    @staticmethod
    def _sample_covariance(C, C_err):
        '''
            Convenience function that samples from the covariance error that
            result from KLLR to find perturbations of the covariance matrix
            that are symmetric and positive definite.

            --------
            Params
            --------

            C: (numpy array)
                Binned covariance matrices from KLLR.
            
            C_err: (numpy array)
                Matrix of errors in each entry of the binned covariance matrices.

            --------
            Output
            --------

            The perturbed covariance matrix sampled from the error.
            
        '''


        # Helper functions for providing a symmetric matrix and checking if it is positive definite.
        def make_symmetric(matrix):
            # Given a matrix, it makes it symmetric by making the upper triangle
            # equal to the lower triangle. Another method is to average the sum
            # of the matrix and its transpose, but that would alter the distribution of errors.

            # np.triu(slice_2d) will give us the upper triangle with the diagonals included with 0 below the diagonal
            # np.triu(slice_2d, k = 1) will give us the upper triangle only with diagonal and below the diagonal 0
            # Adding these two gives a symmetric matrix.
            symmetric_mat = np.triu(matrix) + np.triu(matrix, k = 1).T

            return symmetric_mat
        
        
        def is_positive_definite(matrix):
            # Return whether the matrix is positive definite or not
            # by checking if the eigenvalues are all positive.
            e_vals, _ = eigh(matrix) # calculate eigenvalues of matrix
            return (np.min(e_vals) > 0) # check if the minimum eigenvalue is positive
        
        # Sample from covariance matrices in each bin, ensuring that each matrix is symmetric and positive definite.
        # If it is not positive definite, we sample until we find one that is positive definite. Another method 
        # to ensure that a matrix is positive definite is to use the Cholesky decomposition where we first replace the non-positive
        # eigenvalues with smaller positive eignevalues and use the decomposition. But, after testing, this was 
        # very unstable on values of mpq.

        # initialize the perturbed covariance matrix, that is a sample from the errors of covariance.
        pert_C = np.zeros_like(C)
        for k in range(C.shape[-1]):
            # For each bin, we sample from a random normal distribution with mean 0 (we will add to nominal covariance later)
            # and with errors for the entries given by entries in C_err. After symmetrizing the matrix, we skip
            # over all perturbed covariance matrices that are not positive definite.
            while True:
                C_perturbations = np.random.normal(np.zeros_like(C[:, :, k]), C_err[:, :, k]) # sample from normal distribution for perturbations
                symm_perturbations  = make_symmetric(C_perturbations) # make perturbations symmetric
                potential_pert_C = C[:, :, k] + symm_perturbations # add perturbation to nominal covariance to find perturbed covariance
                if is_positive_definite(potential_pert_C): # skip over all matrices that are not positive definite
                    pert_C[:, :, k] = C[:, :, k] + symm_perturbations
                    break

        return pert_C
    

    def _mpq_samples(self, slopes, C, slopes_err, C_err):
        '''
        Sample covariance and slopes from the error given from KLLR and use
        it to calculate the MPQ for each sample with statistical error.
        Returns the MPQ samples, slope samples, and covariance samples from MC.
        
        --------
        Params
        --------

        slopes (numpy array):
            Array of KLLR slopes of shape (self.p, self.b) that result
            from the KLLR analysis.

        C: (numpy array)
            Binned covariance matrices from KLLR of shape (self.p, self.p, self.b).

        slopes (numpy array):
            Array of KLLR slopes errors of shape (self.p, self.b) that result
            from the KLLR analysis.

        C_err: (numpy array)
            Matrix of errors in each entry of the binned covariance matrices
            of shape (self.p, self.p, self.b).

        --------
        Output
        --------

        mpq_samples: (numpy array)
            Array of shape (nBootstrap,) that holds the mpq calculated from
            the perturbed covariance and perturbed slopes. This will be
            used to determine the median MPQ and the statistical error
            around the median.

        slope_samples: (numpy array)
            Array of shape (nBootstrap, self.p, self.b) containing the
            perturbed slopes used in the mpq_samples calculation representing
            the perturbed slopes of each property in each bin.

        C_samples: (numpy array)
            Array of shape (nBootstrap, self.p, self.p, self.b)
            containing the perturbed covariances used in the mpq_samples
            calculation representing the perturbed covariances matrix in each bin.
            
        '''

        
        # Set seed for MC samples for the errors.
        np.random.seed(self.seed)
        # This empty list will hold the result mpq from the MC.
        mpq_samples = [] # empty list will hold the result mpqs from the MC, shape: (nBootstrap, p, bins)
        slope_samples = [] # empty list will hold the result slopes from the MC. shape: (nBootstrap, p, bins)
        C_samples = [] # empty list will hold the result covariance tensor from the MC. shape: (nBootstrap, p,  p, bins)
        # The reason why we use a loop is that _sample_covariance cannot be vectorized since we have the positive
        # definite check on the matrix.
        for _ in range(self.nBootstrap):
            slopes_perturbations = np.random.normal(np.zeros_like(slopes), slopes_err) # create slope perturbations or sample
            pert_slopes = slopes + slopes_perturbations # add to nominal values of slope to find the perturbed slopes
            slope_samples.append(pert_slopes) # add perturbed slope to slope samples list
            pert_C = self._sample_covariance(C, C_err) # sample covariance to find perturbed covariance
            C_samples.append(pert_C)  # add perturbed covariance to covariance samples list

            # Check if the matrix is invertible by numpy, pert_C is guaranteed to be positive definite and therfore,
            # invertible but this will serve as a check.
            try:
                inv_C_pert = np.linalg.inv(pert_C.T).T
            except np.linalg.LinAlgError:
                continue

            sample_mpq = self._mpq(pert_slopes, inv_C_pert) # calculate mpq for this sample and add it to the list of samples
            mpq_samples.append(sample_mpq) # add mpq sample to mpq samples list
        
        # Check if the MC yields valid samples.
        mpq_samples = np.array(mpq_samples)
        if mpq_samples.size == 0:
            raise ValueError("All matrix inversions failed. Check the perturbations.")


        return mpq_samples, np.array(slope_samples), np.array(C_samples)
    

    def mpq_combo(self, num_props_in_combo):
        '''
        Calculate the binned MPQ for any combination of num_props_in_combo
        properties. Returns a dictionary of the MPQ median, lower and upper
        percentiles, MPQ samples, slope samples, and covariance samples
        from MC for each combination of properties.

        --------
        Params
        --------

        num_props_in_combo: (int)
            The number of properties in the combination of properties. This
            will calculate the mpq implied by any combination (of 
            num_props_in_combo properties). For example if num_props_in_combo = 1,
            then we will calculate the individual mpqs of all properties in
            self.properties. If num_props_in_combo = 2, we will calculate
            the combined mpq in any pair of properties in self.properties.

        --------
        Output
        --------

        A dictionary with median, lower, upper percentile mpq and the mpq
        bootstrap samples used to calculate the median and the errors of the
        combination of properties denoted by a string of integers of the index
        of the properties. For example, if 
        properties = ["M_gas", "M_star", "M_BH", "M_DM"], then mpq_0 is
        the mpq of "M_gas" and mpq_012 is the mpq of the combination of
        "M_gas", "M_star", "M_BH" properties. Along with the mpq samples, 
        we return the slope and covariance samples that were used to calculate
        the mpq samples from the MC.

        '''

        assert (self.p >= num_props_in_combo) # check if the passed number of
                                              # props for a combination is less
                                              #  than the total number of properties.

        
        # Dictionary of results that will hold the median mpq and the mpq percentiles.
        res = dict()
        combos = list(it.combinations(range(self.p), num_props_in_combo)) # List of possible combinations of length num_props_in_combo

        # For each combination calculate the mpq with a reference given by the index of the property in original properties array.
        for combo in combos:
            key = "".join(map(str, combo)) # this is the key for the combination, if we wanted a combination of the first
                                           # four properties, the key would be '0123'

            C_combo = self.C[combo, :, :][:, combo, :] # pick out a submatrix of the covariance relevant for the properties in this combination for each bin
            avg_C_err = (np.abs(C_combo - self.C_minus[combo, :, :][:, combo, :]) + 
                         np.abs(C_combo - self.C_plus[combo, :, :][:, combo, :]))/2.0 # we don't consider assymetric slopes here, so we average

            slopes_combo = self.slopes[combo, :] # pick out a subvector of the slopes relevant for the properties in this combination for each bin
            avg_slopes_err = (np.abs(slopes_combo - self.slopes_minus[combo, :]) + 
                              np.abs(slopes_combo - self.slopes_plus[combo, :]))/2.0 # we don't consider assymetric slopes here, so we average
            
            # Calculate the mpq including the percentiles, and get the samples for mpq, slope and covariance
            mpq_samples, slope_samples, C_samples = self._mpq_samples(slopes_combo, C_combo, avg_slopes_err, avg_C_err)

            # Calculate the median, lower percentile, and upper percentile of the mpq using mpq samples.
            mpq_median = np.percentile(mpq_samples, 50, axis = 0) # calculate median mpq from samples
            mpq_minus = np.percentile(mpq_samples, self.percentile[0], axis = 0) # calculate mpq of lower percentile
            mpq_plus = np.percentile(mpq_samples, self.percentile[1], axis = 0) # calculate mpq of upper percentile

            # Save mpq single values and samples to return.
            res["mpq_" + key] = mpq_median # median mpq
            res["mpq-_" + key] = mpq_minus # lower percentile of mpq
            res["mpq+_" + key] = mpq_plus # upper percentile of mpq
        
            res["mpq_samples_" + key] = np.array(mpq_samples) # mpq samples from MC, shape (nBootstrap, p, bins)
            res["slope_samples_" + key] = np.array(slope_samples) # slope samples from MC, shape (nBootstrap, p, bins)
            res["C_samples_" + key] = np.array(C_samples) # covariance samples from MC, shape (nBootstrap, p, p, bins)
        

        return res
    
    
    def get_kllr_parameters(self):
        '''
        
        Accesses and returns binned KLLR parameters in a dictionary.

        --------
        Output
        --------
        
        A dictionary of KLLR parameters. 
        
        scale - the binned values of the scale variable.
            norms - the normalizations of each property shape (self.p, self.b)
            i.e. the first dimension is for each property and the second
            is for each bin.
        norms- - the normalizations of the lower percentile, similar to norms.
        norms+ - the normalizations of the upper percentile, similar to norms.
        slopes - the slopes of each property shape (self.p, self.b)
                i.e. the first dimension is for each property and the second
                is for each bin.
        slopes- - the slopes of the lower percentile, similar to slopes.
        slopes+  - the slopes of the upper percentile, similar to slopes.
        scatters - the scatters (natural log units) of each property
                    shape (self.p, self.b) i.e. the first dimension is
                    for each property and the second is for each bin.
        scatters- - the scatters of the lower percentile, similar to scatters.
        scatters+ - the scatters of the upper percentile, similar to scatters.

        '''

        res = dict()

        res['scale'] = self.scale.copy()
        res['norms'] = self.norms.copy()
        res['norms-'] = self.norms_minus.copy()
        res['norms+'] = self.norms_plus.copy()
        res['slopes'] = self.slopes.copy()
        res['slopes-'] = self.slopes_minus.copy()
        res['slopes+'] = self.slopes_plus.copy()
        res['scatters'] = self.scatters.copy()
        res['scatters-'] = self.scatters_minus.copy()
        res['scatters+'] = self.scatters_plus.copy()

        return res
    

    def get_covariance(self):
        '''
        Accesses and returns binned KLLR covariances in a dictionary.

        --------
        Output
        --------

        A dictionary of KLLR parameters. 
            scale - the binned values of the scale variable.
            cov - binned covariance matrices of the properties given in
                  ScalingCalculator, shape (self.p, self.p, self.b),
                  the first two dimensions give the covariance matrix
                  at a fixed bin, and the last is for different bins.
            cov- - the covariances of the lower percentile, similar to cov.
            cov+ - the covariances of the upper percentile, similar to cov.

        '''

        res = dict()

        res["scale"] = self.x.copy()
        res["cov"] = self.C.copy()
        res["cov-"] = self.C_minus.copy()
        res["cov+"] = self.C_plus.copy()

        return res

    
    def get_correlation(self):
        '''
        Accesses and returns binned KLLR covariances in a dictionary.

        --------
        Output
        --------

        A dictionary of KLLR parameters. 
            scale - the binned values of the scale variable.
            corr - binned correlation matrices of the properties given in
                   ScalingCalculator, shape (self.p, self.p, self.b),
                   the first two dimensions give the correlation matrix
                   at a fixed bin, and the last is for different bins.
            corr- - the correlations of the lower percentile, similar to corr.
            corr+ - the correlations of the upper percentile, similar to corr.

        '''

        res = dict()

        res["scale"] = self.x_corr.copy()
        res["corr"] = self.corr.copy()
        res["corr-"] = self.corr_minus.copy()
        res["corr+"] = self.corr_plus.copy()

        return res
    

    def clear_params(self):
        '''
        Clears all KLLR calculations of parameters, covariances, and correlations.
        
        '''

        self.scale = np.zeros(shape = (1, self.bins)) 
        self.norms = np.zeros(shape = (self.p, self.bins))
        self.norms_minus = np.zeros(shape = (self.p, self.bins)) 
        self.norms_plus = np.zeros(shape = (self.p, self.bins)) 
        self.slopes = np.zeros(shape = (self.p, self.bins)) 
        self.slopes_minus = np.zeros(shape = (self.p, self.bins)) 
        self.slopes_plus = np.zeros(shape = (self.p, self.bins)) 
        self.scatters = np.zeros(shape = (self.p, self.bins)) 
        self.scatters_minus = np.zeros(shape = (self.p, self.bins)) 
        self.scatters_plus = np.zeros(shape = (self.p, self.bins)) 


        self.x = np.zeros(shape = (1, self.bins)) 
        self.C = np.zeros(shape = (self.p, self.p, self.bins)) 
        self.C_minus = np.zeros(shape = (self.p, self.p, self.bins))
        self.C_plus = np.zeros(shape = (self.p, self.p, self.bins)) 

        self.x_corr = np.zeros(shape = (1, self.bins)) 
        self.corr = np.zeros(shape = (self.p, self.p, self.bins)) 
        self.corr_minus = np.zeros(shape = (self.p, self.p, self.bins))
        self.corr_plus = np.zeros(shape = (self.p, self.p, self.bins)) 