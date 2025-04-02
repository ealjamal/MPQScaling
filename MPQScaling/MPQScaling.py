import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import eigh
import kllr as kl
import itertools as it


# Set the units of scatter to be natural log.
kl.set_params(scatter_factor = np.log(10)) 

class MPQScaling:
    '''
    This class sets the KLLR parameters by initalization and has method and 
    retreive scaling parameters, covariance, correlations and MPQs.

    '''
    
    def __init__(self, properties, scale_var, bins = 10, xrange = None,
                 nBootstrap = 100, percentile = [16., 84.],
                 kernel_type = "gaussian", kernel_width = 0.2,
                 seed = None, verbose = True):
        
        '''
        Initialize the MPQScaling class as a context for KLLR where we
        take inputs for the KLLR settings and initialize the attributes filled
        with zeros and will be populated by the KLLR analysis.
        
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
        
        self.properties = properties
        self.p = len(self.properties) # number of properties used in the analysis
        # KLLR settings.
        self.scale_var = scale_var # scale variable
        self.bins = bins # number of bins 
        self.xrange = xrange # scale_var range for KLLR analysis
        self.nBootstrap = nBootstrap # number of Bootstrap samples
        self.percentile = percentile # percentile range for bootstrap uncertainty
        self.kernel_type = kernel_type # kernel type, 'gaussian' or 'top hat'
        self.kernel_width = kernel_width # kernel width
        self.seed = seed # random generator seed for monte carlo
        self.verbose = verbose # verbosity control

        # The following attributes will be initialized to zero and filled by the KLLR analysis.

        # Binned KLLR parameters of normalization, slope, and scatter (natural log)
        self.scale = np.zeros(shape = (1, self.bins)) # the binned scale variable.
        self.norms = np.zeros(shape = (self.p, self.bins)) # the KLLR normalizations
        self.norms_minus = np.zeros(shape = (self.p, self.bins)) # KLLR normalization lower percentile
        self.norms_plus = np.zeros(shape = (self.p, self.bins)) # KLLR normalization upper percentile
        self.slopes = np.zeros(shape = (self.p, self.bins)) # KLLR slopes
        self.slopes_minus = np.zeros(shape = (self.p, self.bins)) # KLLR slope lower percentile
        self.slopes_plus = np.zeros(shape = (self.p, self.bins)) # KLLR slope upper percentile
        self.scatters = np.zeros(shape = (self.p, self.bins)) # KLLR scatters
        self.scatters_minus = np.zeros(shape = (self.p, self.bins)) # KLLR scatter lower percentile
        self.scatters_plus = np.zeros(shape = (self.p, self.bins)) # KLLR scatter upper uncertainties

        # Binned covariance matrices.
        self.x = np.zeros(shape = (1, self.bins)) # binned scale result from covariance calculation, same as scale
        self.C = np.zeros(shape = (self.p, self.p, self.bins)) # KLLR covariances between properties for each bin
        self.C_minus = np.zeros(shape = (self.p, self.p, self.bins))# KLLR covariance lower percentile
        self.C_plus = np.zeros(shape = (self.p, self.p, self.bins)) # KLLR covariance upper percentile

        # Binned correlation matrix.
        self.x_corr = np.zeros(shape = (1, self.bins)) # binned scale result from correlation calculation, same as scale
        self.corr = np.zeros(shape = (self.p, self.p, self.bins)) # KLLR correlation between properties for each bin
        self.corr_minus = np.zeros(shape = (self.p, self.p, self.bins)) # KLLR correlation lower percentile
        self.corr_plus = np.zeros(shape = (self.p, self.p, self.bins)) # KLLR correlation upper percentile
        
        
    def calculate_scaling_parameters(self, data):
        '''
        Calculate the biined scaling parameters of the properties provided to
        MPQScaling.properties and store the results in the attributes.
        
        --------
        Params
        --------

        data: (pd.DataFrame)
            Data Frame containing all properties of interest.

        '''

        # Check if scale_var is in data columns.
        if self.scale_var not in set(data.columns):
            raise KeyError(f"The scale variable, {self.scale_var}, must be in the columns of the data.")

        # Raise KeyError if the data frame does not include a column that is in MPQScaling.properties.
        if not set(self.properties).issubset(set(data.columns)):
            raise KeyError("Input data frame must include all columns supplied to MPQScaling 'properties' argument.")

        # If xrange is None, then set the bounds to be min and max of x variable.
        if self.xrange is None:
            self.xrange = (np.min(data[f"{self.scale_var}"]),
                           np.max(data[f"{self.scale_var}"]))


        for k, prop in enumerate(self.properties):
            if self.verbose:
                print(f"\n\033[33mLoading KLLR parameters data for property {k}: {prop}...\033[0m")

            # Use KLLR to calculate the binned scaling parameters.
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
        Calculate the binned pariwise property covariance matrix of the
        properties provided to MPQScaling.properties and store results in the
        attributes.

        --------
        Params
        --------

        data: (pd.DataFrame)
            Data frame containing all properties of interest.
            
        '''

        # Check if scale_var is in data columns.
        if self.scale_var not in set(data.columns):
            raise KeyError(f"The scale variable, {self.scale_var}, must be in the columns of the data.")

        # Raise KeyError if the data frame does not include a column that is in MPQScaling.properties.
        if not set(self.properties).issubset(set(data.columns)):
            raise KeyError("Input data frame must include all columns supplied to MPQScaling 'properties' argument.")

        # If xrange is None, then set the bounds to be min and max of x variable.
        if self.xrange is None:
            self.xrange = (np.min(data[f"{self.scale_var}"]),
                           np.max(data[f"{self.scale_var}"]))

        if self.verbose:
            print(f"\n\033[36mCalculating covariances of {self.p} properties...\033[0m")

        pairs = list(it.combinations(range(self.p), 2)) # Define possible pairs of indices in covariance matrix.
        for i, j in pairs:
            prop1, prop2 = self.properties[i], self.properties[j]

            # Use KLLR to calculate the binned pairwise property covariance matrices.
            cov, _ = kl.Plot_Cov_Corr_Matrix(data, xlabel = self.scale_var, ylabels = [prop1, prop2], 
                                             bins = self.bins, xrange = self.xrange, nBootstrap = self.nBootstrap, 
                                             Output_mode = "Covariance", kernel_type = self.kernel_type,
                                             kernel_width = self.kernel_width, percentile = self.percentile,
                                             verbose = self.verbose)
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
        Calculate the binned pariwise property correlation matrix of the
        properties provided to MPQScaling.properties and store results in the
        attributes.

        --------
        Params
        --------

        data: (pd.DataFrame)
            Data frame containing all properties of interest.
            
        '''

        # Check if scale_var is in data columns.
        if self.scale_var not in set(data.columns):
            raise KeyError(f"The scale variable, {self.scale_var}, must be in the columns of the data.")

        # Raise KeyError if the data frame does not include a column that
        # is in MPQScaling.properties.
        if not set(self.properties).issubset(set(data.columns)):
            raise KeyError("Input data frame must include all columns supplied to MPQScaling 'properties' argument.")

        # If xrange is None, then set the bounds to be min and max of x variable.
        if self.xrange is None:
            self.xrange = (np.min(data[f"{self.scale_var}"]),
                           np.max(data[f"{self.scale_var}"]))

        if self.verbose:
            print(f"\n\033[36mCalculating correlations of {self.p} properties...\033[0m")

        pairs = list(it.combinations(range(self.p), 2)) # Define possible pairs of indices in correlation matrix.
        for i, j in pairs:
            prop1, prop2 = self.properties[i], self.properties[j]

            # Use KLLR to calculate the binned pairwise property correlation matrices.
            corr, _ = kl.Plot_Cov_Corr_Matrix(data, xlabel = self.scale_var, ylabels = [prop1, prop2], 
                                              bins = self.bins, xrange = self.xrange, nBootstrap = self.nBootstrap, 
                                              Output_mode = "Correlation", kernel_type = self.kernel_type,
                                              kernel_width = self.kernel_width, percentile = self.percentile,
                                              verbose = self.verbose)
            plt.close()
            
            # Store the correlation matrices in attributes.
            # Nominal correlation values, remembering correlation is symmetric with diagonal of 1s.
            self.corr[i, i, :] = 1
            self.corr[i, j, :] = self.corr[j, i, :] = corr["corr_" + prop1 + "_" + prop2]
            self.corr[j, j, :] = 1

            # Correlation values for the minus percentiles.
            self.corr_minus[i, i, :] = 1
            self.corr_minus[i, j, :] = self.corr_minus[j, i, :] = corr["corr_" + prop1 + "_" + prop2 + "-"]
            self.corr_minus[j, j, :] = 1

            # Correlation values for the plus percentile.
            self.corr_plus[i, i, :] = 1
            self.corr_plus[i, j, :] = self.corr_plus[j, i, :] = corr["corr_" + prop1 + "_" + prop2 + "+"]
            self.corr_plus[j, j, :] = 1
        
        # This is the scale that results from KLLR correlation calculation,
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

        # calculate C^{-1} * slopes where C is of shape (self.p, self.p, self.b) and
        # slopes is of shape (self.p, self.b), result shape: (self.p, b).
        inv_C_slope_prod = np.einsum('ptb,pb->tb', inverse_C, slopes) 
        # calculate slopes * (C^{-1} * slopes), result shape: (b, )
        mass_variance = 1/(np.einsum('pb,pb->b', slopes, inv_C_slope_prod))
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

        The perturbed binned covariance matrix sampled from the error,
        ensured to be symmetric and positive definite.
            
        '''


        # Helper functions for providing a symmetric matrix and checking if
        #  it is positive definite.
        def make_symmetric(matrix):
            # Given a matrix, it makes it symmetric by making the upper triangle
            # equal to the lower triangle. Another method is to average the sum
            # of the matrix and its transpose, but that would alter the distribution of errors.

            # np.triu(slice_2d) will give us the upper triangle with the diagonals included with 0 below the diagonal 
            # np.triu(slice_2d, k = 1) will give us the upper triangle only with diagonal and below the diagonal 0.
            #  Adding these two gives a symmetric matrix.
            symmetric_mat = np.triu(matrix) + np.triu(matrix, k = 1).T

            return symmetric_mat
        
        def is_positive_definite(matrix):
            # Return whether the matrix is positive definite or not
            # by checking if the eigenvalues are all positive.
            e_vals, _ = eigh(matrix) # calculate eigenvalues of matrix
            return (np.min(e_vals) > 0) # check if the minimum eigenvalue is positive
        
        # Sample from covariance matrices in each bin, ensuring that each matrix is symmetric and positive definite.
        # If it is not positive definite, we sample until we find one that is positive definite. Another method 
        # to ensure that a matrix is positive definite is to use the Cholesky decomposition where we first 
        # replace the non-positive eigenvalues with smaller positive eignevalues and use the decomposition
        #  But, after testing, this was very unstable on values of mpq.

        # Initialize the perturbed covariance matrix, that is a sample from the errors of covariance.
        pert_C = np.zeros_like(C)
        for k in range(C.shape[-1]):
            # For each bin, we sample from a random normal distribution with mean 0 (we will add to nominal covariance
            # later) and with errors for the entries given by entries in C_err. After symmetrizing the matrix, we skip
            # over all perturbed covariance matrices that are not positive definite.
            while True:
                # sample from normal distribution for perturbations
                C_perturbations = np.random.normal(np.zeros_like(C[:, :, k]), C_err[:, :, k])
                symm_perturbations  = make_symmetric(C_perturbations) # make perturbations symmetric
                # add perturbation to nominal covariance to find perturbed covariance
                potential_pert_C = C[:, :, k] + symm_perturbations
                if is_positive_definite(potential_pert_C): # skip over all matrices that are not positive definite
                    pert_C[:, :, k] = C[:, :, k] + symm_perturbations
                    break
        

        return pert_C
    

    def _mpq_samples(self, slopes, C, slopes_err, C_err):
        '''
        Sample covariance and slopes from the error given from KLLR and use
        it to calculate the MPQ for each sample with statistical error.
        
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

        '''

        
        # Set seed for MC samples for the errors.
        np.random.seed(self.seed)
        # This empty list will hold the result mpq from the MC.
        mpq_samples = [] # empty list will hold the result mpqs from the MC, shape: (nBootstrap, p, bins)
        # The reason why we use a loop is that _sample_covariance cannot be vectorized since we have the positive
        # definite check on the matrix.
        boot_i = 0
        while boot_i < self.nBootstrap: 
            slopes_perturbations = np.random.normal(np.zeros_like(slopes), slopes_err) # create slope perturbations
            pert_slopes = slopes + slopes_perturbations # add to nominal values of slope to find the perturbed slopes
            pert_C = self._sample_covariance(C, C_err) # sample covariance to find perturbed covariance

            # Check if the matrix is invertible by numpy, pert_C is guaranteed to be positive definite and therfore,
            # invertible but this will serve as a check.
            try:
                inv_C_pert = np.linalg.inv(pert_C.T).T
            except np.linalg.LinAlgError:
                continue

            sample_mpq = self._mpq(pert_slopes, inv_C_pert) # calculate mpq for this sample
            mpq_samples.append(sample_mpq) # add mpq sample to mpq samples list
            boot_i += 1
        
        # Check if the MC yields valid samples.
        mpq_samples = np.array(mpq_samples)
        if mpq_samples.size == 0:
            raise ValueError("All matrix inversions failed. Check the perturbations.")


        return mpq_samples

    @staticmethod
    def _indices_to_props(indices, properties):
        '''
        Convenience function to map a a list indices to associated properties
        If there is more than one index, there properties will
        be linked to each other with an underscore.

        --------
        Params
        -------- 

        indices: (list of ints)
            A list containing the integer indices for the properties that are 
            going to be mapped.
        
        properties: (list of strings)
            All the properties of interest that are passed to MPQScaling.

        --------
        Output
        -------- 

        props: (str)
            A string of properties where different properties are separated by
            underscores.

        '''

        prop_list = []
        for ind in indices:
            prop = properties[int(ind)]
            prop_list.append(prop)

        props = "_".join(prop_list)

        return props


    def _mpq_combo(self, combo_inds):
        '''
        Calculate the MPQ of a combination of properties given by the indices
        of properties of interest in MPQScaling.properties.

        --------
        Params
        --------

        combo_inds: (list or tuple of ints)
            Indices of the corresponding properties in MPQScaling.properties for
            which to calculate the individual or combined MPQ.

        --------
        Output
        --------

        mpq_result: (dict)
            Median, lower, and upper percentile MPQs of the property(ies). If
            MPQScaling.properties = ["M_gas", "M_star", "M_BH", "M_DM"],
            and we pass "012", then we are requesting the MPQ of the combination
            of "M_gas", "M_star" and "M_BH", and the median, lower, and upper
            percentiles of this combined MPQ can be accessed by using
            mpq_result["M_gas_M_star_M_BH_mpq"], mpq_result["M_gas_M_star_M_BH_mpq-"],
            mpq_result["M_gas_M_star_M_BH_mpq+"].

        '''

        # Raise error if there is a repeated index.
        if len(set(combo_inds)) != len(combo_inds):
            raise ValueError(f"There must not be any repeated indices in 'combo_inds'")

        # Raise an error if the maximum index integer andr passed is not between self.p - 1.
        if (max(combo_inds) >= self.p or min(combo_inds) < 0):
            raise ValueError("Any integers in 'combo_inds' must be between 0 " +
                             f"len(MPQScaling.properties) - 1, {self.p - 1}, inclusive.")

        # Initialize dictionary that will hold the result.
        mpq_result = dict()

        # Find the label to attach to the mpq value in the dictionary.
        props_str = self._indices_to_props(combo_inds, self.properties)

        # Pick out a submatrix of the covariance relevant for the properties in this combination for each bin
        C_combo = self.C[combo_inds, :, :][:, combo_inds, :]
        # We don't consider assymetric slopes here, so we average
        avg_C_err = (np.abs(C_combo - self.C_minus[combo_inds, :, :][:, combo_inds, :]) + 
                     np.abs(C_combo - self.C_plus[combo_inds, :, :][:, combo_inds, :]))/2.0

        # pick out a subvector of the slopes relevant for the properties in this combination for each bin
        slopes_combo = self.slopes[combo_inds, :]
        avg_slopes_err = (np.abs(slopes_combo - self.slopes_minus[combo_inds, :]) + 
                          np.abs(slopes_combo - self.slopes_plus[combo_inds, :]))/2.0
        
        # Calculate the mpq including the percentiles, and get the samples for mpq, slope and covariance
        mpq_samples = self._mpq_samples(slopes_combo, C_combo, avg_slopes_err, avg_C_err)

        # Calculate the median, lower percentile, and upper percentile of the mpq using mpq samples.
        mpq_median = np.percentile(mpq_samples, 50, axis = 0) # calculate median mpq from samples
        mpq_minus = np.percentile(mpq_samples, self.percentile[0], axis = 0) # calculate mpq of lower percentile
        mpq_plus = np.percentile(mpq_samples, self.percentile[1], axis = 0) # calculate mpq of upper percentile

        # Save mpq single values and samples to return.
        mpq_result[props_str + "_mpq"] = mpq_median # median mpq
        mpq_result[props_str + "_mpq-"] = mpq_minus # lower percentile of mpq
        mpq_result[props_str + "_mpq+"] = mpq_plus # upper percentile of mpq
        

        return mpq_result
    
    
    def _mpq_all_combos(self, num_props_in_combo):
        '''
        Given a certain number of properties to combine, find the MPQs of any possible
        combination of the MPQScaling.properties with that length. If we have
        a total of 10 properties and we pass num_props_in_combo = 5, we find the MPQs
        of any combination of 5 properties.

        --------
        Params
        --------

        num_props_in_combo: (int)
            The number of properties to combine.

        --------
        Output
        --------

        mpq_result: (dict)
            Median, lower, and upper percentile MPQs of the property(ies). Access
            is similar to MPQScaling._mpq_combo.

        '''
        

        # Raise an error if the maximum index integer andr passed is not between self.p - 1.
        if (num_props_in_combo > self.p or num_props_in_combo <= 0):
            raise ValueError("The maximum integer in 'combo_inds' must be between 1 and " +
                             f"len(MPQScaling.properties), {self.p}, inclusive.")

        # Initialize dictionary that will include the results.
        mpq_results = dict()
        # Possible combinations of indices of length num_props_in_combo.
        combos = it.combinations(range(self.p), num_props_in_combo)
        for combo in combos:
            mpq_res = self._mpq_combo(combo) # Calculate MPQ for this combo
            mpq_results.update(mpq_res) # Add the resulting dictionary to results dictionary

        
        return mpq_results


    def get_scaling_parameters(self):
        '''
        Retrieve KLLR parameters of scale, normalization, slope, scatter (natural
        log units) with errors for each property passed to the MPQScaling.

        --------
        Output
        -------- 

        kllr_params_dict: (dict)
            Dictionary containing the bin number, scale as well as normalization, slope, and
            scatter for each property. For example for the property "M_gas", the 
            normalization, slope, scatter with their lower and upper percentiles
            can be accessed using kllr_params_dict["M_gas_norm"],
            kllr_params_dict["M_gas_norm-"], kllr_params_dict["M_gas_norm+"],
            kllr_params_dict["M_gas_slope"], kllr_params_dict["M_gas_slope-"],
            kllr_params_dict["M_gas_slope+"], kllr_params_dict["M_gas_scatter"],
            kllr_params_dict["M_gas_scatter-"], kllr_params_dict["M_gas_scatter+"],
            respectively. This is designed to be easily convertible to a 
            data frame.
        
        '''
        
        # Create data frame with scaling parameters matched to bin and scale and
        # turn it into dictionary.
        scaling_params_data = np.stack([self.norms.copy(), self.norms_minus.copy(),
                                        self.norms_plus.copy(), self.slopes.copy(),
                                        self.slopes_minus.copy(), self.slopes_plus.copy(),
                                        self.scatters.copy(), self.scatters_minus.copy(),
                                        self.scatters_plus.copy()],
                                       axis = 1).reshape(-1, self.bins)
        
        # Define columns for the scaling parameters.
        scaling_params_columns = []
        for prop in self.properties:
            scaling_params_columns  = scaling_params_columns \
             + [prop + "_norm"] + [prop + "_norm-"] + [prop + "_norm+"] +  [prop + "_slope"] + [prop + "_slope-"] \
             + [prop + "_slope+"] + [prop + "_scatter"] + [prop + "_scatter-"] + [prop + "_scatter+"]
        
        # Define data frame for scaling parameters and then turn it into a dictionary, this will
        # make it easier to turn the result dictionary to a data frame with proper formatting of columns.
        scaling_params_df = pd.DataFrame(data = scaling_params_data.T, columns = scaling_params_columns)
        scaling_params_df.insert(0, "bin", np.arange(self.bins)) # insert bin number
        scaling_params_df.insert(1, f"{self.scale_var}", self.scale) # insert scale variable column

        # Convert data frame to dictionary and return.
        scaling_params_dict = scaling_params_df.to_dict()


        return scaling_params_dict
    
    
    @staticmethod
    def _flatten_lower_triangle_of_binned_matrices(binned_matrices):
        '''
        Convenience method that extracts the covariance/correlation between each pair of
        properties and flattens them to rows where each entry is the correlation
        of a pair of properties at a specific bin. It does this for each pair of 
        properties of interest passed into MPQScaling.

        --------
        Params
        --------    

        binned_matrices: (numpy array)
            Must be of shape (# properties, # properties, # bins) that comes
            from the KLLR caclulations of covariance/correlation.

        --------
        Output
        -------- 

        binned_matrices_flat: (numpy array)
            Shape (# properties choose 2, # of bins). Flattened covariances/
            correlations where the second dimension traverses the bins
            and the first dimension is for each pair of properties.

        combos: (list of tuples)
            The possible pairs of indices for the properties that were used
            to calculate the covariances/correlations in the first dimension
            of binned_matrices_flat in order.

        '''

        binned_matrices_flat = []
        num_props = binned_matrices.shape[0]
        combos = list(it.combinations(range(num_props), 2))

        for combo in combos:
            binned_matrices_flat.append(binned_matrices[combo[0], combo[1], :])

        binned_matrices_flat = np.array(binned_matrices_flat)
            
        return binned_matrices_flat, combos


    def get_covariances(self):
        '''
        Retrieve binned pairwise property covariance.

        --------
        Output
        -------- 

        kllr_params_dict: (dict)
            Dictionary containing the pairwise property covariances. For example,
            if we want to get the covariance between "M_gas" and "M_star" with the
            lower and upper percentiles, we would use cov_dict["M_gas_M_star_cov"],
            cov_dict["M_gas_M_star_cov-"], cov_dict["M_gas_M_star_cov+"], 
            respectively. This is designed to be easily convertible to a 
            data frame.
        
        '''
        
        # Turn covariance matrices to rows to be stored in data frame.
        cov_flat, combos = self._flatten_lower_triangle_of_binned_matrices(self.C.copy())
        cov_minus_flat, _ = self._flatten_lower_triangle_of_binned_matrices(self.C_minus.copy())
        cov_plus_flat, _ = self._flatten_lower_triangle_of_binned_matrices(self.C_plus.copy())

        # Create data frame with pairwise property covariances matched to bin and
        # scale and turn it into dictionary.
        cov_data = np.stack([cov_flat, cov_minus_flat, cov_plus_flat], axis = 1).reshape(-1, self.bins)
        # Create columns for covariance data frame.
        cov_columns = []
        for combo in combos:
            cov_columns = cov_columns + [f"{self.properties[combo[0]]}_" f"{self.properties[combo[1]]}_cov"] + \
                                        [f"{self.properties[combo[0]]}_" f"{self.properties[combo[1]]}_cov-"] + \
                                        [f"{self.properties[combo[0]]}_" f"{self.properties[combo[1]]}_cov+"]

        # Define data frame for scaling parameters and then turn it into a dictionary, this will
        # make it easier to turn the result dictionary to a data frame with proper formatting of columns.
        cov_df = pd.DataFrame(data = cov_data.T, columns = cov_columns)
        cov_df.insert(0, "bin", np.arange(self.bins)) # insert bin number
        cov_df.insert(1, f"{self.scale_var}", self.x) # insert scale variable column

        # Convert data frame to dictionary and return.
        cov_dict = cov_df.to_dict()

        
        return cov_dict

    
    def get_correlations(self):
        '''
        Retrieve binned pairwise property correlations.

        --------
        Output
        -------- 

        kllr_params_dict: (dict)
            Dictionary containing the pairwise property correlations. For example,
            if we want to get the correlation between "M_gas" and "M_star" with the
            lower and upper percentiles, we would use cov_dict["M_gas_M_star_corr"],
            cov_dict["M_gas_M_star_corr-"], cov_dict["M_gas_M_star_corr+"], 
            respectively. This is designed to be easily convertible to a 
            data frame.
        
        '''
        
        # Turn correlation matrices to rows to be stored in data frame.
        corr_flat, combos = self._flatten_lower_triangle_of_binned_matrices(self.corr)
        corr_minus_flat, _ = self._flatten_lower_triangle_of_binned_matrices(self.corr_minus)
        corr_plus_flat, _ = self._flatten_lower_triangle_of_binned_matrices(self.corr_plus)

        # Create data frame with pairwise property correlations matched to bin and
        # scale and turn it into dictionary.
        corr_data = np.stack([corr_flat, corr_minus_flat, corr_plus_flat], axis = 1).reshape(-1, self.bins)
        # Define columns for correlations data frame.
        corr_columns = []
        for combo in combos:
            corr_columns = corr_columns + [f"{self.properties[combo[0]]}_" f"{self.properties[combo[1]]}_corr"] + \
                                          [f"{self.properties[combo[0]]}_" f"{self.properties[combo[1]]}_corr-"] + \
                                          [f"{self.properties[combo[0]]}_" f"{self.properties[combo[1]]}_corr+"]

        # Define data frame for scaling parameters and then turn it into a dictionary, this will
        # make it easier to turn the result dictionary to a data frame with proper formatting of columns.
        corr_df = pd.DataFrame(data = corr_data.T, columns = corr_columns)
        corr_df.insert(0, "bin", np.arange(self.bins)) # insert bin number
        corr_df.insert(1, f"{self.scale_var}", self.x_corr) # insert scale variable column

        # Convert data frame to dictionary and return.
        corr_dict = corr_df.to_dict()        

        
        return corr_dict
    

    def get_mpq(self, num_props_in_combination = None, combination = None):
        '''
        Calculate and retrieve Mass Proxy Quality (MPQ) for either all combinations
        of a certain length by providing num_props_in_combination or a certain
        combination by providing a list of the indices of the properties
        that we want to combine. Only one of num_props_in_combination and
        combination should be provided. It could be a single index if we want a
        single property. For example, if we have properties ["M_gas", "M_star",
        "M_BH", "M_DM"], and we want the MPQ of all pairs of properties, we 
        would pass the argument num_props_in_combination = 2. If we want
        the MPQ of the combination of "M_gas", "M_BH" and "M_DM", we would use
        combination = [0, 1, 2]".

        --------
        Params
        -------- 

        num_props_in_combination: (int)
            Number of properties in the combination for which we want to 
            calculate the MPQ. Supplying this will calculate all possible 
            combination of this length.

        combination: (list or tuple of integers)
            The indices of the properties for which we want to calculate the
            combined MPQ.

        --------
        Output
        --------

        mpq_dict: (dict)
            Dictionary containing the mpqs. The mpqs with lower and upper
            percentiles can be accessed using mpq_dict["M_gas_mpq"],
            mpq_dict["M_gas_mpq-"], mpq_dict["M_gas_mpq+"]. If the MPQ of a 
            combination of properties was calculated, we would access the mpqs
            using mpq_dict["M_gas_M_star_mpq"], mpq_dict["M_gas_M_star_mpq-"],
            mpq_dict["M_gas_M_star_mpq+"]. This is designed to be easily 
            convertible to a data frame.
            
        '''

        # Raise error if both arguments are provided.
        if num_props_in_combination is not None and combination is not None:
            raise ValueError("Only one of 'num_props_in_combination' and 'combination' should be provided, not both.")

        # Calculate MPQs of all combinations.
        if num_props_in_combination is not None:
            mpq_res = self._mpq_all_combos(num_props_in_combination)
        # Calculate MPQ of a single combination.
        elif combination is not None:
            mpq_res = self._mpq_combo(combination)
        # Raise error if no arguments are provided.
        else:
            raise ValueError("One of 'num_props_in_combination' or 'combination' must be provided.")
        
        # Define data frame for scaling parameters and then turn it into a dictionary, this will
        # make it easier to turn the result dictionary to a data frame with proper formatting of columns.
        mpq_df = pd.DataFrame(mpq_res)
        mpq_df.insert(0, "bin", np.arange(self.bins))
        mpq_df.insert(1, f"{self.scale_var}", self.scale)

        # Convert data frame to dictionary.
        mpq_dict = mpq_df.to_dict()
        
        return mpq_dict
            

    def clear_parameters(self):
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