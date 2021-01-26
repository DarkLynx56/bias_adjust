from scipy.stats import uniform
from scipy.interpolate import interp1d
import xarray as xr
import numpy as np

def quantileInterpolate(newx, oldx, oldy):
    return interp1d(oldx, oldy, bounds_error=False,
                    kind='nearest', fill_value='extrapolate')(newx)

def censorValues(_dat):
    epsilon, trace = np.finfo(float).eps, 0.05
    trace_clc = 0.5 * trace
    
    _dat = np.where(_dat >= trace_clc, _dat, -1)
    
    N = len(_dat[_dat < 0])
    
    _dat[_dat < 0] = uniform.rvs(size = N, loc = epsilon, scale = trace_clc)
    
    return _dat

def quantileDeltaMapping(obs_h,mod_h,mod_p):
    mod_h_cp = mod_h.copy(deep=True)
    mod_p_cp = mod_p.copy(deep=True)
    
    mod_h = mod_h.dropna(dim='time') 
    mod_p = mod_p.dropna(dim='time')
    
    # For ratio data, treat exact zeros as left censored values less than trace_clc
    obs_h = xr.apply_ufunc(censorValues,obs_h,input_core_dims=[['time']],
                           output_core_dims=[['time']], vectorize=True,
                          dask='parallelized',output_dtypes=[np.float])
    mod_h = xr.apply_ufunc(censorValues,mod_h,input_core_dims=[['time']],
                          output_core_dims=[['time']], vectorize=True,
                          dask='parallelized',output_dtypes=[np.float])
    mod_p = xr.apply_ufunc(censorValues,mod_p,input_core_dims=[['time']],
                          output_core_dims=[['time']], vectorize=True,
                          dask='parallelized',output_dtypes=[np.float])
    
    # Calculate empirical quantiles 
    N = len(mod_p.time)
    tau = np.linspace(start=0,stop=1,num=N)
    
    obs_h_q = obs_h['pr'].quantile(q=tau, dim='time',skipna=True)
    mod_h_q = mod_h['pr'].quantile(q=tau, dim='time',skipna=True)
    mod_p_q = mod_p['pr'].quantile(q=tau, dim='time',skipna=True)
    
    # Reshape tau to match the dimensions of the generated quantile data arrays
    # Pass this as inputs to the two functions quantileInterpolate
    tau_pr = xr.DataArray(tau,dims=['quantile']).broadcast_like(mod_h_q)
    
    # Apply quantile delta mapping bias correction
    tau_mod_p = xr.apply_ufunc(quantileInterpolate,mod_p,mod_p_q,tau_pr,
                           input_core_dims=[['time'],['quantile'],['quantile']],
                           output_core_dims=[['time']],vectorize=True,
                           dask='parallelized',output_dtypes=[np.float])
    
    approx_t_qmc_tmp = xr.apply_ufunc(quantileInterpolate,tau_mod_p,tau_pr,mod_h_q,
                                 input_core_dims=[['time'],['quantile'],['quantile']],
                                 output_core_dims=[['time']], vectorize=True,
                                 dask='parallelized',output_dtypes=[np.float])
    
    delta_m = np.divide(mod_p,approx_t_qmc_tmp)

    corr_inter_mod_p = xr.apply_ufunc(quantileInterpolate,tau_mod_p,tau_pr,obs_h_q,
                           input_core_dims=[['time'],['quantile'],['quantile']],
                           output_core_dims=[['time']],vectorize=True,
                           dask='parallelized',output_dtypes=[np.float])
    
    corr_mod_p = np.multiply(corr_inter_mod_p,delta_m)

    # Restore missing values in corrected projection data
    if len(mod_p.time) < len(obs_h.time):
        corr_mod_p = corr_mod_p.reindex({'time':mod_p_cp.time})

    N = len(mod_h.time)
    tau = np.linspace(start=0,stop=1,num=N)
    
    obs_h_q = obs_h['pr'].quantile(q=tau,dim='time',skipna=True)
    mod_h_q = mod_h['pr'].quantile(q=tau,dim='time',skipna=True)
     
    corr_mod_h = xr.apply_ufunc(quantileInterpolate,mod_h,mod_h_q,obs_h_q,
                               input_core_dims=[['time'],['quantile'],['quantile']],
                               output_core_dims=[['time']],vectorize=True,
                               dask='parallelized',output_dtypes=[np.float])
    
    # Restore missing values in corrected historical data
    if len(mod_h.time) < len(obs_h.time):
        corr_mod_h = corr_mod_h.reindex({'time':mod_h_cp.time})
    
    return corr_mod_h, corr_mod_p
