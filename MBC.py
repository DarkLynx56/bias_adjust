import numpy as np
import xarray as xr
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

pandas2ri.activate()
    
rMBC = importr("MBC")
    
def applyMBCnR(*argv,num_iter=20,qmap_precalc=False,ratio_seq=np.array([True,False])):    
    datList = [arg for arg in argv]
    datArr = np.array_split(datList,3)
    
    obs_h,mod_h,mod_p, var_cnt = datArr[0].T, datArr[1].T, datArr[2].T, len(datArr[0])

    res = rMBC.MBCn(obs_h,mod_h,mod_p,num_iter,qmap_precalc=qmap_precalc,ratio_seq=ratio_seq)
    
    hist,proj = [res[0][:,i] for i in np.arange(var_cnt)],[res[1][:,i] for i in np.arange(var_cnt)]
    
    dat_collect = *hist,*proj
    
    return dat_collect

def pyMBCn(obs_h,mod_h,mod_p,**kwargs):
    time_hist,time_proj = obs_h['time'].values, mod_p['time'].values
    obs_h,mod_h,mod_p = obs_h.drop('time'),mod_h.drop('time'),mod_p.drop('time')

    datSet_list,var_list = [obs_h,mod_h,mod_p],sorted(list(obs_h.keys()))
    
    datArr_list = [dat[var] for dat in datSet_list for var in var_list]

    time_sgn_i= [['time']]*len(datArr_list) 
    time_sgn_o = [['time']]*(len(datArr_list) - len(var_list))

    mod_corr = xr.apply_ufunc(applyMBCnR,*datArr_list,kwargs=kwargs,input_core_dims=time_sgn_i,output_core_dims=time_sgn_o,vectorize=True,dask='parallelized')                                                                  
    
    dset_h, dset_p = [],[]
    for (k,j,var) in zip(np.arange(len(var_list)),np.arange(len(var_list),len(mod_corr)),var_list):
        dset_h.append(xr.Dataset({var:mod_corr[k]}))
        dset_p.append(xr.Dataset({var:mod_corr[j]}))
    
    mod_corr_h, mod_corr_p = xr.merge(dset_h).round(2).assign_coords({'time':time_hist}),xr.merge(dset_p).round(2).assign_coords({'time':time_proj})

    return mod_corr_h,mod_corr_p
