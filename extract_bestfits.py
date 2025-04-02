from rand_funcs import *
import pickle

def save_obj(obj, name):
    with open('bestfits/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('bestfits/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    

redshifts = [0.5, 1.0, 2.0, 3.0, 4.0]
file = '../ceph/QLF_proj/output/chi2_2P'
fixed, null = best_fit_params_FIXED(redshifts, file)
varied, _ = best_fit_params_VARIED(redshifts, file, null = null)
save_obj(fixed, 'fixed_2P'), save_obj(varied, 'varied_2P')

file = '../ceph/QLF_proj/output/chi2_L'
fixed, null = best_fit_params_FIXED(redshifts, file)
varied, _ = best_fit_params_VARIED(redshifts, file, null = null)
save_obj(fixed, 'fixed_L'), save_obj(varied, 'varied_L')

file = '../ceph/QLF_proj/output/chi2_2P'
null = get_null_a1(file)
fixed, _ = best_fit_params_FIXED(redshifts, file, null = null)
varied, _ = best_fit_params_VARIED(redshifts, file, null = null)
save_obj(fixed, 'fixed_2P_a1'), save_obj(varied, 'varied_2P_a1')