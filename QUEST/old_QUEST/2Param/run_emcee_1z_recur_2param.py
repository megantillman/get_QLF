from run_emcee_1z_2param import *
import sys

redshift = float( sys.argv[1] )
print( 'Running {}'.format( redshift ) )
filename = "output/chain-data_2Param_z="+str(redshift)+".h5py"
    
    
run_emcee(redshift, filename)