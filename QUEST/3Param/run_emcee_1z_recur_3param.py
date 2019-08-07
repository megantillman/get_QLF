from run_emcee_1z_3param import *
import sys

redshift = float( sys.argv[1] )
print( 'Running {}'.format( redshift ) )
filename = "output/chain-data_3Param_z="+str(redshift)+".h5py"
    
    
run_emcee(redshift, filename)
