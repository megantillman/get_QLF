from run_emcee_1z_1param import *
import sys

redshift = float( sys.argv[1] )
print( 'Running {}'.format( redshift ) )
filename = "output/chain-data_1Param_z="+str(redshift)+".h5py"
    
    
run_emcee(redshift, filename)
