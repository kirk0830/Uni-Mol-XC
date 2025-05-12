'''
this script places utility functions for indexing the abacus
jobdir
'''
import os
import unittest

import numpy as np

from UniMolXC.abacus.inputio import read as read_dftparam
from UniMolXC.abacus.struio import read_stru as read_stru_
from UniMolXC.physics.database import convert_l_unit

class AbacusJob:
    
    def init_as_unfinished(self, jobdir):
        '''initialize the AbacusJob instance as a raw jobdir'''
        # INPUT
        fn_input = os.path.join(jobdir, 'INPUT')
        if not os.path.exists(fn_input):
            raise FileNotFoundError('`INPUT` file not found in the job directory.')
        self.fn_input = fn_input
        self.input = read_dftparam(self.fn_input)
        
        # STRU
        fn_stru = os.path.join(jobdir, self.input.get('stru_file', 'STRU'))
        if not os.path.exists(fn_stru):
            raise FileNotFoundError('`STRU` file not found in the job directory.')
        self.fn_stru = fn_stru
        self.stru = None
        
        # KPT
        fn_kpt = os.path.join(jobdir, self.input.get('kpoint_file', 'KPT'))
        if not os.path.exists(fn_kpt) and\
            self.input.get('kspacing') is None and\
                self.input.get('gamma_only') is None:
            raise FileNotFoundError('`KPT` file not found in the job directory.')
        self.fn_kpt = fn_kpt
        self.kpt = None
        
        # after the checking above, it can be a real ABACUS jobdir
        self.path = jobdir

    def init_as_finished(self, jobdir):
        '''initialize the AbacusJob instance as a finished jobdir'''
        # OUT
        outdir = os.path.join(jobdir, f'OUT.{self.input.get("suffix", "ABACUS")}')
        if not os.path.exists(outdir):
            raise FileNotFoundError('`OUT` file not found in the job directory.')
        self.outdir = outdir
        
        # running log
        flog = os.path.join(self.outdir, 
                            f'running_{self.input.get("calculation", "scf")}.log')
        if not os.path.exists(flog):
            raise FileNotFoundError('`running.log` file not found in the job directory.')
        self.flog = flog
        self.log = None
        
        # STRU.cif
        fn_stru_cif = os.path.join(self.outdir, 'STRU.cif')
        if not os.path.exists(fn_stru_cif):
            raise FileNotFoundError('`STRU.cif` file not found in the job directory.')
        self.fn_stru_cif = fn_stru_cif
        self.stru_cif = None
        
        # istate.info
        fn_istate = os.path.join(self.outdir, 'istate.info')
        if not os.path.exists(fn_istate):
            raise FileNotFoundError('`istate.info` file not found in the job directory.')
        self.fn_istate = fn_istate
        self.istate = None

        # STRU_ION_D
        fn_stru_ion_d = os.path.join(self.outdir, 'STRU_ION_D')
        if self.input.get('calculation', 'scf') not in ['scf', 'nscf'] and\
            not os.path.exists(fn_stru_ion_d):
            raise FileNotFoundError('`STRU_ION_D` file not found in the job directory.')
        self.fn_stru_ion_d = fn_stru_ion_d if os.path.exists(fn_stru_ion_d) else None
        self.stru_ion_d = None

    def __init__(self, jobdir):
        '''initialize the AbacusJob instance'''
        self.stru = None
        self.cell = None
        self.atomic_symbols = None
        self.atomic_positions = None
        
        try:
            self.init_as_unfinished(jobdir)
        except FileNotFoundError:
            raise RuntimeError('`jobdir` is not a valid ABACUS job directory.')
        
        try:
            self.init_as_finished(jobdir)
            self.complete = True
        except FileNotFoundError:
            self.complete = False
            
    def read_stru(self, cache=True):
        '''read the structure from the jobdir'''
        if self.stru is None:
            stru = read_stru_(self.fn_stru)
            if cache:
                self.stru = stru
        return stru

    def get_cell(self, unit='angstrom', reload_=False):
        '''get the cell from the jobdir'''
        if self.cell is not None and not reload_:
            return self.cell

        if self.stru is None:
            raise RuntimeError('the function `read_stru` should '
                               'be called with `cache=True` before calling '
                               'this function.')
        
        self.cell = np.array(self.stru['lat']['vec']) * \
                convert_l_unit(self.stru['lat']['const'], 
                            unit_from='bohr', 
                            unit_to=unit)
        
        return self.cell

    def get_atomic_positions(self, unit='angstrom', reload_=False):
        '''get the atomic positions from the jobdir'''
        if self.atomic_positions is not None and not reload_:
            return self.atomic_positions
        
        if self.stru is None:
            raise RuntimeError('the function `read_stru` should '
                               'be called with `cache=True` before calling '
                               'this function.')
        
        pos = np.array([atom['coord'] 
                        for s in self.stru['species'] 
                        for atom in s['atom']])
        factor = 1 if self.stru['coord_type'].startswith('Cartesian') \
            else self.stru['lat']['const']
        factor *= 1 if 'angstrom' in self.stru['coord_type'] \
            else convert_l_unit(1, 'bohr', 'angstrom')
        pos = pos * factor
        self.atomic_positions = pos.reshape(-1, 3) * \
            convert_l_unit(1, 'angstrom', unit)
        
        return self.atomic_positions

    def get_atomic_symbols(self, reload_=False):
        '''get the atomic symbols from the jobdir'''
        if self.atomic_symbols is not None and not reload_:
            return self.atomic_symbols
        
        if self.stru is None:
            raise RuntimeError('the function `read_stru` should '
                               'be called with `cache=True` before calling '
                               'this function.')
        
        elem = [[s['symbol']] * s['natom'] for s in self.stru['species']]
        elem = [item for sublist in elem for item in sublist]
        self.atomic_symbols = elem
        
        return self.atomic_symbols

class AbacusJobTest(unittest.TestCase):
    
    def setUp(self):
        testfiles = os.path.dirname(__file__)
        testfiles = os.path.dirname(testfiles)
        self.testfiles = os.path.abspath(os.path.join(testfiles, 'testfiles'))
    
    def test_init_as_unfinished(self):
        jobdir = os.path.join(self.testfiles, 'scf-unfinished')
        job = AbacusJob(jobdir)
        self.assertEqual(job.fn_input, os.path.join(jobdir, 'INPUT'))
        self.assertEqual(job.fn_stru, os.path.join(jobdir, 'STRU'))
        self.assertEqual(job.fn_kpt, os.path.join(jobdir, 'KPT'))
        self.assertEqual(job.path, jobdir)
        self.assertFalse(job.complete)
        
    def test_init_as_finished(self):
        jobdir = os.path.join(self.testfiles, 'scf-finished')
        job = AbacusJob(jobdir)
        self.assertEqual(job.outdir, os.path.join(jobdir, 'OUT.ABACUS'))
        self.assertEqual(job.flog, os.path.join(job.outdir, 'running_scf.log'))
        self.assertEqual(job.fn_stru_cif, os.path.join(job.outdir, 'STRU.cif'))
        self.assertEqual(job.fn_istate, os.path.join(job.outdir, 'istate.info'))
        self.assertTrue(job.complete)
        
if __name__ == '__main__':
    unittest.main()