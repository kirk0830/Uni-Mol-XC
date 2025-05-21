'''
this scripts places all the functions for extracting information from
the ABACUS dumped running_*.log files
'''
import os
import subprocess
import unittest

from UniMolXC.utility.units import convert_energy_unit

def grep(pat, fn):
    '''
    grep information with specific pattern from the file
    
    Parameters
    ----------
    pat : str
        the pattern to be searched, currently only string
        is supported
    fn : str
        the file name to be searched
    '''
    if not os.path.exists(fn):
        raise FileNotFoundError(f'File {fn} not found.')
    
    result = subprocess.run(['grep', pat, fn], stdout=subprocess.PIPE)
    lines = result.stdout.decode('utf-8').split('\n')
    lines = [line for line in lines if line]
    return lines

def read_energy(fn, term=None, unit='eV') -> list|dict:
    '''
    read the energy from the file
    
    Parameters
    ----------
    fn : str
        the file name to be searched
    term : str or None
        the energy term to be searched, currently the following terms
        are supported:
        'KS', 'Harris', 'Fermi', 'Band', 'One-Electron',
        'Hartree', 'XC', 'Ewald', 'VDW'.
        If None, all the energy terms will be returned.
    unit : str
        the unit of the energy, currently only 'eV' and 'Ry'
        are supported
    
    Returns
    -------
    list or dict
        the energy values in the specified unit
        if term is None, a dictionary with the energy terms as keys
        and the energy values as values will be returned.
        if term is not None, a list of energy values will be returned.
    '''
    acrynm = {'KS': 'E_KohnSham', 'Harris': 'E_Harris', 'Fermi': 'E_Fermi',
              'Band': 'E_band',   'One-Electron': ' E_one_elec',
              'Hartree': 'E_Hartree', 'XC': 'E_xc', 'Ewald': 'E_Ewald',
              'VDW': 'E_vdw'}
    
    if term is not None:
        raw = grep(acrynm[term], fn)
        if not raw:
            raise ValueError(f'Energy term `{term}` not found in {fn}.')
        return [convert_energy_unit(float(l.split()[2]), 'eV', unit) for l in raw if l]
    else:
        return dict([(k, read_energy(fn, term=k, unit=unit)) for k in acrynm])

class TestLogIO(unittest.TestCase):
    def setUp(self):
        testfiles = os.path.dirname(__file__)
        testfiles = os.path.dirname(testfiles)
        self.testfiles = os.path.abspath(os.path.join(testfiles, 'testfiles'))
    
    def test_grep(self):
        # from a finished ABACUS job
        jobdir = os.path.join(self.testfiles, 'scf-finished')
        fn = os.path.join(jobdir, 'OUT.ABACUS', 'running_scf.log')
        raw = grep('E_KohnSham', fn)
        data = [line.split() for line in raw]
        self.assertTrue(all(isinstance(line, list) for line in data))
        self.assertTrue(all(len(line) == 3 for line in data))
        data = [[line[0], float(line[1]), float(line[2])] for line in data]
        self.assertTrue(all(line[0] == 'E_KohnSham' for line in data))

    def test_read_energy(self):
        # from a finished ABACUS job
        jobdir = os.path.join(self.testfiles, 'scf-finished')
        fn = os.path.join(jobdir, 'OUT.ABACUS', 'running_scf.log')
        data = read_energy(fn, term='KS', unit='eV')
        self.assertTrue(isinstance(data, list))
        self.assertTrue(all(isinstance(d, float) for d in data))
        self.assertTrue(len(data) == 7)
        self.assertTrue(all(abs(x - e0) <= 1e-5) 
                        for x, e0 in zip(data, [-216.4230868440,
                                                -216.4578796366,
                                                -216.4603124282,
                                                -216.4603571058,
                                                -216.4603575398,
                                                -216.4603575421,
                                                -216.4603575420]))
        data = read_energy(fn, term='VDW', unit='eV')
        self.assertTrue(isinstance(data, list))
        self.assertTrue(all(isinstance(d, float) for d in data))
        self.assertTrue(len(data) == 1)
        self.assertTrue(abs(data[0] - -2.4379056085) <= 1e-5)

        data = read_energy(fn, term=None, unit='eV')
        self.assertTrue(isinstance(data, dict))
        self.assertTrue(len(data['KS']) == 7)
        self.assertTrue(len(data['Harris']) == 7)
        self.assertTrue(len(data['Fermi']) == 7)
        self.assertTrue(len(data['Band']) == 1)
        self.assertTrue(len(data['One-Electron']) == 1)
        self.assertTrue(len(data['Hartree']) == 1)
        self.assertTrue(len(data['XC']) == 1)
        self.assertTrue(len(data['Ewald']) == 1)
        self.assertTrue(len(data['VDW']) == 1)

if __name__ == '__main__':
    unittest.main()
    