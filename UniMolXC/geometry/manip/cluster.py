'''
this script is used to implement the geometry manipulation
'''

import unittest
import itertools as it

import numpy as np
from ase.geometry import cellpar_to_cell
from UniMolXC.geometry.visualize import build_ase_atoms, visualize

def _pbcdist_impl(r1, r2, direct, cell):
    '''see func pbcdist'''
    if direct:
        return np.linalg.norm(((r1 - r2 + 0.5) % 1 - 0.5).T, axis=0)
    r1 = np.linalg.solve(cell, r1.T).T
    r2 = np.linalg.solve(cell, r2.T).T
    return np.linalg.norm(cell @ ((r1 - r2 + 0.5) % 1 - 0.5).T, axis=0)  

def pbcdist(r1, r2, direct=False, cell=None):
    '''
    calculate the distance between two points in periodic boundary conditions
    
    Parameters
    ----------
    r1 : np.ndarray
        the first point
    r2 : np.ndarray
        the second point
    direct : bool, optional
        whether the provided coordinates (r1 and r2) are in direct coordinates, 
        by default False
    cell : np.ndarray
        the cell matrix in shape (3, 3)
        
    Returns
    -------
    float
        the distance between the two points. If the direct is True, return
        the distance in direct coordinates, otherwise return the distance in
        cartesian coordinates
    '''
    # sanity check
    if not isinstance(r1, np.ndarray):
        raise TypeError('r1 should be a numpy array')
    if not isinstance(r2, np.ndarray):
        raise TypeError('r2 should be a numpy array')
    if cell is None and not direct:
        raise ValueError('cell should be provided for non-direct coordinates')
    
    # the shape of r1 can be (3,) or (n, 3)
    if r1.ndim == 1:
        r1 = r1.reshape(1, 3)
    # we force r2 to be in shape (3,)
    if r2.ndim != 1:
        raise ValueError('r2 should be in shape (3,)')
    r2 = r2.reshape(1, 3)
    
    return _pbcdist_impl(r1, r2, direct, cell)

def _distmat_impl(pos, direct, cell, supercell):
    '''see func distmat'''
    nat, _ = pos.shape
    out = np.zeros((nat*np.prod(supercell), nat*np.prod(supercell)))
    
    if cell is not None:
        for i in range(nat):
            out[i, :i] = pbcdist(pos[:i], pos[i], direct=direct, cell=cell)
    else:
        for i in range(nat):
            out[i, :i] = np.linalg.norm(pos[:i] - pos[i], axis=1)
    return out + out.T

def distmat(pos, direct=False, cell=None, supercell=None):
    '''
    calculate the distance matrix between all points in r
    
    Parameters
    ----------
    pos : np.ndarray
        the coordinates of the points in shape (n, 3)
    direct : bool, optional
        whether the provided coordinates (r) are in direct coordinates, 
        by default False. If True, the cell should not be None
    cell : np.ndarray
        the cell matrix in shape (3, 3), default is None, which means
        not periodic boundary conditions
    supercell : list, optional
        (not implemented yet!) 
        the supercell in shape (3,), by default None. If None, the supercell
        will be set to [1, 1, 1]
    
    Notes
    -----
    if the `supercell` is not None, the returned distance matrix will taking
    into account the atoms in the supercell. The order of supercell would be:
    (0, 0, 0), (1, 0, 0), ..., (supercell[0]-1, 0, 0), 
    (0, 1, 0), (1, 1, 0), ..., (supercell[0]-1, supercell[1]-1, 0),
    (0, 0, 1), (1, 0, 1), ..., (supercell[0]-1, 0, supercell[2]-1),
    (0, 1, 1), (1, 1, 1), ..., (supercell[0]-1, supercell[1]-1, supercell[2]-1),
        
    Returns
    -------
    np.ndarray
        the distance matrix in shape (n, n)
    '''
    # sanity check
    if not isinstance(pos, np.ndarray):
        raise TypeError('r should be a numpy array')
    _, ndim = pos.shape
    if ndim != 3:
        raise ValueError('r should be in shape (n, 3)')
    if not direct and cell is None:
        raise ValueError('cell should be provided for non-direct coordinates')
    if cell is not None:
        if not isinstance(cell, np.ndarray):
            raise TypeError('cell should be a numpy array')
        if cell.shape != (3, 3):
            raise ValueError('cell should be in shape (3, 3)')
    if supercell is not None:
        raise NotImplementedError('supercell is not implemented yet')
        if not isinstance(supercell, list):
            raise TypeError('supercell should be a list')
        if len(supercell) != 3:
            raise ValueError('supercell should be a list with length 3')
        if not all(isinstance(i, int) for i in supercell):
            raise ValueError('supercell should be a list of integers')
        if not all(i >= 1 for i in supercell):
            raise ValueError('supercell should be a list of positive integers')
    else:
        supercell = [1, 1, 1]
        
    # calculate
    return _distmat_impl(pos, direct, cell, supercell)

def _dist_to_one_plane(a, b, c, plane='ab'):
    '''calculate the distance from a point to a plane'''
    assert plane in ['ab', 'bc', 'ca'], 'plane should be one of ab, bc, ca'
    if plane == 'ab':
        e = np.cross(a, b)
        return np.abs(np.dot(e, c)) / np.linalg.norm(e)
    elif plane == 'bc':
        e = np.cross(b, c)
        return np.abs(np.dot(e, a)) / np.linalg.norm(e)
    else:
        e = np.cross(c, a)
        return np.abs(np.dot(e, b)) / np.linalg.norm(e)

def _nsupercell(cell, rc):
    '''calculate the number of supercells needed to cover the 
    cutoff radius rc in three dimensions'''
    assert cell.shape == (3, 3), 'cell should be in shape (3, 3)'
    assert rc > 0, 'rc should be positive'
    
    return np.ceil([rc/_dist_to_one_plane(cell[(i+1)%3], cell[(i+2)%3], cell[i], p) 
                    for i, p in enumerate(['bc', 'ca', 'ab'])]).astype(int)

def _clustergen_impl(pos, i, rc, direct, cell, elem):
    '''see func clustergen'''
    # raise NotImplementedError('clustergen is not implemented yet')
    # duplicate the coordiantes and cell
    nsupercell = 2 * _nsupercell(cell, rc) - 1

    pos = np.array([[p + np.array([x, y, z]) @ cell.T for p in pos]
        for x, y, z in it.product(*[range(n) for n in nsupercell])]).reshape(-1, 3)
    elem = np.array([elem] * np.prod(nsupercell), dtype=str).flatten() \
        if elem is not None else None
    cell = np.array([c*n for c, n in zip(cell, nsupercell)])
    
    # calculate the distance matrix
    assert direct == False, 'direct coordinates are not supported yet'
    dist = distmat(pos, direct=direct, cell=cell)
    pos = pos[dist[i] <= rc].reshape(-1, 3) - pos[i]
    elem = elem[dist[i] <= rc] if elem is not None else None
    
    # move all atoms' direct coordinates within the range [-0.5, 0.5]
    pos_d = np.linalg.solve(cell, pos.T).T
    return ((pos_d + 0.5) % 1 - 0.5) @ cell.T, cell, elem

def clustergen(pos, direct=False, i=-1, rc=None, cell=None, elem=None):
    '''
    generate the cluster of atoms around the atom i
    
    Parameters
    ----------
    pos : np.ndarray
        the coordinates of the points in shape (n, 3)
    direct : bool, optional
        whether the provided coordinates (r) are in direct coordinates, 
        by default False. If True, the cell should not be None
    i : int
        the index of the atom to be used as the center of the cluster,
        this parameter must be provided.
    rc : float
        the cutoff radius, which means the distance between the center atom
        and the atoms in the cluster should be less than rc. This parameter
        must be provided. If `direct` is True, the cutoff radius should be
        in inteval [0, 1].
    cell : np.ndarray
        the cell matrix in shape (3, 3), default is None, which means
        not periodic boundary conditions. If `direct` is True, the cell
        should be provided.
    elem : list, optional
        the element of the atoms in the cluster, by default None. 
        
    Returns
    -------
    np.ndarray
        the coordinates of the atoms in the cluster in shape (n, 3)
    np.ndarray
        the cell matrix in shape (3, 3)
    '''
    if not isinstance(pos, np.ndarray):
        raise TypeError('pos should be a numpy array')
    if pos.ndim != 2:
        raise ValueError('pos should be in shape (n, 3)')
    if pos.shape[1] != 3:
        raise ValueError('pos should be in shape (n, 3)')
    if not isinstance(i, int):
        raise TypeError('i should be a integer as atomic index')
    if i < 0 or i >= pos.shape[0]:
        raise ValueError('i should be in range [0, %d]' % (pos.shape[0]-1))
    if not isinstance(rc, (float, int)):
        raise TypeError('rc should be a float')
    if rc <= 0:
        raise ValueError('rc should be positive')
    if not direct and cell is None:
        raise ValueError('cell should be provided for non-direct coordinates')
    if cell is not None:
        if not isinstance(cell, np.ndarray):
            raise TypeError('cell should be a numpy array')
        if cell.shape != (3, 3):
            raise ValueError('cell should be in shape (3, 3)')
    if elem is not None:
        if not isinstance(elem, list):
            raise TypeError('elem should be a list')
        if len(elem) != pos.shape[0]:
            raise ValueError('elem should be in shape (n, )')
        if not all(isinstance(e, str) for e in elem):
            raise ValueError('elem should be a list of strings')

    return _clustergen_impl(pos, i, rc, direct, cell, elem) # (n, 3)

class TestCluster(unittest.TestCase):
        
    def test_pbcdist(self):
        dist = pbcdist(np.array([0.1, 0.1, 0.1]), 
                       np.array([0.9, 0.9, 0.9]), 
                       direct=True)[0]
        self.assertAlmostEqual(dist, 0.2*np.sqrt(3), places=5)
        
        dist = pbcdist(np.array([0.1, 0.1, 0.1]),
                       np.array([0.9, 0.9, 0.9]), 
                       direct=False,
                       cell=np.eye(3))[0]
        self.assertAlmostEqual(dist, 0.2*np.sqrt(3), places=5)

        dist = pbcdist(np.array([0.5, 0.5, 0.5]),
                       np.array([0.9, 0.9, 0.9]), 
                       direct=False,
                       cell=np.eye(3))[0]
        self.assertAlmostEqual(dist, 0.4*np.sqrt(3), places=5)

        dist = pbcdist(np.array([0.1, 0.1, 0.1]),
                       np.array([0.5, 0.5, 0.5]), 
                       direct=False,
                       cell=np.eye(3))[0]
        self.assertAlmostEqual(dist, 0.4*np.sqrt(3), places=5)

        dist = pbcdist(np.array([0.1, 0.1, 0.1]),
                       np.array([0.9, 0.9, 0.9]), 
                       direct=False,
                       cell=np.eye(3)*2)[0]
        self.assertAlmostEqual(dist, 0.8*np.sqrt(3), places=5)
        
    def test_distmat(self):
        r = np.array([[0.1, 0.1, 0.1],
                      [0.9, 0.9, 0.9],
                      [0.5, 0.5, 0.5]])
        dist = distmat(r, direct=True, cell=np.eye(3))
        self.assertAlmostEqual(dist[0, 1], 0.2*np.sqrt(3), places=5)
        self.assertAlmostEqual(dist[0, 2], 0.4*np.sqrt(3), places=5)
        self.assertAlmostEqual(dist[1, 2], 0.4*np.sqrt(3), places=5)
        self.assertTrue(np.allclose(dist, dist.T))

    def test_nsupercell(self):
        cell = np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]])
        rc = 1.0
        nsc = _nsupercell(cell, rc)
        self.assertTrue(np.all(nsc == np.array([1, 1, 1])))

        cell = np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]])
        rc = np.sqrt(3)
        nsc = _nsupercell(cell, rc)
        self.assertTrue(np.all(nsc == np.array([2, 2, 2])))

        cell = np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 2]])
        rc = np.sqrt(3)
        nsc = _nsupercell(cell, rc)
        self.assertTrue(np.all(nsc == np.array([2, 2, 1])))
        
        cell = np.array([[1, 1, 0],
                         [1, 0, 1],
                         [0, 1, 1]])
        rc = np.sqrt(3)
        nsc = _nsupercell(cell, rc)
        self.assertTrue(np.all(nsc == np.array([2, 2, 2])))

    def test_clustergen(self):
        
        rc = 10
        i = 0
        cluster, cell, _ = clustergen(
            pos=np.array([[0, 0, 0]]), 
            direct=False, 
            i=i, 
            rc=rc, 
            cell=np.array(cellpar_to_cell([3, 3, 3, 60, 60, 60])))
        # mycluster = build_ase_atoms('Pt', cluster)
        # visualize(mycluster, fn='cluster.xyz', show=False)
        self.assertTrue(isinstance(cluster, np.ndarray))
        self.assertTrue(cluster.ndim == 2)
        nat, nd = cluster.shape
        self.assertTrue(nd == 3)
        self.assertTrue(nat > 0)
        self.assertTrue(all(d <= rc for d in distmat(cluster, cell=cell)[i]))
        
if __name__ == '__main__':
    unittest.main()