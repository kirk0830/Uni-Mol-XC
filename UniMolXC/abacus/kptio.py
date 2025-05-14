'''
read and write the ABACUS KPT file

For function read, will return a dict with the following keys:
'mode' : str
    the mode of the KPT file, can be 'mp', 'gamma-centered-mp',
    and 'line'
'kpoints' : list of list of float
    the kpoints coordinates in the reciprocal space, its value
    will be None when the 'mode' is not 'line'
'nk' : list of int
    the number of kpoints either for sampling in three 
    directions, or the number of interpolation points between
    two kpoints
'kshift' : list of float
    the shift of the Monkhorst-Pack grid, its value will be None
    when the 'mode' is 'line'
'''

def _read_mpmesh(raw):

    temp = [int(x) for x in raw[3].split()]
    assert len(temp) == 6
    return {
        'mode': 'mp' if raw[2] != 'Gamma' else 'gamma-centered-mp',
        'kpoints': None,
        'nk': temp[:3],
        'kshift': [float(x) for x in temp[4:]]
    }
    
def _read_line(raw):

    raise NotImplementedError('the line mode is not implemented yet')

def read(fn):
    '''
    read the KPT file
    
    Parameters
    ----------
    fn : str
        the KPT file name
    
    Returns
    -------
    see annotation at the top of this file
    '''
    with open(fn) as f:
        raw = [l.strip() for l in f.readlines()]
    raw = [l for l in raw if l]
    # empty lines removed
    assert raw[0] == 'K_POINTS'
    if raw[2] == 'Line':
        return _read_line(raw)
    elif raw[2] in ['Gamma', 'MP']:
        return _read_mpmesh(raw)
    else:
        raise NotImplementedError(f'KPT mode {raw[2]} is not implemented yet')
    
def _write_mpmesh(kpt):
    
    out = 'K_POINTS\n0\n'
    out += 'Gamma\n' if kpt['mode'] == 'gamma-centered-mp' else 'MP\n'
    out += ' '.join([str(x) for x in kpt['nk'] + kpt['kshift']]) + '\n'
    return out

def _write_line(kpt):
    
    raise NotImplementedError('the line mode is not implemented yet')

def write(kpt, fn):
    '''
    write the KPT file
    
    Parameters
    ----------
    kpt : dict
        the KPT dict, see annotation at the top of this file
    fn : str
        the KPT file name
    '''
    if kpt['mode'] == 'line':
        out = _write_line(kpt)
    elif kpt['mode'] in ['mp', 'gamma-centered-mp']:
        out = _write_mpmesh(kpt)
    else:
        raise NotImplementedError(f'KPT mode {kpt["mode"]} is not implemented yet')
    with open(fn, 'w') as f:
        f.write(out)
