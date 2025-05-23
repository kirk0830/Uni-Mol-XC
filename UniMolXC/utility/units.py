'''
this script places the common physical constants, and provide the
unit conversion functions
'''

def convert_energy_unit(e, unit_from='eV', unit_to='Ry'):
    '''convert the energy unit'''
    # factor based on eV
    factor = {'eV': 1.0, 
              'Ry': 1.0 / 13.605693122994, 
              'Ha': 1.0 / 27.211386245988,
              'kcal/mol': 1.0 / 0.0433641,
              'kJ/mol': 1.0 / 0.0015936,
              'cm-1': 1.0 / 8065.54429,
              'J': 1.0 / 1.602176634e-19}
    if unit_from not in factor:
        raise ValueError(f'unit_from {unit_from} not supported')
    if unit_to not in factor:
        raise ValueError(f'unit_to {unit_to} not supported')
    return e / factor[unit_from] * factor[unit_to]

def convert_length_unit(l, unit_from='nm', unit_to='bohr'):
    '''convert the length unit'''
    # factor based on nm
    factor = {'nm': 1.0, 
              'bohr': 1.0 / 0.052917721092,
              'a.u': 1.0 / 0.052917721092,
              'angstrom': 1.0 / 0.1,
              'pm': 1.0 / 1e-12}
    if unit_from not in factor:
        raise ValueError(f'unit_from {unit_from} not supported')
    if unit_to not in factor:
        raise ValueError(f'unit_to {unit_to} not supported')
    return l / factor[unit_from] * factor[unit_to]

