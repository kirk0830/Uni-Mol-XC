import subprocess

try:
    import abacustest
except ImportError:
    raise ImportError('abacustest is not installed')

def submit():
    '''
    submit the job via abacustest package
    '''