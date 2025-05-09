'''
the input and output of ABACUS INPUT file

The ABACUS INPUT has the following format:

```
INPUT_PARAMETERS
calculation scf # comment

# comment

out_chg 1 10
```
'''
import os
import uuid
import unittest

def read(fn):
    '''
    read the input file
    '''
    with open(fn) as f:
        raw = [l.strip() for l in f.readlines()]
    # wash out the comment lines
    raw = [l for l in raw if l and \
        not (l.startswith('#') or l.startswith('INPUT_PARAMETERS'))]
    raw = [l.split('#')[0].strip().split() for l in raw]
    
    pairs = [(l[0], ' '.join(l[1:])) for l in raw]
    return dict(pairs)

def write(param, fn):
    '''
    write the input file
    '''
    with open(fn, 'w') as f:
        f.write('INPUT_PARAMETERS\n')
        for k, v in param.items():
            if isinstance(v, str):
                f.write(f'{k:<30s} {v}\n')
            elif isinstance(v, list):
                f.write(f'{k:<30s} {" ".join(map(str, v))}\n')
            else:
                raise ValueError(f'Unknown type {type(v)} for {k}')
        f.write('\n')
    return fn

class TestInputIO(unittest.TestCase):
    def setUp(self):
        self.fn = os.path.join(os.path.dirname(__file__), 'test_input')
        self.param = {
            'calculation': 'scf',
            'out_chg': '1 10',
            'stru_file': 'STRU',
            'kpoint_file': 'KPT',
            'suffix': str(uuid.uuid4()),
        }
        write(self.param, self.fn)
    
    def test_read(self):
        param = read(self.fn)
        for k, v in self.param.items():
            self.assertEqual(param[k], v)
    
    def test_write(self):
        param = read(self.fn)
        fn = write(param, self.fn)
        self.assertEqual(fn, self.fn)
        param2 = read(fn)
        for k, v in self.param.items():
            self.assertEqual(param2[k], v)
            
    def tearDown(self):
        os.remove(self.fn)
        
if __name__ == '__main__':
    unittest.main()