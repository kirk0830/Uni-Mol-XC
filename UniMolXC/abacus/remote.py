'''
this module is used to submit abacus jobs to a remote server
by abacustest package. For installing the abacustest, please
follow steps:
```bash
# install the abacustest kernel
git clone https://github.com/pxlxingliang/abacus-test.git
cd abacus-test
python3 setup.py install

# install the lebegue kernel
pip install lbg -U

# install the dflow kernel
pip install pydflow

# configure your Bohrium account
export BOHRIUM_ACCOUNT=your_account_name
export BOHRIUM_PASSWORD=your_password
export BOHRIUM_PROJECT_ID=your_project_id
```
'''
# built-in modules
import os
import subprocess
import json

# third-party modules
try:
    import abacustest
except ImportError:
    raise ModuleNotFoundError('abacustest is not installed')

try:
    import lbgcore
except ImportError:
    raise ModuleNotFoundError('lebegue is not installed')

try:
    import dflow
except ImportError:
    raise ModuleNotFoundError('dflow is not installed')

def submit(jobdir,
           jobroot=None,
           command='ulimit -c 0; OMP_NUM_THREADS=1 mpirun -np 32 abacus | tee out.log',
           container_image='registry.dp.tech/dptech/abacus:3.8.4',
           jobgroup_prefix='myjob',
           run_on='ali:c32_m64_cpu',
           auto_download=True,
           outdir='out',
           compress=True,
           wait=False):
    '''
    submit the job via abacustest package
    '''
    supplier, machine = run_on.split(':')
    
    examples = []
    if jobroot is not None:
        print(f'WARNING: the `jobdir` (assigned as {jobdir}) parameter is omitted'
              ' due to the `jobroot` parameter is provided.',
              flush=True)
        examples = [f for f in os.listdir(jobroot)
                    if not os.path.isfile(os.path.join(jobroot, f))]
        print(f'Found {len(examples)} examples in {jobroot}', flush=True)
    else:
        examples = [jobdir]
        print(f'Submitting job in {jobdir}', flush=True)
        root = os.path.dirname(os.path.abspath(jobdir))
        print(f'Redirect the output to {root}', flush=True)
    
    for e in examples:
        print(e, flush=True)
    
    param = {
        "bohrium_group_name": 
            f"abacustest-autosubmit-{jobgroup_prefix}",
        "save_path": f"{outdir}",
        "run_dft": [
            {
                "ifrun": True,
                "command": command,
                "extra_files": [],
                "image": container_image,
                "example": examples,
                "bohrium": {
                    "scass_type": machine,
                    "job_type": "container",
                    "platform": supplier,
                    "on_demand": 1
                }
            }
        ],
        "compress": compress
    }
    fn = f'abacustest-{jobgroup_prefix}.json'
    with open(os.path.join(root, fn), 'w') as f:
        json.dump(param, f, indent=4)
        
    print(f'Parameter file saved to {fn}', flush=True)
    print(f'Submitting job to {supplier}:{machine}', flush=True)
    
    # open a subprocess to run the abacustest command
    # redirect the stdout, stderr to the file 'abacustest.log'
    cmd = ['abacustest', 'submit', '-p', fn]
    if not auto_download:
        cmd += ['--download 0']
    with open(os.path.join(root, 'abacustest.log'), 'w') as f:
        p = subprocess.Popen(cmd, cwd=root, stdout=f, stderr=subprocess.STDOUT)
        if wait:
            p.wait()
            print('Job submitted successfully', flush=True)
        else:
            print('Job submitted in background', flush=True)
    return p.pid if not wait else None