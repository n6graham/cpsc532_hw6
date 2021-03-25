import json
import subprocess

#def daphne(args, cwd='../HW5/daphne'):
def daphne(args, cwd='../daphne'):
#    proc = subprocess.run(['/Users/berendz/bin/lein','run'] + args,
    proc = subprocess.run(['lein','run'] + args,
                          capture_output=True, cwd=cwd)
    if(proc.returncode != 0):
        raise Exception(proc.stdout.decode() + proc.stderr.decode())
    return json.loads(proc.stdout)

