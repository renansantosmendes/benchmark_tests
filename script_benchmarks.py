import os
import threading

def execute(script):
    os.system('python {}'.format(script))

benchmarks = ['DTLZ1', 'DTLZ2', 'DTLZ3', 'DTLZ4']
frequencies = ['1','25','50']

threads = []
for benchmark in benchmarks:
    for frenquency in frequencies:
        threads.append(threading.Thread(target=execute, args=('{}_{}.py'.format(benchmark, frenquency),) ))

for thread in threads:
    thread.start()