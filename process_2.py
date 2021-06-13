from datetime import datetime
print('Process 2')
print('start:', datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
for _ in range(100000000):
    pass

print('end:', datetime.now().strftime("%d/%m/%Y %H:%M:%S"))