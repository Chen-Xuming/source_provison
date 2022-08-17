import numpy as np
import math
def compute_queuing_delay(arrival_rate, service_rate, num_server):
    
    lam = float(arrival_rate)
    mu = float(service_rate)
    c = num_server
    r = lam/mu
    rho = r/c
    assert rho < 1
    p0 = 1/(math.pow(r,c) / (float(math.factorial(c))*(1-rho)) + sum([math.pow(r,n)/float(math.factorial(n)) for n in range(0,c)]))
    queuing_delay = (math.pow(r,c) / (float(math.factorial(c)) * float(c) * mu * math.pow(1-rho,2))) * p0
    assert queuing_delay >= 0.
    return queuing_delay 

def compute_queuing_delay_iteratively_new(arrival_rate, service_rate, num_server):
    lam = float(arrival_rate)
    mu = float(service_rate)
    c = num_server
    r = lam/mu
    rho = r/c
    assert rho < 1    
    
    p0c_2 = 1.
    n = 1
    p0c_1 = r/n
    n += 1
    while n <= c:
        p0c_2 += p0c_1            
        p0c_1 *= r/n
        n += 1

    p0 = 1/(p0c_1/(1-rho) + p0c_2)
    wq = p0c_1*p0/c/(mu * (1-rho)**2)
    return wq


arrival_rate = 900
service_rate = 1000
for n in range(1,20):
    num_server = n
    queuing_delay = compute_queuing_delay(arrival_rate, service_rate, num_server)
    queuing_delay_iteratively = compute_queuing_delay_iteratively_new(arrival_rate, service_rate, num_server)
    print('n = ',n)
    print('queuing_delay: ',queuing_delay)
    print('queuing_delay_iteratively:', queuing_delay_iteratively)
    print('different: ', queuing_delay_iteratively - queuing_delay)