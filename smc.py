#from primitives import Categorical
from evaluator import evaluate
import torch
import numpy as np
import json
import sys
import torch.distributions as dist
import matplotlib.pyplot as plt




def run_until_observe_or_end(res):
    cont, args, sigma = res
    res = cont(*args)
    while type(res) is tuple:
        if res[2]['type'] == 'observe':
            #print("res here is", res)
            return res
        cont, args, sigma = res
        res = cont(*args)

    res = (res, None, {'done' : True}) #wrap it back up in a tuple, that has "done" in the sigma map
    return res


def resample_particles(particles, log_weights):
    inds = []

    for i in range(len(particles)):
        inds.append(dist.Categorical(logits=torch.tensor(log_weights)).sample())

    new_particles = [ particles[i] for i in inds ]

    new_weights = torch.logsumexp(torch.tensor(log_weights),0)

    logZ = new_weights/len(particles)

    return logZ, new_particles



def SMC(n_particles, exp):

    particles = []
    weights = []
    logZs = []
    output = lambda x: x

    for i in range(n_particles):

        if i%400 ==0:
            print('running on step {}'.format(i))

        res = evaluate(exp, env=None)('addr_start', output) # a bit confused by this line
        logW = 0.

        particles.append(res) #initialize particles
        weights.append(logW)

    #can't be done after the first step, under the address transform, so this should be fine:
    
    done = False
    smc_cnter = 0


    while not done:

        print('In SMC step {}, Zs: '.format(smc_cnter), logZs)

        address = ''

        for i in range(n_particles): #Even though this can be parallelized, we run it serially
            
            res = run_until_observe_or_end(particles[i])
            
            if 'done' in res[2]: #this checks if the calculation is done
                particles[i] = res[0]
                if i == 0: 
                    done = True  # and enforces everything to be the same as the first particle
                    address = ''
                else:
                    if not done:
                        raise RuntimeError('Failed SMC, finished one calculation before the other')
            else:
                address = res[2]['alpha']
                #if True:
                particles[i]=res
                weights[i]= res[2]['logW']
                assert address == particles[0][2]['alpha']
                    #print("\n res is now:", res)
                    #pass #TODO: check particle addresses, and get weights and continuations
                    #else:
                    #    raise RuntimeError('Address check failed')


        if not done:
            #resample and keep track of logZs
            logZn, particles = resample_particles(particles, weights)
            logZs.append(logZn)
        smc_cnter += 1

    logZ = sum(logZs)


    return logZ, particles


if __name__ == '__main__':

    #for i in range(1,5):
    for i in range(4,5):
        with open('programs/{}.json'.format(i),'r') as f:
            exp = json.load(f)

        #n_particles = None #TODO 
        n_particles = 40 # for now

        #n_p = [1, 10, 100, 1000, 10000, 100000]

        n_p = [1, 10, 100, 1000, 10000, 100000]


        logZ, particles = SMC(n_particles, exp)


        print('logZ: ', logZ)

        values = torch.stack(particles)

        #TODO: some presentation of the results

        print(torch.mean(values))
        print(torch.var(values))


        plot_results = []
        Z_results = []

        s=2

        fig = plt.figure(figsize=(3*s,3*s))
        fig.suptitle("Program 4")
        grid = plt.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

        axes = {}

        j = 0
        for n in range(2):
            for m in range(3):
                axes[str(j)] = fig.add_subplot(grid[n,m])
                j = j + 1

        j = 0
        for n in n_p:
            logZ, particles = SMC(n, exp)
            values = torch.stack(particles)

            Z = torch.exp(logZ)

            print("\n Marginal probability/evidence for {} particles is".format(n), Z)
            
            print("the mean is ", torch.mean(values))

            print("the variance is", torch.var(values))

            Z_results.append(Z)
            plot_results.append(particles)
            
            axes[str(j)].hist(values)
            axes[str(j)].set_title( "{} particles".format(n) )
            j=j+1
            
        plt.savefig('../HW6/tex/program4_hist.png')

        

    '''

        vals = []

        f, a = plt.subplots(6, 3, figsize=(20, 35))
        a = a.ravel()
        for i, ax in enumerate(a):
            if i == 17:
                break
            ax.hist(values[i].numpy(), density=True, bins=3)
            ax.set_ylabel('Probability')
            ax.set_xlabel('values')
            ax.set_title('Histogram for program 4 with {} particles'.format(i+1))

        plt.savefig('../HW5/tex/program4_hist.png')

    '''

