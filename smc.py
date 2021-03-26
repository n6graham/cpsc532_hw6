#from primitives import Categorical
from evaluator import evaluate
import torch
import numpy as np
import json
import sys
import torch.distributions as dist
import matplotlib.pyplot as plt
import time



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

    d = dist.Categorical(logits=torch.tensor(log_weights))
    
    for i in range(len(particles)):
        inds.append(d.sample())

    new_particles = [ particles[i] for i in inds ]

    new_weights = torch.logsumexp(torch.tensor(log_weights),0)

    logL = torch.log(torch.tensor(len(log_weights), dtype=float))

    logZ = new_weights - logL

    return logZ, new_particles



def SMC(n_particles, exp):

    particles = []
    weights = []
    logZs = []
    output = lambda x: x

    for i in range(n_particles):

        

        res = evaluate(exp, env=None)('addr_start', output) # a bit confused by this line
        logW = 0.

        particles.append(res) #initialize particles
        weights.append(logW)

    #can't be done after the first step, under the address transform, so this should be fine:
    
    done = False
    smc_cnter = 0


    while not done:

        #print('In SMC step {}, Zs: '.format(smc_cnter), logZs)
        print('In SMC step {}'.format(smc_cnter))


        address = ''

        for i in range(n_particles): #Even though this can be parallelized, we run it serially
            
            if i%2000 ==0:
                print('running on step {}'.format(i))

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

    '''
    
    #for i in range(1,5):
    for i in range(2,3):
        with open('programs/{}.json'.format(i),'r') as f:
            exp = json.load(f)

        #n_particles = None #TODO 
        #n_particles = 40 # for now

        #n_p = [1, 10, 100, 1000, 10000, 100000]


        #logZ, particles = SMC(n_particles, exp)
        #print('logZ: ', logZ)

        #values = torch.stack(particles)

        #print(torch.mean(values))
        #print(torch.var(values))

        #TODO: some presentation of the results

        #plot_results = []
        #Z_results = []

        n_p = [ 1, 10, 100, 1000, 10000, 100000 ]

        

        s=2

        fig = plt.figure(figsize=(3*s,3*s))
        fig.suptitle("Program 2")
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

            values = values*1.0
            Z = torch.exp(torch.tensor(float(logZ)))

            print("\n Marginal probability/evidence for {} particles is: Z =".format(n), Z)
            
            print("the mean is ", torch.mean(values))

            print("the variance is", torch.var(values))

            #Z_results.append(Z)
            #plot_results.append(particles)
            
            axes[str(j)].hist(values.numpy())
            axes[str(j)].set_title( "{} particles".format(n) )
            j=j+1
            
        plt.savefig('../HW6/tex/program{}_hist.png'.format(i))
    
    '''
    
        


    
    #for i in range(1,5):
    for i in range(3,4):
        print("\n\n Program ",str(i))
        with open('programs/{}.json'.format(i),'r') as f:
            exp = json.load(f)
        # n_p = [1, 10, 100, 1000, 10000, 100000]
        n_p = [1,10,100,1000,10000,100000]
        fig = plt.figure(figsize=(10,6))
        grid = plt.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
        axes = {}
        j = 0
        for n in range(2):
            for m in range(3):
                axes[str(j)] = fig.add_subplot(grid[n,m])
                j = j + 1
        j = 0
        for n in n_p:
            print("\n number of particles: ",str(n))
            tic = time.perf_counter()
            logZ, particles = SMC(n, exp)

            #file = open('p3particles_{}.text'.format(n),'w')
            #file.write(particles)
            #file.close()

            

            toc = time.perf_counter()
            print('elapsed time is: ',str(toc-tic))
            values = torch.stack(particles)
            if n == 1:
                if i != 3:
                    values = torch.tensor([float(values[k]) for k in range(len(values))])
            if type(logZ) == int:
                Z = np.exp(float(logZ))
            else:
                Z = np.exp(logZ)

            #file = open('p3Z_{}.text'.format(n),'w')
            #file.write(Z)
            #file.close()
            print("\n \n n = {}, Marginal probability/evidence Z=".format(n), Z)
            if i == 3:
                num_variables = len(values[0])
                expectation = [None]*num_variables
                variance = [None]*num_variables
                values_binned = [None]*num_variables
                exp_string = ''
                var_string = ''
                for k in range(num_variables):
                    variable_vals = [values[j][k] for j in range(n)]
                    variance[k] =  np.var(variable_vals)
                    expectation[k] =  np.mean(variable_vals)
                    variable_bins = np.digitize(variable_vals, range(3))
                    values_binned[k] = [np.sum(variable_bins==bin_val) for bin_val in range(1,4)]
                    if k in [0,1,2,3, 5,6,7,8, 10,11,12,13, 15]:
                        #exp_string = exp_string + str(expectation[k]) + ' & '
                        exp_string = exp_string + f"{expectation[k]:.1f}" + ' & '
                        #var_string = var_string + str(variance[k]) + ' & '
                        var_string = var_string + f"{variance[k]:.1f}" + ' & '
                    elif k in [4,9,14]:
                        exp_string = exp_string + f"{expectation[k]:.1f}" + ' \\\\ '
                        #exp_string = exp_string + str(expectation[k]) + ' \\\\ '
                        var_string = var_string + f"{variance[k]:.1f}" + ' \\\\ '
                        #var_string = var_string + str(variance[k]) + ' \\\\ '
                    elif k ==16:
                        exp_string = exp_string + f"{expectation[k]:.1f}" + '& & &'
                        #exp_string = exp_string + str(expectation[k]) + ' '
                        var_string = var_string + f"{variance[k]:.1f}" + ' & & & '
                        #var_string = var_string + str(variance[k]) + ' \\\\ '
                #file = open('p3exp_{}.txt'.format(n),'w')
                #file.write(expectation)
                #file.close
                #file= open('p3var_{}.txt'.format(n),'w')
                #file.write(variance)
                #file.close
                
                #print("\nthe mean is ", expectation)
                print("\n mean for latex table: ", exp_string)
                #print("\nthe variance is", variance)
                print("\n variance for latex table: ", var_string)

            else:
                float_vals = [float(values[k]) for k in range(len(values))]
                print("\nthe mean is ", np.mean(float_vals))
                print("\nthe variance is", np.var(float_vals))
            if i ==3:
                cax = axes[str(j)].imshow(values_binned,aspect='auto',cmap='Blues')
                cbar = fig.colorbar(cax,ax = axes[str(j)])
                # cbar.ax.set_yticklabels(['0','1','2'])
                axes[str(j)].set_ylabel('Variables')
                axes[str(j)].set_xticks([0,1,2])
                axes[str(j)].set_yticks(ticks=np.arange(0,num_variables,2))
                axes[str(j)].set_yticks(ticks=np.arange(0,num_variables),minor=True)
                axes[str(j)].grid(which='both',axis='y')
            else:
                axes[str(j)].hist(values)
            axes[str(j)].set_title( "{} particles".format(n) )
            j=j+1
            fig.savefig('../HW6/tex/program{}_hist.png'.format(i))


        # plt.show()
        #fig.savefig('../HW6/tex/program{}_hist.png'.format(i))

    