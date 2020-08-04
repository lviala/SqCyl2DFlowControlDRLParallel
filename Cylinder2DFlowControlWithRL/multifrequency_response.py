import os
import socket
import numpy as np
import csv

from tensorforce.agents import Agent

saver_restore = os.getcwd() + "/saver_data/"

agent = Agent.load(directory = saver_restore)

# If folder does not exist, create it
if(not os.path.exists("frequency_response")):
        os.mkdir("frequency_response")

### System parameters ###

# Vortex shedding cycle
t_vs = 6.860

# Forcing sampling time
t_s = 1/2.92

### Controller harmonic forcing

def one_run(frequency=1, length = 10*t_s, t_s = t_s):
    
    omega = 2*np.pi*frequency

    internals = agent.initial_internals()
    ANN_IO = []
    
    for k in range(length//t_s):
        state = np.sin(omega*k*t_s)
        action, internals = agent.act(state, evaluation=True, internals=internals)
        ANN_IO.append([k*t_s,state['obs'][0],action[0]])

    # Output average values for the single run (Note that values for each timestep are already reported as we execute)
    name = "IO_f_" + str(frequency).replace('.','p') + ".csv"
    
    with open("frequency_response/"+name, "w") as csv_file:
        spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
        spam_writer.writerow(["Time", "Input", "Output"])
        spam_writer.writerows(ANN_IO)

one_run(frequency=0.5)
