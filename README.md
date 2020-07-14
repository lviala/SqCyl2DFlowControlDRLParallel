# SqCyl2DFlowControlDRLParallel

This repository contains code for training Reinforcement Learning based control aimed at reducing the drag due to vortex shedding in the wake of a rectangular cylinder in 2D.

This is a further improvement on the work published in "Accelerating Deep Reinforcement Learning strategies of Flow Control through a multi-environment approach", Rabault and Kuhnle, Physics of Fluids (2019), preprint accessible at https://arxiv.org/abs/1906.10382, and in "Artificial Neural Networks trained through Deep Reinforcement Learning discover control strategies for active flow control", Rabault et. al., Journal of Fluid Mechanics (2019), preprint accessible at https://arxiv.org/pdf/1808.07664.pdf, code available at https://github.com/jerabaul29/Cylinder2DFlowControlDRL.

If you find this work useful and / or use it in your own research, please cite these works:

```
Rabault, J., Kuhnle, A (2019).
Accelerating Deep Reinforcement Leaning strategies of Flow Control through a
multi-environment approach.
Physics of Fluids.

Rabault, J., Kuchta, M., Jensen, A., Réglade, U., & Cerardi, N. (2019).
Artificial neural networks trained through deep reinforcement learning discover
control strategies for active flow control.
Journal of Fluid Mechanics, 865, 281-302. doi:10.1017/jfm.2019.62
```
## Getting started

The main code is located in **Cylinder2DFlowControlWithRL**. There, the simulation template to be run is in the **simulation_base** folder. If you want to run different simulations, this is where your modified files will have to go (see the section under for more details about user-defined cases).

## Training as an interactive job

The main script for launching trainings is the **script_launch_parallel.sh** script. It takes care of both launching simulation servers, and launching the parallel training. Launching the scripts takes a few minutes, be a bit patient with it.

The recommended method of execution is with the docker container, provided at https://folk.uio.no/jeanra/Informatics/cylinder2dflowcontrol_Parallel_v1.tar (careful, this is several GB in size). This will make sure that all packages are available in the right versions.

Docker explanations are available in the **Docker** folder. See **README_container.md** for a simple, general introduction to docker. See the **Code_Location_use_docker_Fenics_Tensorforce_parallel.md** file for explanations on how to get the docker container, and run the code inside of it. Once you are familiar with how the code works, you should use the **script_launch_parallel.sh** to launch the servers and clients for you automatically, by executing the following command:

```bash
bash script_launch_parallel.sh <session_name> <first_port> <number_parallel_envs>
```

## Training as a batch job

The main script for launching trainings as batch jobs is the **script_launch_parallel_cluster.sh** script. The script assumes a PBS job manager. It takes care of both launching simulation servers in the background, and launching the parallel training. Launching the scripts takes a few minutes, be a bit patient with it. Make sure enough time is given for the servers to initialize, or a socket connection error will be raised and the job will abort.

Make the job is sized correctly. For a mesh of around 10000 elements and a timestep of dt=0.004, these conservative guidelines are a good starting point:
- wall_time = 30 minutes * #_episodes / #_parallel environments
- n_cpus = #_parallel environments + 2

The job submission requires two environment variables **FIRST_PORT** and **NUM_PORT** to be set prior to execution. This can be done through the *-v <env_vars>* argument of the *qsub* command, e.g:

```bash
qsub script_launch_parallel_cluster.sh -v FIRST_PORT=<first_port>,NUM_PORT=<number_parallel_envs>
```
NOTE: make sure there are no spaces between the different environment variables names and values!

## Troubleshooting

If you encounter problems, please:

- look for help in the .md readme files of this repo
- look for help on the github repo of the JFM paper used for serial training
- if this is not enough to get your problem solved, feel free to open an issue and ask for help.

## Main scripts

- **script_launch_parallel.sh**: automatically launch the training as a parallel interactive job (use the -h option to get help).
- **script_launch_parallel_cluster.sh**: automatically launch the training as a parallel batch job.
- **python3 single_runner.py**: evaluate the latest saved policy as a local job.
- **script_single_runner_cluster.sh**: evaluate the latest saved policy as a batch job.

## CFD simulation fenics, and user-defined user cases

For more details about the CFD simulation and how to build your own user-defined cases, please consult the Readme of the JFM code, availalbe at https://github.com/jerabaul29/Cylinder2DFlowControlDRL.
