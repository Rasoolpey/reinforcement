# matlab-python integration Reinforcement learning
This code allows you to run simulink model as an environment and send and receive data to python synchronously to train your reinforcement model more reliably. 

## how to use the matlab-python integration code
before starting with this code you need to make sure that you have already installed matlab-python api
to do so, follow the instruction below:

https://mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

## how to set up `matlab.ipynb` 
in the `matlab api connection` section, change the address to the directory where you put your simulink file

in the ` tcp connection` section, the defult ip address (127.0.0.1) has been selected, which supposedly work on any machine but in case it doesn't. change the `TCP_IP` address in this section to your own ip.

 in linux terminal you can write, `ip a` and under `inet` you can find your ip. Plus, you need to update simulink send and receive blocks ip to your own ip as well. 