# how to use the matlab and python integration code
before starting with this code you need to make sure that you have already installed matlab-python api
to do so, follow the instruction below:

https://mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

## how to set up `matlab.ipynb` 
in the `matlab api connection` section, change the address to the directory where you put your simulink file

in the ` tcp connection` section, change the tcp ip to your own ip, in linux terminal you can write, `ip a` and under `inet` you can see your ip. (don't select 127.0.0.1 ip)

you need to update simulink send and receive blocks ip with your own ip as well. 