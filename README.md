# Reinforcement Learning self-taught

This project starts when I try to create an agent to play in Google Research Football https://github.com/google-research/football

This documentation is made to myself however, fell free to sugest changes or make changes.

I am c# developer not a python expert. The most of the code that I am going to share are not for python developer, are more to understand what is doing the code against the theory of reinforcement.


I started in this order 
* https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/
* https://deepsense.ai/what-is-reinforcement-learning-the-complete-guide/

## Windows
### gym Environment
* Install anaconda enviroment (it is an integrated environment with Phyton and a lot of IDE), the better you can install with clicks. There is a "Free" version https://www.anaconda.com/products/individual
* Open administratora terminal Anaconda Prompt
* pip install gym

## Google Football Windows
* I try to install under windows, however I was unable to build the engine in windows.
* Use a Virtual Machine to run Ubuntu
    1. Install VMPlayer https://www.vmware.com/ca/products/workstation-player/workstation-player-evaluation.html, 
    2. Use this site https://www.osboxes.org/vmware-images/ to download a image for your VMWarePlayer
    3. Install Ubuntu donwload https://www.osboxes.org/ubuntu/#ubuntu-20-10-vmware
* From the google site https://github.com/google-research/football follow the instruccions for Linux
    I have some problems with the module pygame, I modified the script setup.py to use the version pygame==2.0.1

## Experiments
In the folder footballTests there are multiples experiments, this files are incremental. You can compare between files to check what was the changes in order to work




