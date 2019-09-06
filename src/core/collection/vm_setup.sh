# Code to setup a fresh Ubuntu VM
# Tested using ubuntu-16.04.3-desktop-amd64.iso

sudo apt-get update

# install some utilities
sudo apt-get install openssh-server
sudo apt-get install screen
sudo apt-get upgrade python3
sudo apt-get install ipython3
sudo apt-get install python3-pip
sudo pip install numpy
sudo pip install pandas
sudo apt-get install vim

# install docker
# https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/
sudo apt-get install \
    linux-image-extra-$(uname -r) \
    linux-image-extra-virtual

sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

sudo apt-get update
apt-cache search docker-ce
sudo apt-get install docker-ce

# download kaggle docker container
sudo docker pull kaggle/python
