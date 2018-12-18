# FLUX - Together we fight fire

Official repository for Project "Deep Learning Approach to automate Fake News Detection using CNN-GRU". This project is initiated with the objective to develop a deep learning model to detect fake news in Malaysia. Different neural net architectures are used, including the hybrid neural net proposed by other literatures. This project aims to modify the existing neural net architecture so that it able to perform better than current benchmark. 


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them.

* Python 3.6 - See this link [How to install Python on Window](https://www.howtogeek.com/197947/how-to-install-python-on-windows/) if you are beginner to Python. 
* To ease the set up, you may download [Anaconda](https://www.anaconda.com/download/). In this case, you are no longer require to install Python 3.6 because anaconda comes together with python sdk. Just select which OS that your laptop are running on and make sure the python version of the anaconda installer is Python 3.6. 

Just to make sure everthing is ok, you may run these commands on your cmd. This only possible once you set your anaconda directory in your system enviroment PATH.

```
conda --version
conda 4.5.11

python --version
Python 3.6.5 :: Anaconda, Inc.
```

### Installing

A step by step series of examples that tell you how to get a development environment running.

1. The first step would be to clone this repo in a folder in your local machine. To do that you need to run following command in command prompt or in git bash.

```
git clone https://github.com/syazanihussin/FakeHope.git
```


## Deployment

Add additional notes about how to deploy this on a live system


## Built With

* [Keras](https://keras.io/) - Python Deep Learning Library used for constructing neural net architecture
* [Flask](http://flask.pocoo.org/) - Python Web Framework used for deploying the final classifier model as Restful API


## Contributing

Please read [CONTRIBUTING.md](https://github.com/syazanihussin/b24679402957c63ec426/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.


## Authors

* **Syazani Hussin** - *Initial work* - [My Repositories](https://github.com/syazanihussin)

See also the list of [contributors](https://github.com/syazanihussin/FakeHope/graphs/contributors) who participated in this project.


## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/syazanihussin/FakeHope/blob/master/LICENSE) file for details


## Acknowledgments

* Z. Zhang, D. Robinson, J. Tepper, “Detecting Hate Speech on Twitter Using a Convolution-GRU Based Deep Neural Network,” ESWC, 2018.
