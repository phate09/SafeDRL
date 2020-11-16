## Getting Started

These instructions will help with setting up the project

### Prerequisites
Create a virtual environment with conda:
```
conda env create -f environment.yml
conda activate safedrl
```
This will take care of installing all the dependencies needed by python

In addition, download PRISM from the following link: https://github.com/phate09/prism

Ensure you have Gradle installed (https://gradle.org/install/) 

## Running the code
Before running any code, in a new terminal go to the PRISM project folder and run
```
gradle run
``` 
This will enable the communication channel between PRISM and MOSAIC

###Training
Run the ``train_pendulum.py`` inside ``agents/dqn`` to train the agent on the inverted pendulum problem and record the location of the saved agent

###Analysis
Run the ``domain_analysis_sym.py`` inside ``runnables/symbolic/dqn`` changing paths to point to the saved network