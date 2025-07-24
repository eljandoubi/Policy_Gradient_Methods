# Installation

* Clone the repo onto the machine you are planning to use. Make sure your current working directory is the directory containing this README file.

* I recommend using a Python `venv` or a `conda` environment to prevent dependency issues. You will need to use Python <=3.10.

    * Install the requirements for this assignments.

    ```bash
    pip install -r requirements.txt
    ```





# How to Run This Code

```shell
python main.py --env-name ENV --seed SEED --METHOD
```

ENV: cartpole/pendulum/cheetah

SEED: 1/2/3

METHOD: baseline/no-baseline/ppo

(note: no-baseline is REINFORCE)

e.g. For cartpole

```shell
python main.py --env-name cartpole --seed 1 --ppo
```


For env's plot,

```shell
python plot.py --env-name ENV --seeds SEEDS
```
