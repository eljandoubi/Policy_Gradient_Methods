# Installation

* Clone the repo onto the machine you are planning to use.
```bash
git clone https://github.com/eljandoubi/Policy_Gradient_Methods.git && cd Policy_Gradient_Methods
```

* I recommend using [`uv`](https://docs.astral.sh/uv/).

    * Build the environment

    ```bash
   make build
    ```

# How to Run This Code

```shell
python src/main.py --env-name ENV --seed SEED --METHOD
```

ENV: cartpole/pendulum/cheetah

METHOD: baseline/no-baseline/ppo

(note: no-baseline is REINFORCE)

e.g. For cartpole

```shell
python src/main.py --env-name cartpole --seed 1 --ppo
```


For env's plot,

```shell
python src/plot.py --env-name ENV --seeds SEEDS
```

# Remove

If you work in Linux/Mac:

* You can delete the the environment and the result folder with:

    ```bash
   make clean
    ```