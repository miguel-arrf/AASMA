# AASMA - WareBot
Autonomous Agents and Multi-Agent Systems 2021 - 2022


## 😎 | The team:

- Miguel Ferreira (90899)
- Guilherme Pires (102132)
- Gonçalo Falcão (102219)

___

## 📦 | Installation details:

- Make sure to use Python 3.8. We have not tested in other versions.
- Make sure to create a virtual environment with all the packages in the ```requirements.txt``` file.
  - If for some reason an error related with ```SMAC```appear, make sure to execute ```pip install git+https://github.com/oxwhirl/smac.git```.

___ 

## 🚀 | How to run our custom agents:

- To run our collaborative agent, you can choose which heuristics to use (h1, h2, h3, h4 or h5):
  - ``` python collaborativeEnvironment/run_simulation.py --env colab --firstHeuristic h5 --secondHeuristic h2 ```

- To run our custom single agent:
  - ```python customSingle/customAgent.py```

- To run our custom multi agent:
  - ```python customMulti/customAgent.py```

## 🤖 | How to run our RL agents:

- Run RL single agent:
  - ```python singleRL1AgentsMAPPO/src/main.py --config=mappo --env-config=gymma with env_args.time_limit=500 env_args.key="rware:rware-tiny-1ag-v1"```

- Run RL multi agent:
  - ```python multiRL2AgentsMAPPO/src/main.py --config=mappo --env-config=gymma with env_args.time_limit=500 env_args.key="rware:rware-tiny-2ag-v1"```

- Run RL collaborative multi agent:
  - ```python collaborativeRL2AgentsMAA2C/src/main.py --config=maa2c --env-config=gymma with env_args.time_limit=500 env_args.key="rware:rware-tiny-2ag-v1"```
