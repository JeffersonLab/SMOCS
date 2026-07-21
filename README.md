# SMOCS - Streaming Monitoring Optimization Control System

## Docker Version:
4.38.0 (181591) ==> https://docs.docker.com/desktop/release-notes/#4380

## Software Requirement

- Python 3.9

## Directory Organization
```
├── env.yaml                          : Conda setup file with package requirements
├── setup.py                          : Python setup file with requirements files
├── README.md                         : Readme documentation
├── utests                            : Folder containing a collection of unit tests
├── deployment_example                 : Folder containing a collection of example (streaming virtual environment/MQTT, etc.)
    ├── kafka_fnal                    : Folder containing FNAL monitoring/diagnostic workflow 
    ├── kafka_uitf                    : Folder containing UITF monitoring/diagnostic workflow 
├── accelerate
    ├── core                          : Folder containing core software (agent, buffer, model, etc.)
    ├── agents                        : Folder containing agents for opt/control, diagnostic, etc.
    ├── data_parcers                  : Folder containing data parcers (streaming, files, etc.)
    ├── data_preps                    : Folder containing data prep (normalization, etc.)
    ├── cfgs                          : Folder containing accelerator configurations
    ├── models                        : Folder containing different models (mlp, lstm, etc.)
```

## Installing

- Clone code from repo and move into directory
```
git clone https://github.com/JeffersonLab/SMOCS.git
cd SMOCS
```

## Citation

If you use this work, please cite:

```bibtex
@misc{kasparian2026smocsstreamingframeworksimplified,
      title={SMOCS: A Streaming Framework for Simplified Deployment, Monitoring, and Optimization of ML Systems in Production}, 
      author={Armen Kasparian and Kishansingh Rajput and Malachi Schram and John Vennekate},
      year={2026},
      eprint={2607.02731},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2607.02731}, 
}
```

## Contacts

If you have any questions or concerns please contact Malachi Schram (schram@jlab.org), Kishan Rajput (kishan@jlab.org), Armen Kasparian (armenk@jlab.org).
