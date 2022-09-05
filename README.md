# OuterAdapter
An implementation for "Domain Adaptation with Dynamic Open-Set Targets" (KDD'22) [[Paper]](https://dl.acm.org/doi/abs/10.1145/3534678.3539235).

## Environment Requirements
The code has been tested under Python 3.6.5. The required packages are as follows:
* numpy==1.18.1
* sklearn==0.22.1
* scikit-image==0.16.2
* torch==1.4.0
* torchvision==0.5.0

## Data Sets
We used the following data sets in our experiments:
* [Office-Home](https://www.hemanthdv.org/officeHomeDataset.html)
* [Office-31](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/)
* [Syn2Real-O](https://ai.bu.edu/visda-2018/)

## Run the Codes
For dynamic open-set domain adaptation on Office-Home/Office-31/Syn2Real-O, please run
```
python main.py
```

## Acknowledgement
This is the latest source code of **OuterAdapter**. If you find that it is helpful for your research, please consider to cite our paper:

```
@inproceedings{wu2022domain,
  title={Domain Adaptation with Dynamic Open-Set Targets},
  author={Wu, Jun and He, Jingrui},
  booktitle={Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={2039--2049},
  year={2022}
}
```
