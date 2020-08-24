# MERL Shopping for Action Recognition

The MERL Shopping Dataset is develloped by MERL ([Mitsubishi Electric Research Laboratories](https://www.merl.com/demos/merl-shopping-dataset)) for training and testing of action detection algorithms.

Take a look at the original [README.md](https://github.com/quental96/merl-shopping/blob/master/ORIGINAL.md) here.

<br/>

## Action recogniton vs action detection

>"In computer vision, action recognition refers to the act of classifying an action that is present in a given video and action detection involves locating actions of interest in space and/or time. The process of action recognition and detection often begins with extracting useful features and encoding them to ensure that the features are specific to serve the task of action recognition and detection."

Originally, the MERL dataset is adapted to action detection, which ensues that the labels are saved through start/stop frame indexes of certain actions.  
The aim of this repo is to be able to use the dataset for action recognition also. The ```utils``` folder thus contains:

* ```det2rec.py``` : separate videos into small numpy clips and store label information in a pandas dataset
* ```test.py```verify that class distribution is the same after detection to recognition codes

<br/>

## Example use

**Input**

```bash
python det2rec.py --start 1 --end 106
```

**Output**

```bash
Working directory: .../merl5


Content:
clips
dataframes
labels
utils
flow_clips
ORIGINAL.md
README.md
results
videos


Class distribution: [1711, 1621, 562, 674, 809]

...
```

