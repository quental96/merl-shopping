import os
import glob
import argparse
from cv2 import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat
print('\n')


def data_info():
  os.chdir(os.path.dirname(os.getcwd()))
  
  print(f'Working directory: {os.getcwd()}')
  print('\n')
  
  print('Content:')
  for element in os.listdir():
      if not element.startswith('.'):
          print(element)
  print('\n')
  
  actions = glob.glob('labels/' + '/*.mat')
  videos = glob.glob('videos/' + '/*.mp4')
  dataframes = glob.glob('dataframes/' + '/*.csv')

  # Better implementation.
  #actions = sorted(glob.glob('labels/' + '/*.mat'))
  #videos = sorted(glob.glob('videos/' + '/*.mp4'))
  
  class_distribution = [0]*5
  for action in actions:
    action = loadmat(action)
    for index, value in enumerate(action['tlabs']):
      class_distribution[index] += len(value[0])
  
  print(f'Class distribution: {class_distribution}')
  print(sum(class_distribution))
  print('\n')
  return videos, actions, dataframes


def main():

  _, _, dataframes = data_info()

  class_distribution = [0]*5
  s=0

  for i in range(106):

    df = pd.read_csv(dataframes[i])
    s+=len(df)
    temp = df['class'].value_counts().reset_index()

    for j in range(len(temp)):
      idx = temp.at[j, 'index']
      class_distribution[idx-1] += temp.at[j, 'class']

  print(f'Class distribution verification: {class_distribution}')
  print(s)

  print('\n')

  print(temp)
  print(type(temp))
  print(temp.at[0, 'index'])

  return


if __name__ == '__main__':
  main()
