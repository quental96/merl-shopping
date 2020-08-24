import os
import glob
import argparse
from cv2 import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat
print('\n')


parser = argparse.ArgumentParser(description='Detection to recognition for MERL Shopping.')
parser.add_argument('--start', default=1, type=int, help='start video')
parser.add_argument('--end', default=1, type=int, help='end video')


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

  #actions = sorted(glob.glob('labels/' + '/*.mat'))
  #videos = sorted(glob.glob('videos/' + '/*.mp4'))
  
  class_distribution = [0]*5
  for action in actions:
    action = loadmat(action)
    for index, value in enumerate(action['tlabs']):
      class_distribution[index] += len(value[0])
  
  print(f'Class distribution: {class_distribution}')
  print('\n')
  return videos, actions


def crop_center_square(frame):
  y, x = frame.shape[0:2]
  min_dim = min(y, x)
  start_x = (x // 2) - (min_dim // 2)
  start_y = (y // 2) - (min_dim // 2)
  return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]


def cv2npy(video):
  id_str = os.path.basename(video)
  print(id_str)
  
  cap = cv2.VideoCapture(video)
  print('Stream start.')
  
  h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  t = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  
  clip_mat = []
  flow_clip_mat = []
  
  old_ret, old_frame = cap.read()
  if not old_ret:
          print('Stream end.\n')
          return
  old_frame = cv2.resize(old_frame, (224,224))
  old_gray_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
  old_frame = old_frame[:, :, [2, 1, 0]]
  clip_mat.append(old_frame)
  
  while cap.isOpened():
      
      ret, frame = cap.read()
      # If frame is read correctly ret is True.
      if not ret:
          print('Stream end.\n')
          break    
      
      # Fill in video array.
      frame = crop_center_square(frame)
      frame = cv2.resize(frame, (224,224))
      gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      frame = frame[:, :, [2, 1, 0]]
      clip_mat.append(frame)
      
      # Calculate optical flow.
  
      # ([, numLevels[, pyrScale[, fastPyramids[, 
      # winSize[, numIters[, polyN[, 
      # polySigma[, flags]]]]]]]])
      retval = cv2.FarnebackOpticalFlow_create(5,0.5,False,15,3,5,1.2,0)       
  
      # ([, tau[, lambda[, theta[, 
      # nscales[, warps[, epsilon[, 
      # innnerIterations[, outerIterations[, scaleStep[, 
      # gamma[, medianFiltering[, useInitialFlow]]]]]]]]]]]])
      #retval	=	cv2.optflow.DualTVL1OpticalFlow_create(0.25,0.15,0.3,5,5,0.01,30,10,0.8,0.0,5,False)	
  
      flow = retval.calc(old_gray_frame, gray_frame, None)
      
      # Fill in flow array.
      flow_clip_mat.append(flow)
      
      old_frame = frame
  
  cap.release()
  
  clip_mat = np.array(clip_mat) / 255.0
  flow_clip_mat = np.array(flow_clip_mat) / 255.0 ##########################
  
  print(f'Frame height: {h}')
  print(f'Frame width: {w}')
  print(f'Frame count: {t}')
  print('\n')
  return id_str, clip_mat[1:], flow_clip_mat


def main():
  global args
  args = parser.parse_args()

  videos, actions = data_info()

  start_video = args.start
  end_video = args.end

  for i in range(start_video, end_video+1):

    df = pd.DataFrame(columns=['name', 'class'])

    j = 1
    video = videos[i-1]

    os.mkdir(f'clips/video_{i}')
    os.mkdir(f'flow_clips/video_{i}')

    print(f'video_{i}')

    #id_str = os.path.basename(video) ###

    id_str, clip_mat, flow_clip_mat = cv2npy(video)
    print(id_str)
    id_str = id_str[:-8] # Remove .mp4 extension and text for id.

    idx = actions.index('labels/' + id_str + 'label.mat')

    action = loadmat(actions[idx])

    for index, value in enumerate(action['tlabs']):
      #print(value[0])
      for start, stop in value[0]:
        np.save(f'clips/video_{i}/clip_{j}', clip_mat[start:stop, ...])
        print(clip_mat[start:stop, ...].shape)
        np.save(f'flow_clips/video_{i}/flow_clip_{j}', flow_clip_mat[start:stop, ...])
        print(flow_clip_mat[start:stop, ...].shape)
        df = df.append({'name' : f'clip_{j}' , 'class' : index+1}, ignore_index=True) 
        j+=1

    df.to_csv(f'dataframes/dataframe_{i}.csv', index=False)

  return


if __name__ == '__main__':
  main()
