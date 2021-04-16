DIR="log/all_results/"
if [ -d "$DIR" ]; then
  ### Take action if $DIR exists ###
  echo "Directory for all results exists. Verify that all main experiments are in the directory log/all_results/main_experiments and that all files of the ablation study of the reward are on the folder log/all_results/main_experiments"
else
  ###  Control will jump here if $DIR does NOT exists ###
  echo "Directory for all results does not exist. Creating folder and downloading files."
  #create folders
  mkdir log
  mkdir log/all_results
  mkdir log/all_results/ablation_reward
  mkdir log/all_results/main_experiments
  #download files
  cd log
  wget https://syncandshare.lrz.de/dl/fi6xxN2LvPusXttqAo9jB96s/ablation_reward_MovingCom.tar.gz
  wget https://syncandshare.lrz.de/dl/fiEP5fmRC4L7F9w4D2NiSoi9/ablation_reward_MovingObstacle.tar.gz
  wget https://syncandshare.lrz.de/dl/fiKWqj2c8RvFFqfor5qUEJXW/main_experiments_MovingCom.tar.gz
  wget https://syncandshare.lrz.de/dl/fiYaWUZ4N8rMTQ996BoBrKqg/main_experiments_SlideObstacle.tar.gz
  wget https://syncandshare.lrz.de/dl/fi5FhSgVfPh4L1mC1ic6zVMH/main_experiments_MovingObstacle.tar.gz
  wget https://syncandshare.lrz.de/dl/fiBZ9PVxMzN4CLFLJ9vtoUkC/main_experiments_DoubleObstacle.tar.gz
  #unpacking folder
  tar -xzvf ablation_reward_MovingCom.tar.gz -C all_results/ablation_reward
  tar -xzvf ablation_reward_MovingObstacle.tar.gz -C all_results/ablation_reward
  tar -xzvf main_experiments_DoubleObstacle.tar.gz -C all_results/main_experiments
  tar -xzvf main_experiments_MovingCom.tar.gz -C all_results/main_experiments
  tar -xzvf main_experiments_MovingObstacle.tar.gz -C all_results/main_experiments
  tar -xzvf main_experiments_SlideObstacle.tar.gz -C all_results/main_experiments
  #return to parent directory
  cd ../
fi
#the plot of the results will be in the same directory that is passed to python as argument

#plots main experiments
python plot.py log/all_results/main_experiments/MovingObstacle FetchMovingObstacle --naming 8
python plot_bar.py log/all_results/main_experiments/MovingObstacle FetchMovingObstacle

python plot.py log/all_results/main_experiments/MovingCom FetchMovingCom --naming 8
python plot_bar.py log/all_results/main_experiments/MovingCom FetchMovingCom

python plot.py log/all_results/main_experiments/SlideObstacle FetchSlideObstacle --naming 8
python plot_bar.py log/all_results/main_experiments/SlideObstacle FetchSlideObstacle

python plot.py log/all_results/main_experiments/DoubleObstacle FetchDoubleObstacle --naming 8
python plot_bar.py log/all_results/main_experiments/DoubleObstacle FetchDoubleObstacle

#plots ablation study of reward
python plot.py log/all_results/ablation_reward/MovingObstacle FetchMovingObstacle --naming 8
python plot_bar.py log/all_results/ablation_reward/MovingObstacle FetchMovingObstacle

python plot.py log/all_results/ablation_reward/MovingCom FetchMovingCom --naming 8
python plot_bar.py log/all_results/ablation_reward/MovingCom FetchMovingCom



