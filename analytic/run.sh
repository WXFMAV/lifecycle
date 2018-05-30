#!/bin/sh
source ~/.bash_profile
set -x
#ossget.sh res.txt
#ossget.sh eps.txt
datpath="./dat/"
figpath="./fig/"
#tfevent="tf_event/dat_low_dimensions"
tfevent="tf_event/dat_no_sort"

python extract.py $datpath $figpath $tfevent 
#python plot_reward.py $datpath $figpath reward.txt reward_no_sort 
#python plot_critic.py $datpath $figpath critic_loss.txt critic_loss_no_sort
#python plot_qsa.py $datpath $figpath actor_loss.txt qsa_no_sort

ossput.sh $figpath"reward_no_sort.png" $figpath"critic_loss_no_sort.png" $figpath"qsa_no_sort.png"
