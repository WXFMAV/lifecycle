#!/bin/sh
source ~/.bash_profile
datpath=dat/
figpath=fig/
figfile=reward_repeated
tablename=lc_reward_all
bizdate=20180323
expname=avg
parname=param_FPC
datfile1=$tablename.$bizdate.$expname.param_FPC".txt"
datfile2=$tablename.$bizdate.$expname.param_FPC_CNN".txt"
datfile3=$tablename.$bizdate.$expname.param_T_Perm".txt"
datfile4=$tablename.$bizdate.$expname.param_FPC_CNN_EXP".txt"
python plot_reward_many.py $datpath $figpath $figfile $datfile1 $datfile2 $datfile3 $datfile4

exit
datpath=dat/
figpath=fig/
figfile=reward_many
datfile1=param_low_dim1d/reward_avg.txt
datfile2=param_high_dim1d/reward_avg.txt
datfile3=param_cnn_dim2d/reward_avg.txt
datfile4=param_exp_cnn/reward_avg.txt
python plot_reward_many.py $datpath $figpath $figfile $datfile1 $datfile2 $datfile3 $datfile3

figfile=critic_loss_many
datfile1=param_low_dim1d/critic_loss.txt
datfile2=param_high_dim1d/critic_loss.txt
datfile3=param_cnn_dim2d/critic_loss.txt
datfile4=param_exp_cnn/critic_loss.txt
python plot_critic_many.py $datpath $figpath $figfile $datfile1 $datfile2 $datfile3 $datfile3

figfile=qsa_many
datfile1=param_low_dim1d/actor_loss.txt
datfile2=param_high_dim1d/actor_loss.txt
datfile3=param_cnn_dim2d/actor_loss.txt
datfile4=param_exp_cnn/actor_loss.txt
python plot_qsa_many.py $datpath $figpath $figfile $datfile1 $datfile2 $datfile3 $datfile3
