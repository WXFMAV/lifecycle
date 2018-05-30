#!/bin/sh
source ~/.bash_profile

a=`ps -ef | grep 'lifecycle/main.py' | awk '{print $2}'`
echo $a
kill $a
