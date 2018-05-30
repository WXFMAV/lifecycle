#!/bin/bash
source ~/.bash_profile
set -x

python item.py

ossput.sh dat/res.txt

head dat/res.txt
