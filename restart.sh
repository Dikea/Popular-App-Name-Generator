#!/bin/bash
# Notice: please run this script in ROOT mode
pid=`ps -ax | grep server.py | grep -in python | cut -d ' ' -f 1 | cut -d ":" -f 2`
echo $pid
kill -9 $pid
python server.py --batch_size=1 --use_beam_search=True --root_path=model_v0

