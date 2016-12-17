#!/bin/sh

DIR="$( dirname $( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd ))"
export PYTHONPATH=$PYTHONPATH:$DIR
echo "Appending to PYTHONPATH "$DIR
