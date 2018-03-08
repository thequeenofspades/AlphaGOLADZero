#!/bin/bash

BASEDIR=`pwd`
java -jar $BASEDIR/build/libs/match-wrapper-*.jar "$(cat test/wrapper-commands.json)"
echo "${?}"

# python3 export_logs.py resultfile.json game.txt player_log