#!/bin/bash

BASEDIR=`pwd`
java -jar $BASEDIR/build/libs/match-wrapper-*.jar "$(cat test/passbot-wrapper-commands-flipped.json)"
echo "${?}"

# python3 export_logs.py resultfile.json game.txt player_log