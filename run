#!/bin/bash

set -e # exit when any command fails

wrkdir="."
venvcmd="python3 -m venv env"
activate="env/bin/activate"
install="pip3 install -r requirements.txt"
runcmd="python3 main.py --summary"

log()
{
    echo [$(date)] "$1"
}

while getopts "w:v:a:i:r:" args; do
  case $args in
    w) wrkdir=$OPTARG;;
    v) venvcmd=$OPTARG;;
    a) activate=$OPTARG;;
    i) install=$OPTARG;;
    r) runcmd=$OPTARG;;
  esac
done

log "Switching to working directory '$wrkdir'"
cd $wrkdir

log "Creating virtual environment '$venvcmd'"
$venvcmd

log "Activating virtual environment '$activate'"
source $activate

log "Installing requirements '$install'"
$install

log "Running '$runcmd'"
$runcmd
