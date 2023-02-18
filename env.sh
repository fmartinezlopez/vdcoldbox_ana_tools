#!/bin/bash

#------------------------------------------------------------------------------
declare -a missing_pypkg
missing_pypkg=()

function chkpypkg() {
  if python3 -c "import pkgutil; raise SystemExit(1 if pkgutil.find_loader('${1}') is None else 0)" &> /dev/null; then

    if [[ ! -z "$2" ]]; then
        if python3 -c "import ${1}; assert ${1}.__version__=='${2}'" &> /dev/null; then
            echo "${1} (${2}) is installed"
        else
            echo "${1} - wrong version installed - expected ${2}, detected $(python3 -c "import ${1}; print ${1}.__version__")"
            missing_pypkg+=(${1})
        fi
    else
        echo "${1} is installed"
    fi

else
    echo "Error: package '${1}' is not installed"
    missing_pypkg+=(${1})
fi
}
#------------------------------------------------------------------------------

function get_realpath() {

    [[ ! -f "$1" ]] && return 1 # failure : file does not exist.
    [[ -n "$no_symlinks" ]] && local pwdp='pwd -P' || local pwdp='pwd' # do symlinks.
    echo "$( cd "$( echo "${1%/*}" )" 2>/dev/null; $pwdp )"/"${1##*/}" # echo result.
    return 0 # success

}
#------------------------------------------------------------------------------

chkpypkg click
chkpypkg numpy
chkpypkg pandas
chkpypkg matplotlib
chkpypkg seaborn
chkpypkg scipy

(( ${#missing_pypkg[@]} > 0 )) &&  return 1
unset missing_pypkg

# get installation location
SCRIPT=$(get_realpath "$BASH_SOURCE")
INSTALL_LOC=$(dirname "$SCRIPT")

export LANG="en_US.utf8"
export LC_ALL="en_US.utf8"

echo "Installation location :" $INSTALL_LOC

export DF_ROOT="$INSTALL_LOC"
# set environment variables

export PYTHONPATH="$PYTHONPATH:$DF_ROOT/src"
export PATH="$PATH:$DF_ROOT/bin"

