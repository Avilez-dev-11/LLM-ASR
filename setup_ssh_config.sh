#!/bin/bash

# Prompt user for their NLUG account
echo "Enter your NLUG (CCI) account username:"
read NLUG_ACCOUNT

# Check if the .ssh directory exists, if not, create it
if [ ! -d "$HOME/.ssh" ]; then
    echo "Creating .ssh directory..."
    mkdir -p "$HOME/.ssh"
fi

# Check if the config file exists, if not, create it
CONFIG_FILE="$HOME/.ssh/config"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Creating .ssh/config file..."
    touch "$CONFIG_FILE"
fi

# Append the SSH configuration to the config file
echo "Writing configuration to .ssh/config file..."

cat <<EOL >> "$CONFIG_FILE"
Host cci01
    HostName blp01.ccni.rpi.edu
    User $NLUG_ACCOUNT
    ControlMaster auto
    ControlPath ~/.ssh/control:%h:%p:%r
    ControlPersist yes

Host cci02
    HostName blp02.ccni.rpi.edu
    User $NLUG_ACCOUNT
    ControlMaster auto
    ControlPath ~/.ssh/control:%h:%p:%r
    ControlPersist yes

Host cci03
    HostName blp03.ccni.rpi.edu
    User $NLUG_ACCOUNT
    ControlMaster auto
    ControlPath ~/.ssh/control:%h:%p:%r
    ControlPersist yes

Host cci04
    HostName blp04.ccni.rpi.edu
    User $NLUG_ACCOUNT
    ControlMaster auto
    ControlPath ~/.ssh/control:%h:%p:%r
    ControlPersist yes

Host fen01
    HostName dcsfen01.ccni.rpi.edu
    User $NLUG_ACCOUNT
    ControlMaster auto
    ControlPersist yes
    ControlPath ~/.ssh/control:%h:%p:%r
    ProxyCommand ssh -q -W %h:%p cci01

Host fen02
    HostName dcsfen02.ccni.rpi.edu
    User $NLUG_ACCOUNT
    ControlMaster auto
    ControlPersist yes
    ControlPath ~/.ssh/control:%h:%p:%r
    ProxyCommand ssh -q -W %h:%p cci02

Host fen03
    HostName dcsfen03.ccni.rpi.edu
    User $NLUG_ACCOUNT
    ControlMaster auto
    ControlPersist yes
    ControlPath ~/.ssh/control:%h:%p:%r
    ProxyCommand ssh -q -W %h:%p cci03

Host fen04
    HostName dcsfen04.ccni.rpi.edu
    User $NLUG_ACCOUNT
    ControlMaster auto
    ControlPersist yes
    ControlPath ~/.ssh/control:%h:%p:%r
    ProxyCommand ssh -q -W %h:%p cci04

Host npl01
    HostName nplfen01.ccni.rpi.edu
    User $NLUG_ACCOUNT
    ControlMaster auto
    ControlPersist yes
    ControlPath ~/.ssh/control:%h:%p:%r
    ProxyCommand ssh -q -W %h:%p cci01

Host drp01
    HostName drpfen01.ccni.rpi.edu
    User $NLUG_ACCOUNT
    ControlMaster auto
    ControlPersist yes
    ControlPath ~/.ssh/control:%h:%p:%r
    ProxyCommand ssh -q -W %h:%p cci01

Host drp02
    HostName drpfen02.ccni.rpi.edu
    User $NLUG_ACCOUNT
    ControlMaster auto
    ControlPersist yes
    ControlPath ~/.ssh/control:%h:%p:%r
    ProxyCommand ssh -q -W %h:%p cci02

Host erp01
    HostName erpfen01.ccni.rpi.edu
    User $NLUG_ACCOUNT
    ControlMaster auto
    ControlPersist yes
    ControlPath ~/.ssh/control:%h:%p:%r
    ProxyCommand ssh -q -W %h:%p cci01

Host erp02
    HostName erpfen02.ccni.rpi.edu
    User $NLUG_ACCOUNT
    ControlMaster auto
    ControlPersist yes
    ControlPath ~/.ssh/control:%h:%p:%r
    ProxyCommand ssh -q -W %h:%p cci02
EOL

# Make sure permissions are set correctly for .ssh directory and config file
chmod 700 "$HOME/.ssh"
chmod 600 "$CONFIG_FILE"

echo "SSH configuration has been successfully set up!"