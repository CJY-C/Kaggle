#!/bin/bash
# Set up a link to the API key to root's home.
if [ ! -d "/root/.kaggle" ]; then
  mkdir /root/.kaggle
fi

if [ ! -f "/root/.kaggle/kaggle.json" ]; then
  ln -s /workspaces/Kaggle/kaggle.json /root/.kaggle/kaggle.json
  chmod 600 /root/.kaggle/kaggle.json
fi
