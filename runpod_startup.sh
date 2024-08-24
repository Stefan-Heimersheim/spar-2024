#!/bin/bash

if [ -n "$GITHUB_ACCESS_TOKEN" ]; then
  git config --global credential.helper store
  echo "https://$GITHUB_ACCESS_TOKEN:x-oauth-basic@github.com" > $HOME/.git-credentials
fi

if [ ! -d "$HOME/spar-2024" ]; then
  git clone https://github.com/Stefan-Heimersheim/spar-2024.git $HOME/spar-2024
fi

if [ -f "$HOME/spar-2024/requirements.txt" ]; then
  pip3 install -r $HOME/spar-2024/requirements.txt
fi

# Set PYTHONPATH for the user's environment
export PYTHONPATH="$HOME/spar-2024:${PYTHONPATH}"

exec "$@"
