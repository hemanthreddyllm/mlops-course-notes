#!/usr/bin/env bash
# Commands to run ON THE EC2 INSTANCE (Ubuntu) for Chapter 18. Don't run this on your laptop.
# Copy the blocks one at a time; the runner tokens come from your own GitHub repo page.
set -euo pipefail

# 1 · update the machine (optional in the course, recommended)
sudo apt-get update -y
sudo apt-get upgrade -y

# 2 · install Docker (official convenience script) and let the `ubuntu` user use it without sudo
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker            # start a shell with the new group (or log out and back in)
docker --version

# 3 · register this machine as a GitHub self-hosted runner
#     GitHub repo → Settings → Actions → Runners → New self-hosted runner → Linux x64.
#     Copy the exact commands from that page: they contain the current runner version and a
#     registration token that expires after about an hour. They look like this:
#
#   mkdir actions-runner && cd actions-runner
#   curl -o actions-runner-linux-x64-<VERSION>.tar.gz -L https://github.com/actions/runner/releases/download/v<VERSION>/actions-runner-linux-x64-<VERSION>.tar.gz
#   tar xzf ./actions-runner-linux-x64-<VERSION>.tar.gz
#   ./config.sh --url https://github.com/<you>/<repo> --token <REGISTRATION_TOKEN>
#       runner group → Enter · runner name → anything (the course types "self-hosted")
#       labels → Enter (every runner gets the labels self-hosted, Linux, X64 anyway) · work folder → Enter
#   ./run.sh                      # "Connected to GitHub … Listening for Jobs"

# 4 · better than ./run.sh: install the runner as a service so it survives logout and reboots
#   sudo ./svc.sh install
#   sudo ./svc.sh start
#   sudo ./svc.sh status

# 5 · after the first deployment: check the container
#   docker ps
#   curl -s localhost:8080/
