---
category: npm
created: '2022-12-10T13:29:15.916Z'
date: '2023-12-22 23:52:59'
description: This article provides a comprehensive guide on how to set the NODE_PATH
  environment variable for npm global package installation across different operating
  systems. It explains the process in detail for ZSH, Bash, Fish, Windows, Termux,
  Kali, and MacOS, ensuring proper importing of packages from specific paths without
  any issues.
modified: '2022-12-10T13:36:25.418Z'
tags:
- npm
- NODE_PATH
- environment variable
- global package installation
- operating systems
- ZSH
- Bash
- Fish
- Windows
- Termux
- Kali
- MacOS
title: nodejs NODE_PATH for npm global package installation
---

# nodejs NODE_PATH for npm global package installation

when installing global ackages, we do not need to specify NODE_PATH. but it is not configured beforehand thus when you want to import packages from there you will face issue.

for zsh/bash/fish:

```bash
export NODE_PATH=<NODE_PATH>
```

on windows just use the old school drill (open environment editor)

chech the exact path of `NODE_PATH` after invoking `npm install -g <package_name>`, then check if the installed package exists in that path you guessed.

on termux: `/data/data/com.termux/files/usr/lib/node_modules`

on kali: `/usr/local/lib/node_modules` (may be inaccurate)

on macos: `/opt/homebrew/lib/node_modules` (nodejs installed via brew)

