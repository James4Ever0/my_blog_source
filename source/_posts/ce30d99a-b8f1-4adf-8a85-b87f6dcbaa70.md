---
category: Android Reverse Engineering
created: 2022-11-03 13:24:48+08:00
date: '2022-11-03 13:24:48'
description: This article provides a comprehensive guide on Android reverse engineering
  tools, specifically focusing on IDA, Ghidra, Frida, GDA, and Flowdroid. It explains
  how to use Frida to attach an existing process and demonstrates its usage with WeChat
  as an example.
modified: 2022-11-03 13:39:26+08:00
tags:
- android
- reverse engineering
- ida
- ghidra
- frida
- gda
- flowdroid
title: 'Mastering Android Reverse Engineering Tools: IDA, Ghidra, Frida, GDA and Flowdroid'
---

# 安卓反编译

ida ghidra frida

[frida extension/helper methods](https://github.com/iGio90/frida-java-ext)

attach existing process

```bash
sudo frida-ps
sudo frida -n WeChat
sudo frida -p [pid]
```

[gda](http://www.gda.wiki:9090/index.php) 交互式Android反编译 支持[数据流追踪](http://www.gda.wiki:9090/dataFlow.php)

[flowdroid](https://github.com/secure-software-engineering/FlowDroid)
