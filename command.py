import os
os.system('pyinstaller ./pyQt.py -F -w -n "报文解析工具" --icon=ikun.ico')
#pyinstaller ./pyQt.py -F -w -n "报文解析工具" --icon=ikun.ico
# pyinstaller ./pyQt.py -F -w -n "报文解析工具_JX20251121" --icon=ikun.ico --upx-dir=D:\安装包集合\upx-4.0.2-win64
# pyinstaller --path D:\miniforge3\envs\jupyter\Lib\site-packages\PyQt5\Qt5\bin -D ./pyQt.py -n "报文解析工具_JX20250530" --icon=ikun.ico --upx-dir=D:\安装包集合\upx-4.0.2-win64


# 更新日志：
# 20251121更新：新增支持AIOP人体关键点报文解析