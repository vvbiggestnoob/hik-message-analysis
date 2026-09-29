# 报文解析综合工具 (hik-message-analysis)

基于 PyQt5 的算法报文解析可视化工具，用于解析算法平台（AIOP 等）回流的报文数据，将目标框、掩码、关键点、标签等信息绘制到原图上，并汇总输出结果表格。

## 功能特性

- 支持检测 + ROI 自由编排的报文解析
- 支持 AIOP 人体关键点报文解析
- 支持分割任务（报文中含掩码信息字段时自动判定），解析掩码并叠加显示
- 异步架构：asyncio 异步读取文件 + ThreadPoolExecutor 多线程处理报文 + 异步写出结果
- 解析结果输出：结果图（原图叠加框/掩码/标签）+ result.xls 汇总表
- 图形界面实时显示解析日志

## 文件说明

| 文件/目录 | 说明 |
|---|---|
| `pyQt.py` | PyQt5 图形界面入口程序 |
| `subFunc_JX20251121.py` | 核心解析逻辑（报文解析、画框、掩码叠加、结果导出） |
| `config/setting_JX20251121.json` | 算法配置文件示例（字段映射、标签映射、展示开关等） |
| `command.py` | PyInstaller 打包脚本 |
| `报文解析工具.spec` | PyInstaller 打包配置 |
| `ikun.ico` | 程序图标 |
| `data/` | 示例数据（图片 + 报文 + 掩码 + 解析结果示例） |

## 使用方法

1. 运行 `pyQt.py`（或使用打包好的 exe）
2. 点击「请打开文件夹」选择待解析的数据文件夹（需包含 picture / AIOPData 等子目录）
3. 点击「加载配置文件」选择对应算法的 json 配置文件
4. 点击「开始解析」，解析过程中日志实时显示在界面上
5. 解析结果输出到数据文件夹下的 `01报文解析结果` 目录及 `result.xls`

## 依赖环境

- Python 3.x
- PyQt5
- opencv-python
- numpy
- Pillow
- xlwt（结果表格导出）

## 打包

```bash
python command.py
# 或
pyinstaller ./pyQt.py -F -w -n "报文解析工具" --icon=ikun.ico
```

## 更新日志

- 20260915：分割任务判定改为报文内容特征判定；重构为 asyncio 异步架构
- 20251121：新增支持 AIOP 人体关键点报文解析
