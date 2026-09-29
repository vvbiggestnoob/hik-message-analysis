import sys
import time
import threading
import traceback
from subFunc_JX20251121 import *
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import QFileSystemWatcher, QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QPlainTextEdit, QComboBox


print("开始展示QT界面！")


class MyWindow(QMainWindow):
    # 解析完成信号（工作线程通过它安全地通知GUI线程恢复按钮状态）
    parse_finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.parsing = False
        self.parse_finished.connect(self.on_parse_finished)

        # 创建窗口和菜单栏
        self.initUI()

    def initUI(self):
        # 设置窗口标题
        self.setWindowTitle('算法报文解析工具  by: zhangle16@hikvision.com.cn')
        # 添加按钮
        self.button1 = QPushButton('请打开文件夹', self)
        self.button1.setFixedSize(190, 50)
        self.button1.setStyleSheet("""
                            background-color: #E1E0E7;
                            color: black;
                            border-radius: 5px;
                            border: 2px solid green;
                            font-size: 20px;
                            font-family: Times New Roman;
                        """)
        self.button1.move(1110, 50)
        self.button1.clicked.connect(self.button1Clicked)

        self.button2 = QPushButton('加载配置文件', self)
        self.button2.setFixedSize(190, 50)
        self.button2.setStyleSheet("""
                            background-color: #E1E0E7;
                            color: black;
                            border-radius: 5px;
                            border: 2px solid green;
                            font-size: 20px;
                            font-family: Times New Roman;
                        """)
        self.button2.move(1110, 150)
        self.button2.clicked.connect(self.button2Clicked)

        self.button3 = QPushButton('开始解析', self)
        self.button3.setFixedSize(150, 50)
        self.button3.setStyleSheet("""
                            background-color: orange;
                            color: white;
                            border-radius: 5px;
                            border: 2px solid green;
                            font-size: 20px;
                            font-family: 黑体;
                        """)
        self.button3.move(287, 250)
        self.button3.clicked.connect(self.button3Clicked)

        self.button4 = QPushButton('关闭程序', self)
        self.button4.setFixedSize(150, 50)
        self.button4.setStyleSheet("""
                            background-color: red;
                            color: white;
                            border-radius: 5px;
                            border: 2px solid green;
                            font-size: 20px;
                            font-family: 黑体;
                        """)
        self.button4.move(963, 250)
        self.button4.clicked.connect(self.button4Clicked)

        # 创建标签控件
        self.label1 = QLabel('请选择需要解析的文件！', self)
        font = QFont('Arial', 10)
        self.label1.setFont(font)
        self.label1.setStyleSheet("border-style: solid; "
                                  "border-width: 2px; "
                                  "border-color: red; "
                                  "border-radius: 5px; "
                                  "color: green;")
        self.label1.setFixedSize(1000, 50)
        self.label1.move(100, 50)

        self.label2 = QLabel('请选择对应算法的配置文件！', self)
        font = QFont('Arial', 10)
        self.label2.setFont(font)
        self.label2.setStyleSheet("border-style: solid; "
                                  "border-width: 2px; "
                                  "border-color: red; "
                                  "border-radius: 5px; "
                                  "color: green;")
        self.label2.setFixedSize(1000, 50)
        self.label2.move(100, 150)

        # 创建一个QPlainTextEdit控件
        self.text_edit = QPlainTextEdit(self)
        self.text_edit.setFixedSize(1200, 500)
        self.text_edit.move(100, 350)

        # 监视log文件的变化
        self.watcher = QFileSystemWatcher(self)
        self.watcher.addPath('log.txt')
        self.watcher.fileChanged.connect(self.check_file_changes)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_file_changes)
        self.timer.start(5000)  # 设置定时器间隔为1秒

        # 设置窗口大小
        self.setGeometry(250, 100, 1400, 900)

        # 设置图标，同级目录下
        self.setWindowIcon(QIcon("ikun.ico"))

    def button1Clicked(self):
        # 打开文件夹对话框
        global folder_path
        if os.path.exists("./setting.json"):
            config_data = read_json_file("./setting.json")
            default_add = config_data["上次打开路径"]
        else:
            default_add = "./"
        folder_path = QFileDialog.getExistingDirectory(self, '选择文件夹', default_add)
        # 更新标签文本
        self.label1.setText(folder_path)

    def button2Clicked(self):
        # 打开文件夹对话框
        global config_json
        config_json, _ = QFileDialog.getOpenFileName(self, '输入配置文件', "./setting.json", 'Config Files (*.json)')
        # 更新标签文本
        self.label2.setText(config_json)

    def button3Clicked(self):
        # 当<开始解析>按钮被单击时执行的操作
        if self.parsing:
            logging.info("正在解析中，请勿重复点击！！！")
            return
        if os.path.isdir(folder_path):
            self.parsing = True
            self.button3.setEnabled(False)
            self.button3.setText('解析中...')
            # 解析放到后台线程执行，避免阻塞GUI事件循环导致界面卡顿
            thread = threading.Thread(target=self._run_parsing, daemon=True)
            thread.start()
        else:
            logging.info("请选择文件夹和配置文件后解析！！！")

    def _run_parsing(self):
        # 后台线程：执行报文解析（内部为asyncio异步流水线）
        try:
            message_Parsing(file_dir=folder_path, config_dir=config_json)
        except Exception as e:
            logging.error(f"解析过程异常：{e}\n{traceback.format_exc()}")
        finally:
            self.parse_finished.emit()

    def on_parse_finished(self):
        # GUI线程：恢复按钮状态
        self.parsing = False
        self.button3.setEnabled(True)
        self.button3.setText('开始解析')

    def button4Clicked(self):
        # 当<关闭程序>被单击时执行的操作
        if os.path.exists("./setting.json"):
            config_data = read_json_file("./setting.json")
            config_data["上次打开路径"] = folder_path
        if os.path.exists("./setting.json"):
            with open("./setting.json", 'w', encoding='utf-8') as file:
                json.dump(config_data, file, ensure_ascii=False, indent=4)
        self.close_window()

    def check_file_changes(self):
        if self.watcher.files():
            path = self.watcher.files()[0]
            # 读取log文件的内容，并更新QPlainTextEdit控件中的文本内容
            with open("log.txt", "r", encoding='utf-8') as f:
                text = f.read()
            self.text_edit.setPlainText(text)

    def close_window(self):
        # 关闭窗口并退出应用程序
        self.close()


if __name__ == '__main__':

    # 清空log文件
    with open('log.txt', 'w', encoding='utf-8') as f:
        f.write('')
        f.close()

    global folder_path
    global config_json
    folder_path = "上层目录初始化"
    config_json = "事件目录初始化"

    try:
        # 创建应用程序对象
        app = QApplication(sys.argv)
        # 创建窗口对象
        window = MyWindow()
        # 显示窗口
        window.show()
        # 进入事件循环
        sys.exit(app.exec_())

    except Exception as e:
        print(traceback.format_exc())
        time.sleep(10)
