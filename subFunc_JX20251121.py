"""
新增支持自由编排的报文解析：检测+ROI
新增支持AIOP关键点报文解析
20260915更新：
1. 分割任务判定改为报文内容特征判定（报文中存在掩码信息字段），不再只看masks/Images/Annotations三个文件夹
2. 重构为异步架构：asyncio异步读取文件 + ThreadPoolExecutor多线程处理报文 + asyncio异步写出结果
"""
import lzma
import os
import cv2
import json
import math
import asyncio
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageFile

# 遇到坏图片就跳过去
ImageFile.LOAD_TRUNCATED_IMAGES = True
# log的格式设定
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
time_format = "%Y-%m-%d %H:%M:%S"

# 先创建FileHandler，指定编码
file_handler = logging.FileHandler("log.txt", encoding='utf-8')
file_handler.setFormatter(logging.Formatter(log_format, datefmt=time_format))

# 配置logger
logging.basicConfig(level=logging.INFO, handlers=[file_handler])


# 画框写label函数1，color为框的颜色，默认为红色；如果绿色无效则需要手动修改，labels支持列表格式
def drawOneBox(img, x1, y1, x2, y2, labels, width, height, color=(0, 0, 255), alpha=256, font_size=28, keyPointInfo=False):
    # 目标框绘制
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    # 将BGR转换为RGB
    img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个新的RGBA图像用于绘制透明效果
    overlay = Image.new('RGBA', img_PIL.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    # 初始化参数
    text_width_lst = [0]
    text_height_lst = [0]
    # 加载字体
    font = ImageFont.truetype("C:/Windows/Fonts/simkai.ttf", font_size, encoding='utf-8')
    # 获取中英文最大的高度
    label_test = "中文_English_123"
    bbox_test = draw.textbbox((0, 0), label_test, font=font)
    text_height_test = math.ceil(bbox_test[3] - bbox_test[1])
    for label in labels:
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = math.ceil(bbox[2] - bbox[0])
        text_width_lst.append(text_width)
        text_height_lst.append(text_height_test)
    text_width_max = max(text_width_lst)
    text_height_sum = sum(text_height_lst)
    # 1.位置判断
    x = min(x1, int(width - text_width_max))
    y = min(y1, int(height - text_height_sum))
    position1 = [x, y]
    # 绘制半透明矩形 (RGBA格式，alpha=128表示50%透明度)
    draw.rectangle(
        (position1[0], position1[1], position1[0] + text_width_max, position1[1] + text_height_sum),
        fill=(color[2], color[1], color[0], alpha)  # 注意颜色顺序是RGB
    )
    # 绘制文本
    for i, label in enumerate(labels):
        position1[1] = position1[1] + text_height_lst[i]
        position2 = (position1[0], position1[1])
        draw.text(position2, label, fill=(0, 0, 0, 255), font=font)  # 黑色不透明文本
    # 将透明层与原图合并
    img_PIL = Image.alpha_composite(img_PIL.convert('RGBA'), overlay)
    # 格式转换回cv2 (BGR)
    img = cv2.cvtColor(np.asarray(img_PIL.convert('RGB')), cv2.COLOR_RGB2BGR)
    # 绘制关键点
    palette = np.array([[0, 255, 0], [255, 128, 0], [255, 51, 255], [51, 153, 255]])
    # 关键点基线，修改过协议
    skeleton = [[0, 1], [0, 2], [3, 1], [3, 2], [3, 4], [3, 5], [4, 5], [4, 6], [6, 8],
                [5, 7], [7, 9], [4, 10], [5, 11], [10, 11], [10, 12], [12, 14], [11, 13], [13, 15]]
    pose_limb_color = palette[[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3]]
    if keyPointInfo:
        keyPoint_dict = {}
        for keyPoint in keyPointInfo:
            index = keyPoint.get("index")
            visible = keyPoint.get("visible", 1)
            confidence = keyPoint.get("confidence")
            point = keyPoint.get("point")
            x = int(float(point["x"]) * img.shape[1])
            y = int(float(point["y"]) * img.shape[0])
            if visible == 1:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            # 绘制点，区分可见性
            cv2.circle(img, (x, y), 5, color, -1)
            keyPoint_dict[index] = [x, y, confidence]
            try:
                # 设置字体
                font = cv2.FONT_HERSHEY_SIMPLEX
                # 使用putText函数在图片上添加文字
                cv2.putText(img, str(index), (x, y), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            except:
                pass
        # plot skeleton
        for sk_id, sk in enumerate(skeleton):
            r, g, b = pose_limb_color[sk_id]
            pos1 = keyPoint_dict[sk[0]][:2]
            pos2 = keyPoint_dict[sk[1]][:2]
            # 绘制线
            cv2.line(img, pos1, pos2, (int(r), int(g), int(b)), thickness=2)
    return img


# 图片文本写入
def drawText(img, label, width, color=(0, 0, 255), alpha=256, font_size=28):
    # 将BGR转换为RGB
    img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个新的RGBA图像用于绘制透明效果
    overlay = Image.new('RGBA', img_PIL.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    # 加载字体
    font = ImageFont.truetype("C:/Windows/Fonts/simkai.ttf", font_size, encoding='utf-8')
    # 获取文本宽高
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width, text_height = math.ceil(bbox[2] - bbox[0]), math.ceil(bbox[3] - bbox[1])
    # 文本位置
    position1 = (width - text_width, 0)
    # 绘制半透明矩形 (RGBA格式，alpha=128表示50%透明度)
    draw.rectangle(
        (position1[0], position1[1], position1[0] + text_width, position1[1] + text_height),
        fill=(color[2], color[1], color[0], alpha)  # 注意颜色顺序是RGB
    )
    draw.text(position1, label, fill=(0, 0, 0, 255), font=font)  # 黑色不透明文本
    # 将透明层与原图合并
    img_PIL = Image.alpha_composite(img_PIL.convert('RGBA'), overlay)
    # 格式转换回cv2 (BGR)
    img = cv2.cvtColor(np.asarray(img_PIL.convert('RGB')), cv2.COLOR_RGB2BGR)
    return img


# OCR文本写入
def drawOcrText(img, x, y, labels, width, height, color=(0, 0, 255), alpha=256, font_size=28):
    # 将BGR转换为RGB
    img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个新的RGBA图像用于绘制透明效果
    overlay = Image.new('RGBA', img_PIL.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    # 初始化参数
    text_width_lst = [0]
    text_height_lst = [0]
    # 加载字体
    font = ImageFont.truetype("C:/Windows/Fonts/simkai.ttf", font_size, encoding='utf-8')
    # 获取中英文最大的高度
    label_test = "中文_English_123"
    bbox_test = draw.textbbox((0, 0), label_test, font=font)
    text_height_test = math.ceil(bbox_test[3] - bbox_test[1])
    for label in labels:
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = math.ceil(bbox[2] - bbox[0])
        text_width_lst.append(text_width)
        text_height_lst.append(text_height_test)
    text_width_max = max(text_width_lst)
    text_height_sum = sum(text_height_lst)
    # 1.位置判断
    x = min(x, int(width - text_width_max))
    y = min(y, int(height - text_height_sum))
    position1 = [x, y]
    # 绘制半透明矩形 (RGBA格式，alpha=128表示50%透明度)
    draw.rectangle(
        (position1[0], position1[1], position1[0] + text_width_max, position1[1] + text_height_sum),
        fill=(color[2], color[1], color[0], alpha)  # 注意颜色顺序是RGB
    )
    # 绘制文本
    for i, label in enumerate(labels):
        position1[1] = position1[1] + text_height_lst[i]
        position2 = (position1[0], position1[1])
        draw.text(position2, label, fill=(0, 0, 0, 255), font=font)  # 黑色不透明文本
    # 将透明层与原图合并
    img_PIL = Image.alpha_composite(img_PIL.convert('RGBA'), overlay)
    # 格式转换回cv2 (BGR)
    img = cv2.cvtColor(np.asarray(img_PIL.convert('RGB')), cv2.COLOR_RGB2BGR)
    return img


# 使用递归函数寻找对应字段的值，输入报文和目标字段即可！
def find_path(data_dict, target_key):
    if isinstance(data_dict, dict):
        for key, value in data_dict.items():
            if key == target_key:
                return value
            else:
                result = find_path(value, target_key)
                if result is not None:
                    return result
    elif isinstance(data_dict, list):
        for item in data_dict:
            result = find_path(item, target_key)
            if result is not None:
                return result
    elif isinstance(data_dict, str):
        try:
            data_dict = json.loads(data_dict)
            for key, value in data_dict.items():
                if key == target_key:
                    return value
                else:
                    result = find_path(value, target_key)
                    if result is not None:
                        return result
        except Exception as e:
            pass
            # logging.error(f"查找字段报错：{e}")
    return None


def unzip_Bin_data(bin_data):
    # 使用 LZMA 解压
    decompressed_data = lzma.decompress(bin_data)
    logging.info(f"bin解压成功！原始大小：{len(bin_data)} 字节，解压后：{len(decompressed_data)} 字节")
    mask = np.frombuffer(decompressed_data, dtype=np.uint8).reshape(720, 1280)

    # plt.figure(figsize=(12, 6))
    # plt.imshow(mask, cmap='gray', vmin=0, vmax=1)
    # plt.title(f'Mask (0-1), 非零像素: {np.count_nonzero(mask)}')
    # plt.colorbar()
    # plt.show()

    return mask, len(decompressed_data)


def unzip_Bin(file_path):
    # 读取二进制文件
    with open(file_path, 'rb') as f:
        bin_data = f.read()
    return unzip_Bin_data(bin_data)


def find_image_annotation_paths(root_path, Images, Annotations, maskData=None):
    obj_path_lst = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        if maskData == None:
            if Images in dirnames and Annotations in dirnames:
                obj_path_lst.append(dirpath)
        else:
            if Images in dirnames and Annotations in dirnames and maskData in dirnames:
                obj_path_lst.append(dirpath)
    return obj_path_lst


def read_json_file(config_dir):
    with open(config_dir, 'r', encoding='utf-8') as file:
        config_data = json.load(file)
        return config_data


# ===================== 分割任务判定（按报文内容特征） =====================
def is_segment_task(file_dir, Images, Annotations, masks, mask_info_name, mask_info_keys, sample_num=10):
    """
    判断是否分割任务：不能只看是否存在 masks/Images/Annotations 三个文件夹
    （其他任务类型也可能有这三个文件夹，会误判），需要抽样报文json，
    只有报文中存在掩码信息字段、且包含掩码格式要求的全部键时，才判定为分割任务。
    """
    obj_path_lst = find_image_annotation_paths(file_dir, Images, Annotations, masks)
    if not obj_path_lst:
        return False
    for obj_path in obj_path_lst:
        ann_dir = os.path.join(obj_path, Annotations)
        mask_dir = os.path.join(obj_path, masks)
        try:
            # 掩码文件夹里需要有实际的掩码文件（排除空文件夹或只有子目录的情况）
            if not any(os.path.isfile(os.path.join(mask_dir, f)) for f in os.listdir(mask_dir)):
                continue
            ann_files = [f for f in os.listdir(ann_dir) if os.path.isfile(os.path.join(ann_dir, f))]
        except Exception as e:
            logging.error(f"分割任务判定遍历报错：{e}")
            continue
        for ann_file in ann_files[:sample_num]:
            try:
                ann_content = sync_read_ann(os.path.join(ann_dir, ann_file))
            except Exception:
                continue
            if not isinstance(ann_content, dict):
                continue
            mask_info = ann_content.get(mask_info_name)
            if isinstance(mask_info, dict) and set(mask_info_keys).issubset(mask_info.keys()):
                logging.info(f"检测到分割任务报文特征（{mask_info_name}字段）：{obj_path}")
                return True
    return False


def first_file_suffix(directory):
    """取目录下第一个文件（跳过子目录）的后缀名"""
    for f in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, f)) and '.' in f:
            return "." + f.split('.')[-1]
    raise FileNotFoundError(f"{directory} 下没有可用文件，无法确定后缀名")


# ===================== 异步管线基础设施 =====================
# 同步IO/编解码函数（供asyncio通过run_in_executor调用）
def sync_read_bytes(file_path):
    with open(file_path, 'rb') as f:
        return f.read()


def sync_read_image(img_path):
    # 读取中文路径图片，cv2.imdecode对应hwc,rgb
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    if img is None:
        raise ValueError("cv2.imdecode 解码失败")
    return img


def sync_read_ann(ann_path):
    # 读取json所有内容，截取到最后一个}防止报文尾部脏数据
    with open(ann_path, encoding='utf-8') as file:
        ann_content = file.read()
    slice_location = ann_content.rfind("}") + 1
    ann_content = ann_content[:slice_location]
    return json.loads(ann_content)


def sync_write_image(dst_path, img, quality=None):
    ext = os.path.splitext(dst_path)[1]
    if ext.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
        ext = ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if quality else []
    success, encoded_img = cv2.imencode(ext, img, params)
    if not success:
        raise IOError("图片编码失败")
    encoded_img.tofile(dst_path)


async def run_pipeline(items, process_fn, cpu_workers):
    """
    异步流水线：
    1. asyncio + IO线程池 异步并发读取 图片/报文/掩码 文件；
    2. ThreadPoolExecutor 多线程异步处理报文（cv2/numpy计算会释放GIL，多线程有真实加速）；
    3. asyncio 写协程消费结果队列，异步并发写出结果图片。
    """
    if not items:
        return
    loop = asyncio.get_running_loop()
    io_executor = ThreadPoolExecutor(max_workers=32)
    cpu_executor = ThreadPoolExecutor(max_workers=cpu_workers)
    read_sem = asyncio.Semaphore(16)   # 限制并发读取数，防止大量图片同时驻留内存
    write_sem = asyncio.Semaphore(8)   # 限制并发写盘数
    write_queue = asyncio.Queue()
    pending_writes = set()

    async def writer():
        # 输出协程：持续消费结果队列，异步写盘
        while True:
            data = await write_queue.get()
            if data is None:
                break
            dst_path, img, quality = data

            async def do_write(p=dst_path, im=img, q=quality):
                async with write_sem:
                    try:
                        await loop.run_in_executor(io_executor, sync_write_image, p, im, q)
                    except Exception as e:
                        logging.error(f"{p} 写入失败！！！ \n {e}")

            t = asyncio.ensure_future(do_write())
            pending_writes.add(t)
            t.add_done_callback(pending_writes.discard)

    writer_task = asyncio.ensure_future(writer())

    async def handle(item):
        # 1. 异步读取图片
        try:
            async with read_sem:
                img = await loop.run_in_executor(io_executor, sync_read_image, item["img_path"])
        except Exception as e:
            logging.error(f"{item['img_path']} 无法读取！！！ \n {e}")
            return
        # 2. 异步读取报文
        try:
            async with read_sem:
                ann_content = await loop.run_in_executor(io_executor, sync_read_ann, item["ann_path"])
        except Exception as e:
            logging.error(f"{item['ann_path']} 无法解析！！！ \n {e}")
            return
        # 3. 异步读取掩码（分割任务）
        mask_bytes = None
        if item.get("mask_path"):
            try:
                async with read_sem:
                    mask_bytes = await loop.run_in_executor(io_executor, sync_read_bytes, item["mask_path"])
            except Exception as e:
                logging.error(f"{item['mask_path']} 无法读取！！！ \n {e}")
                return
        # 4. 线程池异步处理报文
        try:
            result = await loop.run_in_executor(cpu_executor, process_fn, img, ann_content, mask_bytes, item)
        except Exception as e:
            logging.error(f"{item['img_path']} 报文处理报错！！！ \n {e}")
            return
        # 5. 结果放入输出队列，由写协程异步写出
        if result is not None:
            await write_queue.put((item["dst_path"], result, item.get("quality")))

    await asyncio.gather(*(handle(it) for it in items))
    await write_queue.put(None)
    await writer_task
    if pending_writes:
        await asyncio.gather(*pending_writes)
    io_executor.shutdown(wait=True)
    cpu_executor.shutdown(wait=True)


# 解析报文函数
def message_Parsing(file_dir, config_dir):
    # 读取配置文件
    config_data = read_json_file(config_dir)
    # 文件夹名称
    Images = config_data["图片文件夹名称"]
    Annotations = config_data["报文文件夹名称"]
    masks = config_data["掩码文件夹名称"]

    # 无规则分割模式：通过报文中的掩码信息字段判定
    # （其他任务类型也可能有masks/Images/Annotations三个文件夹，仅看文件夹是否存在会误判）
    mask_info_name = config_data.get("掩码信息名称", "maskInfo")
    mask_info_key = config_data.get("掩码信息格式", ["maskData", "maskChn", "maskWidth", "maskHeight", "maskLength"])
    if is_segment_task(file_dir, Images, Annotations, masks, mask_info_name, mask_info_key):
        irregular_Segment(file_dir, config_dir, Images, Annotations, masks)
        return

    # 目标框
    target_array = config_data["目标框数组字段"]
    target_format = config_data["目标框格式"]
    print(target_format)
    target_sec_array = config_data.get("目标框上级数组", "无二级数组！")
    is_splicing = config_data["是否拼接原图与结果图"]
    # 检测标签
    target_tag = config_data["检测标签字段"]
    is_tag = config_data["检测标签是否展示"]
    target_confidence = config_data["检测置信度字段"]
    is_confidence = config_data["检测置信度是否展示"]
    tag_mapping = config_data["检测标签映射字典"]
    is_tag_mapping = config_data["检测标签是否映射"]
    # 属性标签
    target_att = config_data["属性标签字段"]
    is_att = config_data["属性标签是否展示"]
    target_attconf = config_data["属性置信度字段"]
    is_attconf = config_data["属性置信度是否展示"]
    att_mapping = config_data["属性标签映射字典"]
    is_att_mapping = config_data["属性标签是否映射"]
    # 事件标签
    event_tag = config_data["事件标签展示"]
    is_event_tag = config_data["事件标签是否展示"]
    # 其它配置
    color_box = eval(config_data["框颜色BGR"])
    is_num = config_data["目标数量是否展示"]
    image_rule = config_data["规则数组字段"]
    image_rule_format = config_data["规则框格式"]
    is_rule = config_data["规则是否展示"]
    # 关联字段解析
    is_related = config_data["是否存在目标关联"]
    related_src_array = config_data["关联源数组"]
    related_src_tag = config_data["关联源字段"]
    related_dst_array = config_data["关联目标数组"]
    related_dst_tag = config_data["关联目标字段"]
    related_color_box = eval(config_data["关联框颜色BGR"])
    # 可视化透明度
    alpha_tmp = eval(config_data["透明度"])
    # 字体大小解析
    font_size = eval(config_data["字体大小"])
    # 报文处理线程数（可在配置文件中通过"报文处理线程数"覆盖）
    cpu_workers = int(config_data.get("报文处理线程数", min(8, (os.cpu_count() or 4))))

    # 定义支持的图片类型
    img_cls = ["jpg", "jpeg", "png", "bmp"]

    # 配置打包，随任务传递给报文处理线程
    cfg = {
        "target_array": target_array, "target_format": target_format, "target_sec_array": target_sec_array,
        "is_splicing": is_splicing, "target_tag": target_tag, "is_tag": is_tag,
        "target_confidence": target_confidence, "is_confidence": is_confidence,
        "tag_mapping": tag_mapping, "is_tag_mapping": is_tag_mapping,
        "target_att": target_att, "is_att": is_att, "target_attconf": target_attconf,
        "is_attconf": is_attconf, "att_mapping": att_mapping, "is_att_mapping": is_att_mapping,
        "event_tag": event_tag, "is_event_tag": is_event_tag, "color_box": color_box,
        "is_num": is_num, "image_rule": image_rule, "image_rule_format": image_rule_format,
        "is_rule": is_rule, "is_related": is_related, "related_src_array": related_src_array,
        "related_src_tag": related_src_tag, "related_dst_array": related_dst_array,
        "related_dst_tag": related_dst_tag, "related_color_box": related_color_box,
        "alpha_tmp": alpha_tmp, "font_size": font_size,
    }

    obj_path_lst = find_image_annotation_paths(file_dir, Images, Annotations)

    # 收集待处理的图片/报文任务
    items = []
    for obj_path in obj_path_lst:
        # 新建存储结果的文件夹
        result_dir = os.path.join(obj_path, "01报文解析结果/")
        os.makedirs(result_dir, exist_ok=True)
        # 图片文件夹
        img_dir = obj_path + "/" + Images + "/"
        ann_dir = obj_path + "/" + Annotations + "/"

        try:
            ann_suffix = "." + os.listdir(ann_dir)[0].split('.')[-1]
        except Exception as e:
            logging.error(f"查找ann后缀报错：{e}")
        for image in os.listdir(img_dir):
            if image.split('.')[-1] in img_cls:
                img_path = os.path.join(img_dir, image)
                item_ann = os.path.splitext(image)[0] + ann_suffix
                ann_path = os.path.join(ann_dir, item_ann)
                dst_path = os.path.join(result_dir, image)
                items.append({"img_path": img_path, "ann_path": ann_path, "dst_path": dst_path, "cfg": cfg})

    # 异步流水线：asyncio读取文件 + 线程池处理报文 + asyncio输出
    asyncio.run(run_pipeline(items, process_detection_image, cpu_workers))
    logging.info(f'文件解析完成！！！')


def process_detection_image(img, ann_content, mask_bytes, item):
    """单张图片的检测类报文处理（在ThreadPoolExecutor线程中执行），返回结果图"""
    cfg = item["cfg"]
    ann_path = item["ann_path"]
    target_array = cfg["target_array"]
    target_format = cfg["target_format"]
    target_sec_array = cfg["target_sec_array"]
    is_splicing = cfg["is_splicing"]
    target_tag = cfg["target_tag"]
    is_tag = cfg["is_tag"]
    target_confidence = cfg["target_confidence"]
    is_confidence = cfg["is_confidence"]
    tag_mapping = cfg["tag_mapping"]
    is_tag_mapping = cfg["is_tag_mapping"]
    target_att = cfg["target_att"]
    is_att = cfg["is_att"]
    target_attconf = cfg["target_attconf"]
    is_attconf = cfg["is_attconf"]
    att_mapping = cfg["att_mapping"]
    is_att_mapping = cfg["is_att_mapping"]
    event_tag = cfg["event_tag"]
    is_event_tag = cfg["is_event_tag"]
    color_box = cfg["color_box"]
    is_num = cfg["is_num"]
    image_rule = cfg["image_rule"]
    image_rule_format = cfg["image_rule_format"]
    is_rule = cfg["is_rule"]
    is_related = cfg["is_related"]
    related_src_array = cfg["related_src_array"]
    related_src_tag = cfg["related_src_tag"]
    related_dst_array = cfg["related_dst_array"]
    related_dst_tag = cfg["related_dst_tag"]
    related_color_box = cfg["related_color_box"]
    alpha_tmp = cfg["alpha_tmp"]
    font_size = cfg["font_size"]

    img_copy = img.copy()
    width = img.shape[1]
    height = img.shape[0]

    # 按配置解析报文
    target_array_lst = find_path(ann_content, target_array)

    # 初始化字典，用于兼容positionX/positionY
    positionXY_dict = {"x1": float('inf'), "y1": float('inf'), "x2": float('-inf'), "y2": float('-inf')}
    if not target_array_lst:
        logging.error(f'{ann_path} {target_array} 字段解析失败！')
        pass
    # 兼容关联子目标操作
    if is_related:
        related_dst_array_lst = find_path(ann_content, related_dst_array)
        # 重构dst数组
        related_dst_dict = {}
        for related_dst in related_dst_array_lst:
            related_dst_id = find_path(related_dst, related_dst_tag)
            related_dst_dict[related_dst_id] = related_dst

    # 自由编排报文解析
    if target_format == ["center", "width", "height"]:
        for target in target_array_lst:
            # 异常过滤
            if not find_path(target, target_format[0]):
                logging.error(f'{target_format} 与预期不符！')
                continue
            # 目标框解析
            if float(find_path(target, target_format[2])) <= 1:
                x_c = int(float(find_path(target, target_format[0])["x"]) * width)
                y_c = int(float(find_path(target, target_format[0])["y"]) * height)
                w = int(float(find_path(target, target_format[1])) * width)
                h = int(float(find_path(target, target_format[2])) * height)
                x1 = max(0, int(x_c - w / 2))
                x2 = min(width, int(x_c + w / 2))
                y1 = max(0, int(y_c - h / 2))
                y2 = min(height, int(y_c + h / 2))
            else:
                x_c = int(float(find_path(target, target_format[0])["x"]))
                y_c = int(float(find_path(target, target_format[0])["y"]))
                w = int(float(find_path(target, target_format[1])))
                h = int(float(find_path(target, target_format[2])))
                x1 = max(0, int(x_c - w / 2))
                x2 = min(width, int(x_c + w / 2))
                y1 = max(0, int(y_c - h / 2))
                y2 = min(height, int(y_c + h / 2))
            # 标签解析
            labels = []
            # 如果支持事件标签，则目标标签不展示
            if is_event_tag:
                tag = str(find_path(ann_content, event_tag))
                if tag:
                    labels.append(tag)
            else:
                if is_tag:
                    tag = str(find_path(target, target_tag))
                    if is_tag_mapping:
                        tag = tag_mapping[tag]
                    if tag:
                        labels.append(tag)
                if is_confidence:
                    confidence = str(find_path(target, target_confidence))
                    if confidence:
                        labels.append(confidence)
                if is_att:
                    att = str(find_path(target, target_att))
                    if att != "None":
                        if is_att_mapping:
                            att = att_mapping[att]
                        if att:
                            labels.append(att)
                        if is_attconf:
                            attconf = str(find_path(target, target_attconf))
                            if attconf:
                                labels.append(attconf)
            # 可视化
            img = drawOneBox(img, x1, y1, x2, y2, labels, width, height, color=color_box, alpha=alpha_tmp, font_size=font_size, keyPointInfo=False)
    # AIOP报文解析
    elif target_format == ["x", "y", "w", "h"] or target_format == ["x", "y", "width", "height"] or target_format == ["x", "y", "w", "h", "keyPointInfo"]:
        for target in target_array_lst:
            # 异常过滤
            if not find_path(target, target_format[2]):
                continue
            # 目标框解析
            if float(find_path(target, target_format[2])) <= 1:
                x1 = int(float(find_path(target, target_format[0])) * width)
                y1 = int(float(find_path(target, target_format[1])) * height)
                w = int(float(find_path(target, target_format[2])) * width)
                h = int(float(find_path(target, target_format[3])) * height)
                x2 = x1 + w
                y2 = y1 + h
            else:
                x1 = int(float(find_path(target, target_format[0])))
                y1 = int(float(find_path(target, target_format[1])))
                w = int(float(find_path(target, target_format[2])))
                h = int(float(find_path(target, target_format[3])))
                x2 = x1 + w
                y2 = y1 + h
            # 标签
            labels = []
            # 如果支持事件标签，则目标标签不展示
            if is_event_tag:
                tag = str(find_path(ann_content, event_tag))
                if tag:
                    labels.append(tag)
            else:
                if is_tag:
                    tag = str(find_path(target, target_tag))
                    if is_tag_mapping:
                        tag = tag_mapping[tag]
                    if tag:
                        labels.append(tag)
                if is_confidence:
                    confidence = str(find_path(target, target_confidence))
                    if confidence:
                        labels.append(confidence)
                if is_att:
                    att = str(find_path(target, target_att))
                    if att != "None":
                        if is_att_mapping:
                            att = att_mapping[att]
                        if att:
                            labels.append(att)
                        if is_attconf:
                            attconf = str(find_path(target, target_attconf))
                            if attconf:
                                labels.append(attconf)
            # 可视化
            if target_format == ["x", "y", "w", "h", "keyPointInfo"]:
                keyPointInfo = find_path(target, "keyPointInfo")
                img = drawOneBox(img, x1, y1, x2, y2, labels, width, height, color=color_box, alpha=alpha_tmp, font_size=font_size, keyPointInfo=keyPointInfo)
            else:
                img = drawOneBox(img, x1, y1, x2, y2, labels, width, height, color=color_box, alpha=alpha_tmp, font_size=font_size, keyPointInfo=False)
            # 兼容关联子目标操作
            if is_related:
                related_src_array_lst = find_path(target, related_src_array)
                for related_src in related_src_array_lst:
                    related_src_id = find_path(related_src, related_src_tag)
                    related_array_lst = related_dst_dict[related_src_id]
                    # 目标框
                    if float(find_path(related_array_lst, target_format[2])) <= 1:
                        x1 = int(float(find_path(related_array_lst, target_format[0])) * width)
                        y1 = int(float(find_path(related_array_lst, target_format[1])) * height)
                        w = int(float(find_path(related_array_lst, target_format[2])) * width)
                        h = int(float(find_path(related_array_lst, target_format[3])) * height)
                        x2 = x1 + w
                        y2 = y1 + h
                    else:
                        x1 = int(float(find_path(related_array_lst, target_format[0])))
                        y1 = int(float(find_path(related_array_lst, target_format[1])))
                        w = int(float(find_path(related_array_lst, target_format[2])))
                        h = int(float(find_path(related_array_lst, target_format[3])))
                        x2 = x1 + w
                        y2 = y1 + h
                    # 标签
                    labels = []
                    # 如果支持事件标签，则目标标签不展示
                    if is_event_tag:
                        tag = str(find_path(ann_content, event_tag))
                        if tag:
                            labels.append(tag)
                    else:
                        if is_tag:
                            tag = str(find_path(related_array_lst, target_tag))
                            if is_tag_mapping:
                                tag = tag_mapping[tag]
                            if tag:
                                labels.append(tag)
                        if is_confidence:
                            confidence = str(find_path(related_array_lst, target_confidence))
                            if confidence:
                                labels.append(confidence)
                        if is_att:
                            att = str(find_path(related_array_lst, target_att))
                            if att != "None":
                                if is_att_mapping:
                                    att = att_mapping[att]
                                if att:
                                    labels.append(att)
                                if is_attconf:
                                    attconf = str(find_path(target, target_attconf))
                                    if attconf:
                                        labels.append(attconf)
                    # 可视化
                    img = drawOneBox(img, x1, y1, x2, y2, labels, width, height, color=related_color_box, alpha=alpha_tmp, font_size=font_size, keyPointInfo=False)
    # 数据中心算法xyxy类报文解析
    elif target_format == ["x1", "y1", "x2", "y2"]:
        for target in target_array_lst:
            if float(find_path(target, target_format[2])) <= 1:
                x1 = int(float(find_path(target, target_format[0])) * width)
                y1 = int(float(find_path(target, target_format[1])) * height)
                x2 = int(float(find_path(target, target_format[2])) * width)
                y2 = int(float(find_path(target, target_format[3])) * height)
            else:
                x1 = int(float(find_path(target, target_format[0])))
                y1 = int(float(find_path(target, target_format[1])))
                x2 = int(float(find_path(target, target_format[2])))
                y2 = int(float(find_path(target, target_format[3])))
            # 标签
            labels = []
            # 如果支持事件标签，则目标标签不展示
            if is_event_tag:
                tag = str(find_path(ann_content, event_tag))
                if tag:
                    labels.append(tag)
            else:
                if is_tag:
                    tag = str(find_path(target, target_tag))
                    if is_tag_mapping:
                        tag = tag_mapping[tag]
                    if tag:
                        labels.append(tag)
                if is_confidence:
                    confidence = str(find_path(target, target_confidence))
                    if confidence:
                        labels.append(confidence)
                if is_att:
                    att = str(find_path(target, target_att))
                    if att != "None":
                        if is_att_mapping:
                            att = att_mapping[att]
                        if att:
                            labels.append(att)
                        if is_attconf:
                            attconf = str(find_path(target, target_attconf))
                            if attconf:
                                labels.append(attconf)
            # 可视化
            img = drawOneBox(img, x1, y1, x2, y2, labels, width, height, color=color_box, alpha=alpha_tmp, font_size=font_size, keyPointInfo=False)
    # 数据中心算法positionXY类报文解析
    elif target_format == ["positionX", "positionY"]:
        for target in target_array_lst:
            positionXY_dict["x1"] = min(float(find_path(target, target_format[0])), positionXY_dict["x1"])
            positionXY_dict["y1"] = min(float(find_path(target, target_format[1])), positionXY_dict["y1"])
            positionXY_dict["x2"] = max(float(find_path(target, target_format[0])), positionXY_dict["x2"])
            positionXY_dict["y2"] = max(float(find_path(target, target_format[1])), positionXY_dict["y2"])
        # 遍历结束后再进行缩放确认
        if positionXY_dict["x2"] <= 1:
            x1 = int(positionXY_dict["x1"] * width)
            y1 = int(positionXY_dict["y1"] * height)
            x2 = int(positionXY_dict["x2"] * width)
            y2 = int(positionXY_dict["y2"] * height)
        else:
            x1 = int(positionXY_dict["x1"])
            y1 = int(positionXY_dict["y1"])
            x2 = int(positionXY_dict["x2"])
            y2 = int(positionXY_dict["y2"])
        # 标签
        labels = []
        # 如果支持事件标签，则目标标签不展示
        if is_event_tag:
            tag = str(find_path(ann_content, event_tag))
            if tag:
                labels.append(tag)
        else:
            if is_tag:
                tag = str(find_path(target, target_tag))
                if is_tag_mapping:
                    tag = tag_mapping[tag]
                if tag:
                    labels.append(tag)
            if is_confidence:
                confidence = str(find_path(target, target_confidence))
                if confidence:
                    labels.append(confidence)

            if is_att:
                att = str(find_path(target, target_att))
                if att != "None":
                    if is_att_mapping:
                        att = att_mapping[att]
                    if att:
                        labels.append(att)
                    if is_attconf:
                        attconf = str(find_path(target, target_attconf))
                        if attconf:
                            labels.append(attconf)
        # 可视化
        img = drawOneBox(img, x1, y1, x2, y2, labels, width, height, color=color_box, alpha=alpha_tmp, font_size=font_size, keyPointInfo=False)
    # AIOP OCR算法报文解析
    elif target_format == ["x", "y"]:
        for target in target_array_lst:
            target_sec = find_path(target, target_sec_array)
            pts_scale = []
            # 为空则跳过
            if not target_sec:
                continue
            for item_sec in target_sec:
                x = int(float(item_sec["x"]) * width)
                y = int(float(item_sec["y"]) * height)
                pts_scale.append([x, y])
            pts_scale = np.array(pts_scale, np.int32)
            cv2.polylines(img, [pts_scale], isClosed=True, color=(255, 0, 0), thickness=3)
            # 标签
            labels = []
            # 如果支持事件标签，则目标标签不展示
            if is_event_tag:
                tag = str(find_path(ann_content, event_tag))
                if tag:
                    labels.append(tag)
            else:
                if is_tag:
                    tag = str(find_path(target, target_tag))
                    if is_tag_mapping:
                        tag = tag_mapping[tag]
                    if tag:
                        labels.append(tag)
                if is_confidence:
                    confidence = str(find_path(target, target_confidence))
                    if confidence:
                        labels.append(confidence)
                if is_att:
                    att = str(find_path(target, target_att))
                    if att != "None":
                        if is_att_mapping:
                            att = att_mapping[att]
                        if att:
                            labels.append(att)
                        if is_attconf:
                            attconf = str(find_path(target, target_attconf))
                            if attconf:
                                labels.append(attconf)
            # 可视化
            img = drawOcrText(img, pts_scale[0][0], pts_scale[0][1], labels, width, height, color=color_box,
                              alpha=alpha_tmp, font_size=font_size)
    else:
        logging.error(
            f'目标框格式仅支持：["center", "width", "height"]/["x", "y", "w", "h", "keyPointInfo"]/["x", "y", "w", "h"]/["x", "y", "width", "height"]/["x1", "y1", "x2", "y2"]/["positionX", "positionY"]/["x", "y"]')
    if is_num:
        label = f"目标数量：{len(target_array_lst)}"
        img = drawText(img, label, width, color=color_box, alpha=alpha_tmp, font_size=font_size)
    if is_rule:
        ruleinfo = find_path(ann_content, image_rule)
        if ruleinfo:
            if image_rule_format == ["x", "y"]:
                pts_scale = []
                for item_rule in ruleinfo:
                    print(item_rule)
                    x = int(float(item_rule["x"]) * width)
                    y = int(float(item_rule["y"]) * height)
                    pts_scale.append([x, y])
                pts_scale = np.array(pts_scale, np.int32)
                cv2.polylines(img, [pts_scale], isClosed=True, color=(255, 0, 0), thickness=3)
            elif image_rule_format == ["center", "width", "height"]:
                for item_rule in ruleinfo:
                    # 目标框解析
                    x_c = int(float(find_path(item_rule, image_rule_format[0])["x"]) * width)
                    y_c = int(float(find_path(item_rule, image_rule_format[0])["y"]) * height)
                    w = int(float(find_path(item_rule, image_rule_format[1])) * width)
                    h = int(float(find_path(item_rule, image_rule_format[2])) * height)
                    x1 = max(0, int(x_c - w / 2))
                    x2 = min(width, int(x_c + w / 2))
                    y1 = max(0, int(y_c - h / 2))
                    y2 = min(height, int(y_c + h / 2))
                    # 绘制规则区域
                    cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=3)
    # 拼接原图与结果图
    if is_splicing:
        img = cv2.hconcat([img_copy, img])
    return img


def irregular_Segment(file_dir, config_dir, Images, Annotations, masks):
    logging.info("正在执行分割任务...")
    img_cls = ["jpg", "jpeg", "png", "bmp"]
    obj_path_lst = find_image_annotation_paths(file_dir, Images, Annotations, masks)
    config_data = read_json_file(config_dir)
    maskInfoName = config_data["掩码信息名称"]
    maskInfoKey = config_data["掩码信息格式"]
    ruleName = config_data["规则名称"]
    # 报文处理线程数（可在配置文件中通过"报文处理线程数"覆盖）
    cpu_workers = int(config_data.get("报文处理线程数", min(8, (os.cpu_count() or 4))))
    cfg = {"maskInfoName": maskInfoName, "maskInfoKey": maskInfoKey, "ruleName": ruleName}

    # 收集待处理的图片/报文/掩码任务
    items = []
    for obj_path in obj_path_lst:
        result_dir = obj_path + "/" + "01报文解析结果" + "/"
        os.makedirs(result_dir, exist_ok=True)
        img_dir = obj_path + "/" + Images + "/"
        ann_dir = obj_path + "/" + Annotations + "/"
        mask_dir = obj_path + "/" + masks + "/"

        try:
            ann_suffix = first_file_suffix(ann_dir)
            mask_suffix = first_file_suffix(mask_dir)
        except Exception as e:
            logging.error(f"查找ann/mask后缀报错：{e}")
            continue

        for image in os.listdir(img_dir):
            if image.split(".")[-1] in img_cls:
                # 拿到路径
                img_path = os.path.join(img_dir, image)
                item_ann = os.path.splitext(image)[0] + ann_suffix
                ann_path = os.path.join(ann_dir, item_ann)
                item_mask = os.path.splitext(image)[0] + mask_suffix
                mask_path = os.path.join(mask_dir, item_mask)
                image_name = image.split(".")[0]
                dst_path = result_dir + image_name + ".jpeg"
                items.append({"img_path": img_path, "ann_path": ann_path, "mask_path": mask_path,
                              "dst_path": dst_path, "quality": 95, "cfg": cfg})

    # 异步流水线：asyncio读取文件 + 线程池处理报文 + asyncio输出
    asyncio.run(run_pipeline(items, process_segment_image, cpu_workers))
    logging.info("分割任务解析完成！！！")


def process_segment_image(img, ann_content, mask_bytes, item):
    """单张图片的分割报文处理（在ThreadPoolExecutor线程中执行），返回结果图"""
    cfg = item["cfg"]
    ann_path = item["ann_path"]
    maskInfoName = cfg["maskInfoName"]
    maskInfoKey = cfg["maskInfoKey"]
    ruleName = cfg["ruleName"]
    maskData = "maskData"
    maskChn = "maskChn"
    maskWidth = "maskWidth"
    maskHeight = "maskHeight"
    maskLength = "maskLength"

    img_copy = img.copy()

    try:
        ann_mask_key = ann_content[maskInfoName]
    except Exception as e:
        logging.error(f"{ann_path} 中无掩码信息 \n {e}")
        return None

    if not set(maskInfoKey).issubset(ann_mask_key.keys()):
        logging.info("maskInfo信息不完全！！！")
        return None

    current_mask, act_len = unzip_Bin_data(mask_bytes)
    mask_len = int(ann_mask_key[maskLength] / ann_mask_key[maskChn])

    if mask_inspect(mask_len, act_len):
        return None
    mask_resized = cv2.resize(current_mask, (img_copy.shape[1], img_copy.shape[0]))

    # 创建彩色覆盖层
    overlay = img_copy.copy()
    overlay[mask_resized == 1] = [0, 230, 230]  # 淡黄色标记掩码区域

    # 半透明混合
    alpha = 0.4
    result = cv2.addWeighted(img_copy, 1 - alpha, overlay, alpha, 0)

    # 绘制加粗边缘
    mask_uint8 = mask_resized.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(result, contours, -1, [255, 255, 255], 2)
    name = find_path(ann_content, ruleName)
    for i, contour in enumerate(contours):
        result = add_text_box_pil(result, contour, name, font_size=16)

    logging.info("分割完成...")
    return result


def mask_inspect(mask_len, act_len):
    if mask_len != act_len:
        logging.info("mask解压大小与实际不一致！！！")
        return True

    return False


def add_text_box_pil(image, contour, text, font_size=20,
                    text_color=(255, 255, 255), bg_color=(0, 0, 0),
                    border_color=(100, 100, 100), border_width=2,
                    radius=10, padding_ratio=0.5, margin=0, alpha=0.8):
    """
    使用PIL绘制圆角文本框（自适应padding版本）

    Args:
        padding_ratio: padding与字体大小的比例（0.5表示padding=字体大小的一半）
    """
    contour_points = contour.reshape(-1, 2)

    # 找到轮廓左上角的点
    x, y, w, h = cv2.boundingRect(contour)
    distances = np.sqrt((contour_points[:, 0] - x)**2 + (contour_points[:, 1] - y)**2)
    nearest_idx = np.argmin(distances)
    contour_x = contour_points[nearest_idx, 0]
    contour_y = contour_points[nearest_idx, 1]

    img_h, img_w = image.shape[:2]

    # 转换为PIL格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    # 尝试加载字体

    font = ImageFont.truetype("C:/Windows/Fonts/simkai.ttf", font_size, encoding='utf-8')


    # 获取文字大小
    temp_draw = ImageDraw.Draw(pil_image)
    bbox = temp_draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # 自适应padding
    padding_x = int(font_size * padding_ratio)
    padding_y = int(font_size * padding_ratio * 0.8)

    # 文本框总尺寸
    box_w = text_w + padding_x * 2
    box_h = text_h + padding_y * 2

    # 默认位置：左下角贴近轮廓点
    box_x1 = contour_x
    box_y1 = contour_y - box_h - margin  # 文本框底部贴近轮廓点

    # 检查是否超出画布
    if box_x1 < 0 or box_y1 < 0 or box_x1 + box_w > img_w or box_y1 + box_h > img_h:
        # 改为右上角贴近轮廓点
        box_x1 = contour_x - box_w - margin  # 文本框右侧贴近轮廓点
        box_y1 = contour_y  # 文本框顶部贴近轮廓点

    # 确保不超出画布边界
    box_x1 = max(0, min(box_x1, img_w - box_w))
    box_y1 = max(0, min(box_y1, img_h - box_h))

    box_x2 = box_x1 + box_w
    box_y2 = box_y1 + box_h

    # 创建半透明图层
    overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    # 绘制半透明圆角背景框
    bg_color_rgba = (*bg_color, int(255 * alpha))
    border_color_rgba = (*border_color, int(255 * alpha))

    overlay_draw.rounded_rectangle(
        [box_x1, box_y1, box_x2, box_y2],
        radius=radius,
        fill=bg_color_rgba,
        outline=border_color_rgba,
        width=border_width
    )

    # 计算文字居中位置
    text_center_x = box_x1 + (box_w - text_w) / 2
    text_center_y = box_y1 + (box_h - text_h) / 2

    # 绘制文字（半透明）
    text_color_rgba = (*text_color, int(255 * alpha))
    overlay_draw.text((text_center_x, text_center_y),
                     text, font=font, fill=text_color_rgba)

    # 合并图层
    pil_image = pil_image.convert('RGBA')
    pil_image = Image.alpha_composite(pil_image, overlay)

    # 转换回RGB
    pil_image = pil_image.convert('RGB')

    # 转换回OpenCV格式
    result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return result


# 函数测试
if __name__ == '__main__':
    config_dir = rf"E:\Desktop\报文解析综合工具\config\setting_JX20251121.json"
    file_dir = rf"E:\Desktop\fileDir"
    message_Parsing(file_dir, config_dir)
