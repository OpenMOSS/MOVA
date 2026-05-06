import re
import logging

from typing import Any

SPACE_LIST = [' ', '\t', '\r', '\n']
PUNCTS = [
    '!', ',', '?', '、', '。', '！', '，', '；', '？', '：', '「', '」', '︰', '『', '』',
    '《', '》', '.', '-', '|', '(', ')', '[', ']', '{', '}', '"', "'", '…', '·', '—', '～',
    '【', '】', '〔', '〕', '〈', '〉', '﹃', '﹄', '﹁', '﹂', '‘', '’', '“', '”',
]


logger = logging.getLogger(__name__)


def remove_punct(text: str) -> str:
    """
    移除标点符号并转换为小写。
    
    Args:
        text (str): 输入文本
        
    Returns:
        str: 移除标点后的小写文本
    """
    for p in SPACE_LIST + PUNCTS:
        text = text.replace(p, '')
    text = text.lower()
    return text


def convert_time_to_seconds(time_str: str) -> float:
    """
    将各种时间格式的字符串转换为秒数。
    
    支持的格式:
    - 纯秒数: "123.45", "123", "0.5"
    - MM:SS.ss: "01:23.45", "01:23"
    - HH:MM:SS.ss: "01:23:45.67", "01:23:45"
    - HH:MM:SS: "1:23:45"
    
    Args:
        time_str (str): 时间字符串
        
    Returns:
        float: 总秒数
        
    Example:
        "123.45" -> 123.45
        "01:23.45" -> 83.45
        "01:23:45.67" -> 5025.67
        "1:23:45" -> 5025.0
    """
    time_str = time_str.strip()

    # 尝试分割时间字符串
    parts = time_str.split(':')
    
    if len(parts) == 1:
        # 只有秒数
        try:
            return float(parts[0])
        except ValueError:
            raise ValueError(f"无法解析时间格式: {time_str}")
    
    elif len(parts) == 2:
        # MM:SS 或 MM:SS.ss 格式
        try:
            minutes = int(parts[0])
            seconds_part = parts[1]
            
            # 处理秒数部分（可能包含小数）
            seconds = float(seconds_part)
            
            return minutes * 60 + seconds
        except ValueError:
            raise ValueError(f"无法解析时间格式: {time_str}")
    
    elif len(parts) == 3:
        # HH:MM:SS 或 HH:MM:SS.ss 格式
        try:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            
            return hours * 3600 + minutes * 60 + seconds
        except ValueError:
            raise ValueError(f"无法解析时间格式: {time_str}")
    
    else:
        raise ValueError(f"不支持的时间格式: {time_str}")


def parse_timestamp_text(text: str) -> list[dict[str, Any]]:
    """
    解析带时间戳的多说话人文本，支持多种时间格式：
    
    格式1 (秒数): [0.00] S1: 你好 [1.23] [2.23] S2: 再见 [3.44]
    格式2 (MM:SS): [01:23.45] S1: 你好 [01:25.67] [01:30.00] S2: 再见 [01:35.12]
    格式3 (HH:MM:SS): [01:23:45.67] S1: 你好 [01:23:50.00] [01:24:00.00] S2: 再见 [01:24:10.00]
    格式4 (无内容): [0.00] S1 [1.23] [2.23] S2 [3.44]
    
    Args:
        text (str): 带时间戳的输入文本
        
    Returns:
        List[Dict]: 包含说话人信息的列表，每个元素包含:
            - speaker: 说话人标识 (如 "S1", "S2")
            - content: 说话内容 (可能为空字符串)
            - start_time: 开始时间 (float, 秒)
            - end_time: 结束时间 (float, 秒)
    
    Example:
        Input: "[0.00] S1: 你好 [1.23] [2.23] S2: 再见 [3.44]"
        或者: "[0.00] S1 [1.23] [2.23] S2 [3.44]"
        Output: [
            {"speaker": "S1", "content": "你好", "start_time": 0.00, "end_time": 1.23},
            {"speaker": "S2", "content": "再见", "start_time": 2.23, "end_time": 3.44}
        ]
    """
    # 通用正则表达式，匹配各种时间格式
    # 匹配 [时间] 说话人:? 内容? [时间]
    time_patterns = [
        r'(\d{1,2}:\d{1,2}:\d{1,2}\.?\d*)',  # HH:MM:SS.ss 格式
        r'(\d{1,2}:\d{1,2}\.?\d*)',          # MM:SS.ss 格式  
        r'(\d+\.?\d*)',                      # 秒数格式
    ]
    
    # 尝试不同的时间格式模式
    for time_pattern in time_patterns:
        # 构建完整的匹配模式，使冒号和内容都变为可选
        pattern = rf'\[{time_pattern}\]\s*(S\d+)(?::\s*([^[]*)?)?\s*\[{time_pattern}\]'
        matches = re.findall(pattern, text)
        
        if matches:
            segments = []
            for match in matches:
                start_time_str, speaker, content, end_time_str = match
                # 如果content为None（没有冒号和内容），则设为空字符串
                content = content.strip() if content else ""
                segments.append({
                    "speaker": speaker,
                    "content": content,
                    "start_time": convert_time_to_seconds(start_time_str),
                    "end_time": convert_time_to_seconds(end_time_str)
                })
            return segments

    logger.warning(f"Failed to parse any timestamp format in text: {text}, returning empty list.")
    return []


if __name__ == "__main__":
    # 测试代码
    test_text = "[0.00] S1: 你好 [1.23] [2.23] S2: 再见 [3.44]"
    result = parse_timestamp_text(test_text)
    print(result)

    test_text2 = "[01:23.45] S1: 你好 [01:25.67] [01:30.00] S2: 再见 [01:35.12]"
    result2 = parse_timestamp_text(test_text2)
    print(result2)
    
    test_text3 = "[01:23:45.67] S1: 你好 [01:23:50.00] [01:24:00.00] S2: 再见 [01:24:10.00]"
    result3 = parse_timestamp_text(test_text3)
    print(result3)

    test_text4 = "[0.5 S1: 你好 [1.0] S2: 再见"
    result4 = parse_timestamp_text(test_text4)
    print(result4)

    test_text5 = "[0.00] S1 [1.23] [2.23] S2 [3.44]"
    result5 = parse_timestamp_text(test_text5)
    print(result5)