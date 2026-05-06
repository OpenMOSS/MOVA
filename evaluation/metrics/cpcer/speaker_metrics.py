import re
import itertools
import logging

from collections import defaultdict
from typing import Callable

from jiwer import cer as cer_metric
from jiwer import process_characters

from speaker_timestamp_utils import remove_punct
import re
from typing import List, Dict, Any

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

logger = logging.getLogger(__name__)

def distance_fn(x, y):
    result = process_characters(x, y)
    return result.substitutions + result.insertions + result.deletions


def _minimum_permutation_assignment(
        reference: dict[str, str],
        hypothesis: dict[str, str],
        distance_fn: Callable[[str, str], int] = distance_fn,
        missing: str = ""
) -> tuple:
    """
    Compute the best (lowest distance) assignment of reference and hypothesis
    speakers based on `distance_fn`.

    Returns:
        The assignment and the distance
        The score matrix. Shape (reference hypothesis)

    >>> _minimum_permutation_assignment({}, {}, lambda x, y: 0)
    ((), 0, array([], dtype=float64))
    >>> _minimum_permutation_assignment({}, {'spkA': meeteval.io.SegLST([])}, lambda x, y: 1)
    (((None, 'spkA'),), 1, array([[1]]))
    >>> _minimum_permutation_assignment({'spkA': meeteval.io.SegLST([])}, {}, lambda x, y: 1)
    ((('spkA', None),), 1, array([[1]]))
    """
    import scipy.optimize
    import numpy as np

    cost_matrix = np.array([
        [
            distance_fn(tt, et)
            for et, _ in itertools.zip_longest(
            hypothesis.values(),
            reference.values(),  # ignored, "padding" for underestimation
            fillvalue=missing,
        )
        ]
        for tt, _ in itertools.zip_longest(
            reference.values(),
            hypothesis.values(),  # ignored, "padding" for overestimation
            fillvalue=missing,
        )
    ])

    if cost_matrix.size == 0:
        return (), 0, cost_matrix

    # Find the best permutation with hungarian algorithm
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
    distances = cost_matrix[row_ind, col_ind]
    distances = list(distances)

    # Compute WER from distance
    distance = sum(distances)

    # need `dict.get` of the keys for overestimation
    reference_keys = dict(enumerate(reference.keys()))
    hypothesis_keys = dict(enumerate(hypothesis.keys()))

    assignment = tuple([
        (reference_keys.get(r), hypothesis_keys.get(c))
        for r, c in itertools.zip_longest(row_ind, col_ind)
    ])
    return assignment, int(distance), cost_matrix


def remove_speaker(text: str) -> str:
    """
    Remove speaker tags (e.g., [S1], [S2], etc.) from the input text.

    Args:
        text (str): Input text containing speaker tags.

    Returns:
        str: Text with all speaker tags removed and extra spaces cleaned up.
    
    Example:
        Input: "Hello [S1] how are you [S2] fine"
        Output: "Hello how are you fine"
    """
    return re.sub(r'\s*\[S\d+\]\s*', ' ', text).strip()


def cer_without_speaker(predictions, references, return_detailed=False):
    """
    Compute CER after removing speaker tags.

    Args:
        predictions (list of str): List of predicted transcriptions with speaker tags.
        references (list of str): List of reference transcriptions with speaker tags.
        return_detailed (bool): 是否返回详细的 CER 信息，包括每个样本的错误率。

    Returns:
        float or dict: 
            - 如果 return_detailed=False: CER 的百分比表示（乘以 100）。
            - 如果 return_detailed=True: 包含详细信息的字典，格式为：
                {
                    'overall_cer': float,           # 总体 CER
                    'sample_details': list,         # 每个样本的详细信息
                    'total_chars': int,             # 总字符数
                    'total_errors': float,          # 总错误数
                    'num_samples': int              # 样本数量
                }
    """
    total_chars = 0   # 所有样本参考文本的字符总数
    total_errors = 0  # 加权累计的错误字符数
    sample_details = []  # 存储每个样本的详细信息

    for idx, (pred_text, ref_text) in enumerate(zip(predictions, references)):
        # 去掉 speaker 标签
        cleaned_pred = remove_speaker(pred_text)
        cleaned_ref = remove_speaker(ref_text)

        # 去掉标点符号
        cleaned_pred = remove_punct(cleaned_pred)
        cleaned_ref = remove_punct(cleaned_ref)
        
        # 计算单个样本的 CER
        cer_result = process_characters([cleaned_ref], [cleaned_pred])
        sample_cer = cer_result.cer
        # FIXME: 由于 jiwer 的 bug，这里算的编辑距离是 reference 到 hypothesis 的，因此 deletions 实际上对应了 insertions
        sample_ref_chars = cer_result.hits + cer_result.substitutions + cer_result.deletions
        sample_error_chars = cer_result.substitutions + cer_result.insertions + cer_result.deletions

        # 收集详细信息
        if return_detailed:
            sample_detail = {
                'sample_index': idx,
                # 'reference_text': ref_text,
                # 'prediction_text': pred_text,
                'cleaned_reference': cleaned_ref,
                'cleaned_prediction': cleaned_pred,
                'sample_cer': sample_cer * 100,  # 转换为百分比
                'ref_chars': sample_ref_chars,
                'error_chars': sample_error_chars,
            }
            sample_details.append(sample_detail)

        total_chars += sample_ref_chars
        total_errors += sample_error_chars

    final_cer = 100 * total_errors / total_chars if total_chars > 0 else 0
    
    if return_detailed:
        return {
            'overall_cer': final_cer,
            'sample_details': sample_details,
            'total_chars': total_chars,
            'total_errors': total_errors,
            'num_samples': len(predictions)
        }
    else:
        return final_cer


def split_by_speaker(text):
    """
    按说话人标签分割文本，将每个说话人的发言提取出来。
    
    使用正则表达式查找类似 "[S1]" 的标签，然后将对应文本（直到下一个说话人标签）归为该说话人，
    如果同一说话人出现多次，则将文本连接起来。非说话人标签（如[a]）会被归入前一个说话人。
    
    Args:
        text (str): 带有说话人标签的完整转录文本。
        
    Returns:
        dict: 键为说话人标签（如 "[S1]"），值为该说话人的所有发言文本（已去除标签）。
    """
    # 先找到所有说话人标签的位置
    speaker_positions = []
    for match in re.finditer(r'\[S\d+\]', text):
        speaker_positions.append((match.start(), match.end(), match.group()))
    
    if not speaker_positions:
        return {}
    
    speaker_text = defaultdict(str)
    
    for i, (start, end, speaker) in enumerate(speaker_positions):
        # 确定当前说话人片段的结束位置（下一个说话人标签的开始位置）
        if i < len(speaker_positions) - 1:
            next_start = speaker_positions[i + 1][0]
            content = text[end:next_start]
        else:
            # 最后一个说话人，取到文本末尾
            content = text[end:]
        
        speaker_text[speaker] += content.strip()
    
    return dict(speaker_text)

def cp_cer(predictions, references, return_detailed=False):
    """
    计算 cpCER (Concatenated Permutation Character Error Rate)
    对于多说话人的转录，允许预测中说话人数量与参考中不一致，通过填充空字符串解决，
    然后对预测段按所有可能顺序进行排列，取拼接后 CER 最小者作为该样本的错误率。
    
    Args:
        predictions (list of str): 预测文本列表，每条文本包含说话人标签（如 "[S1]", "[S2]"）。
        references (list of str): 参考文本列表，每条文本包含说话人标签。
        return_detailed (bool): 是否返回详细的 CER 信息，包括每个样本的错误率。
        
    Returns:
        float or dict: 
            - 如果 return_detailed=False: cpCER 的百分比表示（乘以 100）。
            - 如果 return_detailed=True: 包含详细信息的字典，格式为：
                {
                    'overall_cer': float,           # 总体 cpCER
                    'sample_details': list,         # 每个样本的详细信息
                    'total_chars': int,             # 总字符数
                    'total_errors': float,          # 总错误数
                    'num_samples': int              # 样本数量
                }
    """
    total_chars = 0   # 所有样本参考文本的字符总数
    total_errors = 0  # 加权累计的错误字符数
    sample_details = []  # 存储每个样本的详细信息

    for idx, (pred_text, ref_text) in enumerate(zip(predictions, references)):
        # 根据说话人标签划分出发言段（字典 key 为标签，value 为文本）
        pred_spk_text = split_by_speaker(pred_text)
        ref_spk_text = split_by_speaker(ref_text)

        # 删除标点
        pred_spk_text = {k: remove_punct(v) for k, v in pred_spk_text.items()}
        ref_spk_text = {k: remove_punct(v) for k, v in ref_spk_text.items()}

        max_spk_num = max(len(pred_spk_text), len(ref_spk_text))
        if max_spk_num > 100:
            logger.warning(f"Too many speakers: {max_spk_num}, use default assignment")
            assignment = [
                (spk1, spk2) for spk1, spk2 in itertools.zip_longest(
                    ref_spk_text.keys(),
                    pred_spk_text.keys(),
                    fillvalue=None
                )
            ]
        else:
            assignment, distance, score_matrix = _minimum_permutation_assignment(
                ref_spk_text,
                pred_spk_text,
                distance_fn=distance_fn
            )

        ref_text_list = [
            ref_spk_text[k[0]] if k[0] else "" for k in assignment
        ]

        pred_text_list = [
            pred_spk_text[k[1]] if k[1] else "" for k in assignment
        ]

        cer_result = process_characters(ref_text_list, pred_text_list)
        sample_cer = cer_result.cer
        # FIXME: 由于 jiwer 的 bug，这里算的编辑距离是 reference 到 hypothesis 的，因此 deletions 实际上对应了 insertions
        sample_ref_chars = cer_result.hits + cer_result.substitutions + cer_result.deletions
        sample_error_chars = cer_result.substitutions + cer_result.insertions + cer_result.deletions

        # 收集详细信息
        if return_detailed:
            sample_detail = {
                'sample_index': idx,
                # 'reference_text': ref_text,
                # 'prediction_text': pred_text,
                'best_cer': sample_cer * 100,  # 转换为百分比
                'ref_chars': sample_ref_chars,
                'error_chars': sample_error_chars,
                'assignment': assignment,
            }
            sample_details.append(sample_detail)

        total_chars += sample_ref_chars
        total_errors += sample_error_chars

    final_cp_cer = 100 * total_errors / total_chars if total_chars > 0 else 0
    
    if return_detailed:
        return {
            'overall_cer': final_cp_cer,
            'sample_details': sample_details,
            'total_chars': total_chars,
            'total_errors': total_errors,
            'num_samples': len(predictions)
        }
    else:
        return final_cp_cer


def parse_speaker_tags_in_order(text):
    """
    从文本中按出现顺序提取 [S\\d+] 标签，并将其映射成连续的数字ID。
    例如:
        "[S1] [S2] [S2]" -> [1, 2, 2]
        "[S1] [S1] [S2]" -> [1, 1, 2]

    若发现某个标签的编号 > 9，就 raise ValueError。
    """
    pattern = r'\[S(\d+)\]'
    tags = re.findall(pattern, text)  # 找出所有 "1","2","10" 等数字字符串
    
    mapping = {}
    current_id = 1
    sequence = []

    for t in tags:
        num = int(t)
        # if num > 9:
        #     raise ValueError(f"说话人编号 S{num} 超过9，不支持！")

        # 如果没见过这个编号，就赋予下一个可用ID
        if t not in mapping:
            mapping[t] = current_id
            current_id += 1
        sequence.append(str(mapping[t]))
    
    return sequence

def speaker_cp_cer(predictions, references):
    """
    对齐多条 (pred, ref) 的说话人序列，计算 Concatenated Permutation CER (字符级)。
    1) 只比较标签本身 (如 S1 -> "1")，每个标签只算1个字符
    2) 若标签集合不一致或数量超过9，直接报错
    3) 穷举预测标签与参考标签的所有映射方式，取最优 CER
    4) 最后返回一个百分比形式的 cpCER
    
    Args:
        predictions (list of str): 预测文本列表（仅包含说话人标签或混有其他文本都可以）
        references (list of str): 参考文本列表

    Returns:
        float: cpCER 百分比 (0~100)

    Example:
        pred = ["[S1] [S2] [S2]", "[S2] [S1]"]
        ref  = ["[S1] [S1] [S2]", "[S1] [S2]"]

        score = cp_cer_speaker_tags(pred, ref)
    """
    total_chars = 0
    total_errors = 0

    for pred_text, ref_text in zip(predictions, references):
        # 1) 提取标签序列(如 ["1","2","2"])
        pred_seq = parse_speaker_tags_in_order(pred_text)
        ref_seq = parse_speaker_tags_in_order(ref_text)

        # print(f"pred_seq: {pred_seq}, ref_seq: {ref_seq}")

        # 2) 看双方标签集合是否相同 & 个数是否超过9
        pred_set = set(pred_seq)
        ref_set = set(ref_seq)

        # 如果都没有标签，相当于啥都没有，就继续
        if len(pred_seq) == 0 and len(ref_seq) == 0:
            continue

        # 3) 穷举所有可能的标签映射方式: P->R
        #    （思路：对pred_set排序，把它映射到所有perm(ref_set)）
        unique_labels = sorted(pred_set | ref_set)
        min_cer = float('inf')

        # 准备所有排列
        if len(unique_labels) > 8:
            print(f"[WARNING] 说话人种类多于8个：{unique_labels}，无法高效计算，只计算一种映射")
            all_labels = [unique_labels]
        else:
            all_labels = itertools.permutations(unique_labels)
        # 假设 unique_labels = ["1","2"]
        # all_labels 可能是 [("1","2"), ("2","1")]

        # 把参考序列拼成 "112" 形式的字符串，用于 CER 计算
        ref_str = "".join(ref_seq)

        for perm in all_labels:
            # perm 可能是 ("1","2") 或 ("2","1")...
            # 我们要构造一个映射 dict:
            #   原预测标签 unique_labels[i] -> perm[i]
            mapping = { original: new for original, new in zip(unique_labels, perm) }

            # 根据 mapping 替换预测序列
            mapped_pred_seq = [mapping[x] for x in pred_seq]
            pred_str = "".join(mapped_pred_seq)

            # 用 evaluate 里的 CER 做字符级比对，注意它返回 0~1 之间的小数
            cur_cer = cer_metric([ref_str], [pred_str])
            # print(f"perm: {perm}, pred_str: {pred_str}, ref_str: {ref_str}, cur_cer: {cur_cer}")

            if cur_cer < min_cer:
                min_cer = cur_cer

        # 4) 加权累加
        # 参考总字符数就是 ref_seq 长度，因为每个标签就是1个字符
        ref_length = len(ref_seq)
        total_chars += ref_length
        total_errors += min_cer * ref_length

    # 5) 转成百分比
    final_cer = 100.0 * total_errors / total_chars if total_chars > 0 else 0.0
    return final_cer

# 仅匹配秒数：[12] / [12.3] / [12.345]（小数位不限）
_TS_RE  = re.compile(r"^\d+(?:\.\d+)?$")
# 仅匹配 S+数字 的说话人标签：[S1] / [S23]
_SPK_RE = re.compile(r"^S\d+$")
def _parse_seconds_speaker_text(text: str) -> Annotation:
    """
    解析形如 [start_sec][Sx] ... [end_sec] 或 [start_sec] ... [end_sec] 的文本为 pyannote Annotation。
    - 优先使用当前区间内出现的说话人标签 [Sx]
    - 若区间内没有说话人，则回退为上一个已记录片段的说话人
    - 若仍无可用说话人，则跳过该区间
    忽略非时间/说话人的其他方括号内容。
    """
    ann = Annotation()
    if not text:
        return ann

    # 抽取所有方括号token
    tokens = []
    for m in re.finditer(r"\[([^\]]+)\]", text):
        raw = m.group(1).strip()
        if _TS_RE.match(raw):
            tokens.append(("time", float(raw)))
        elif _SPK_RE.match(raw):
            tokens.append(("spk", raw))
        else:
            # 其他方括号（如 [noise]）忽略
            continue

    # 按 [time] (可选若干 [spk]) ... [time] 组装片段
    i, n = 0, len(tokens)
    last_spk = None  # 记录上一个成功写入片段的说话人
    while i < n:
        kind, val = tokens[i]
        if kind != "time":
            i += 1
            continue

        start = float(val)

        # 在下一个 [time] 之前，收集最后一个出现的 [spk]
        j = i + 1
        seg_spk = None
        while j < n and tokens[j][0] != "time":
            if tokens[j][0] == "spk":
                seg_spk = tokens[j][1]  # 若出现多个 [spk]，以最后一个为准
            j += 1

        # 找到结束时间
        if j < n and tokens[j][0] == "time":
            end = float(tokens[j][1])
            # 确定要用的说话人：区间内的 [spk] 优先，否则回退为 last_spk
            spk_to_use = seg_spk if seg_spk is not None else last_spk

            if spk_to_use is not None and end > start:
                ann[Segment(start, end)] = str(spk_to_use)
                last_spk = str(spk_to_use)  # 仅在成功写入片段后更新“上一个说话人”
            # 无可用说话人或 end<=start 的情况直接跳过

            # 跳到 end 之后继续
            i = j + 1
        else:
            # 没有结束时间，终止
            break

    return ann

def speaker_timestamp_der(
    predictions: List[str],
    references: List[str],
    *,
    return_detailed: bool = False,
    collar: float = 0.0,
    skip_overlap: bool = False,
):
    """
    计算 DER（百分比）。输入每条为若干片段拼接，如：
      "[0.00][S1] content [1.23][1.25][S2] other [3.00]"
    只要满足 S+数字 和 秒数时间戳即可。
    """
    assert len(predictions) == len(references), "predictions 与 references 数量不一致"

    metric_total = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    details: List[Dict[str, Any]] = []

    for i, (pred, ref) in enumerate(zip(predictions, references)):
        hyp = _parse_seconds_speaker_text(pred or "")
        gt  = _parse_seconds_speaker_text(ref or "")

        # 单样本度量，便于拿到分子/分母
        m = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
        der = m(gt, hyp)  # 0~1
        total = m["total"] if m["total"] > 0 else 1e-12

        # 累积到总体
        metric_total(gt, hyp)

        if return_detailed:
            details.append({
                "sample_index": i,
                "der": der * 100.0,
                "total": total,
                "false_alarm": m["false alarm"],
                "missed_detection": m["missed detection"],
                "confusion": m["confusion"],
                "num_segments_pred": len(list(hyp.itertracks())),
                "num_segments_ref":  len(list(gt.itertracks())),
            })

    denom = metric_total["total"] if metric_total["total"] > 0 else 1e-12
    overall_der = 100.0 * (
        metric_total["false alarm"] +
        metric_total["missed detection"] +
        metric_total["confusion"]
    ) / denom

    if not return_detailed:
        return overall_der

    return {
        "overall_der": overall_der,
        "sample_details": details,
        "total": denom,
        "false_alarm": metric_total["false alarm"],
        "missed_detection": metric_total["missed detection"],
        "confusion": metric_total["confusion"],
        "num_samples": len(predictions),
        "collar": collar,
        "skip_overlap": skip_overlap,
    }
# if __name__ == "__main__":
#     # Example usage
#     predictions = ["[S2]啊您就刚刚说到这个超高频交易，就是Ultra-[S1]对。 [S2]... high frequency。 [S1]啊， Ultra-high frequency。 [S2] Ultra-high frequency。 [S1]啊。 [S2]也让我想到了，可能最近两年一本比较畅销的书，叫Flash Boy。 [S1]啊，Flash，Flash boy嘛。 [S2]嗯，Flash boy嘛。 [S1]就是Michael Lewis- [S2] Flash boy讲的就是，看他对Michael Lewis讲的他就是-"]
#     references = ["[S2]啊，您这刚刚说到这个超高频交易，就是ultra [S1]对 [S2]high frequency. [S1]啊，ul- ultra high frequency. [S2]也让我想到了可能最近两年一本比较畅销的书，叫Flash Boy. [S1]啊，Flash Boy嘛，就是- [S2]就是那个Michael Lewis. [S1]The Flash Boy讲的就是...啊，对，Michael Lewis讲的他就是-"]
    
#     # 简单模式
#     cer_score = cer_without_speaker(predictions, references)
#     print(f"CER Score: {cer_score}")
    
#     # 详细模式
#     detailed_result = cer_without_speaker(predictions, references, return_detailed=True)
#     if isinstance(detailed_result, dict):
#         print(f"\n详细结果:")
#         print(f"总体CER: {detailed_result['overall_cer']:.2f}%")
#         print(f"总字符数: {detailed_result['total_chars']}")
#         print(f"总错误数: {detailed_result['total_errors']}")
#         print(f"样本数量: {detailed_result['num_samples']}")
        
#         for detail in detailed_result['sample_details']:
#             print(f"\n样本 {detail['sample_index']}:")
#             print(f"  清理后预测: {detail['cleaned_prediction']}")
#             print(f"  清理后参考: {detail['cleaned_reference']}")
#             print(f"  样本CER: {detail['sample_cer']:.2f}%")
#             print(f"  参考字符数: {detail['ref_chars']}")
#             print(f"  错误字符数: {detail['error_chars']}")

if __name__ == "__main__":
    # Example usage —— 采用 [秒][S数字] 内容 [秒] 的格式
    predictions = [
        # 片段1：S1 说话 0.00~1.20
        # 片段2：S2 说话 1.20~2.50
        # 片段3：S1 再说 2.50~3.40
        "[0.00][S1] 大家好 [1.10]"
        "[1.10] 我是S2说话 [2.60]"
        "[2.60][S2] 继续 [3.50]"
    ]
    references = [
        # 参考里时间稍有不同，DER 应该 > 0
        "[0.00][S1] 大家好 [1.10]"
        "[1.10][S1] 我是 S2 说话 [2.60]"
        "[2.60][S2] 继续 [3.50]"
    ]

    # 简单模式
    der_score = speaker_timestamp_der(predictions, references)
    print(f"DER Score: {der_score:.2f}%")

    # 详细模式
    detailed = speaker_timestamp_der(predictions, references, return_detailed=True)
    if isinstance(detailed, dict):
        print("\n详细结果:")
        print(f"总体 DER: {detailed['overall_der']:.2f}%")
        print(f"总时长(total): {detailed['total']:.2f}")
        print(f"False Alarm: {detailed['false_alarm']:.2f}")
        print(f"Missed Detection: {detailed['missed_detection']:.2f}")
        print(f"Confusion: {detailed['confusion']:.2f}")
        print(f"样本数量: {detailed['num_samples']}")

        for item in detailed['sample_details']:
            print(f"\n样本 {item['sample_index']}:")
            print(f"  DER: {item['der']:.2f}%")
            print(f"  total: {item['total']:.2f}")
            print(f"  false_alarm: {item['false_alarm']:.2f}")
            print(f"  missed_detection: {item['missed_detection']:.2f}")
            print(f"  confusion: {item['confusion']:.2f}")
            print(f"  预测片段数: {item['num_segments_pred']}")
            print(f"  参考片段数: {item['num_segments_ref']}")