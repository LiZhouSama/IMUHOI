import argparse
import os
import glob
from typing import List, Optional, Tuple, Dict

import torch
import numpy as np


PREFERRED_KEYS: Tuple[str, ...] = (
	"rotation_local_full_gt_list",
	"position_global_full_gt_world",
	"rotation_global",
	"obj_trans",
	"trans",
)


def _get_seq_len(data: Dict) -> Optional[int]:
	for key in PREFERRED_KEYS:
		if key in data and data[key] is not None:
			val = data[key]
			if isinstance(val, (torch.Tensor, np.ndarray)) and val.ndim >= 1:
				return int(val.shape[0])
	return None


def analyze_dir(root: str, ext: str = ".pt") -> List[Tuple[str, List[int]]]:
	results: List[Tuple[str, List[int]]] = []
	# 仅统计第一层子目录；若根目录下也有 pt 文件，可额外统计
	subdirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
	# 如果根目录本身也有 pt 文件，作为一个分组 "_root"
	root_pts = sorted(glob.glob(os.path.join(root, f"*{ext}")))
	if root_pts:
		subdirs = [root] + subdirs

	for d in subdirs:
		group_name = os.path.basename(d) if d != root else "_root"
		pt_files = sorted(glob.glob(os.path.join(d, f"*{ext}")))
		lens: List[int] = []
		for path in pt_files:
			try:
				data = torch.load(path, map_location="cpu")
				T = _get_seq_len(data)
				if T is not None:
					lens.append(T)
			except Exception:
				# 忽略损坏/不兼容文件
				pass
		results.append((group_name, lens))
	return results


def summarize(lengths: List[int]) -> Tuple[Optional[float], Optional[int], Optional[int]]:
	if not lengths:
		return None, None, None
	arr = np.asarray(lengths, dtype=np.int64)
	return float(arr.mean()), int(arr.min()), int(arr.max())


def main():
	parser = argparse.ArgumentParser(description="统计 OMOMO 处理数据中各子文件夹的序列长度(均值/最小/最大)")
	parser.add_argument(
		"--root",
		type=str,
		default=os.path.join("process", "processed_data_OMOMO"),
		help="根目录，包含若干子文件夹",
	)
	parser.add_argument(
		"--ext",
		type=str,
		default=".pt",
		help="文件后缀，默认 .pt",
	)
	parser.add_argument(
		"--format",
		type=str,
		choices=["csv", "table"],
		default="table",
		help="输出格式：csv 或 table",
	)
	args = parser.parse_args()

	if not os.path.isdir(args.root):
		raise FileNotFoundError(f"目录不存在: {args.root}")

	stats = analyze_dir(args.root, args.ext)

	all_lengths: List[int] = []

	if args.format == "csv":
		print("group,num_files,mean_len,min_len,max_len")
		for group, lens in stats:
			mean_v, min_v, max_v = summarize(lens)
			num_files = len(lens)
			def f(x):
				return f"{x:.2f}" if isinstance(x, float) else (str(x) if x is not None else "NA")
			print(f"{group},{num_files},{f(mean_v)},{f(min_v)},{f(max_v)}")
			all_lengths.extend(lens)
		# overall
		mean_v, min_v, max_v = summarize(all_lengths)
		print(f"OVERALL,{len(all_lengths)},{f(mean_v)},{f(min_v)},{f(max_v)}")
	else:
		# 简单表格输出
		header = f"{'GROUP':20s} {'#FILES':>8s} {'MEAN':>10s} {'MIN':>8s} {'MAX':>8s}"
		print(header)
		print("-" * len(header))
		for group, lens in stats:
			mean_v, min_v, max_v = summarize(lens)
			num_files = len(lens)
			def f(x):
				return f"{x:.2f}" if isinstance(x, float) else (str(x) if x is not None else "NA")
			print(f"{group:20s} {num_files:8d} {f(mean_v):>10s} {f(min_v):>8s} {f(max_v):>8s}")
			all_lengths.extend(lens)
		mean_v, min_v, max_v = summarize(all_lengths)
		print("-" * len(header))
		print(f"{'OVERALL':20s} {len(all_lengths):8d} {f(mean_v):>10s} {f(min_v):>8s} {f(max_v):>8s}")


if __name__ == "__main__":
	main()





