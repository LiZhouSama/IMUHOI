import argparse
import os
from typing import Dict, Tuple, Optional

import torch


def _to_1d_bool(t: torch.Tensor) -> Optional[torch.Tensor]:
	"""
	将输入 tensor 规整为一维 bool 序列 (T,)。
	若输入为 None 或元素个数为 0，则返回 None。
	支持以下形状：
	- (T,)
	- (1, T) / (T, 1)
	- 任何可以 squeeze 到一维的形状，只要最终 dim == 1
	"""
	if t is None:
		return None
	if not isinstance(t, torch.Tensor):
		return None
	if t.numel() == 0:
		return None
	# 尝试转为 bool，再 squeeze
	try:
		bt = t.bool().squeeze()
	except Exception:
		return None
	if bt.ndim == 1:
		return bt
	# 有时是 (1, T) 或 (T, 1) squeeze 后仍可能是二维 (例如全 1 维度未被 squeeze)
	# 再次尝试在可能情况下选取长度更长的维度
	if bt.ndim == 2 and 1 in bt.shape:
		bt = bt.reshape(-1)
		return bt
	return None


def compute_ratio(seq: torch.Tensor) -> Optional[float]:
	seq_1d = _to_1d_bool(seq)
	if seq_1d is None:
		return None
	T = int(seq_1d.shape[0])
	if T == 0:
		return None
	pos = int(seq_1d.sum().item())
	return pos / float(T)


def analyze_file(path: str) -> Dict[str, Optional[float]]:
	data = torch.load(path, map_location="cpu")

	def get(key: str) -> Optional[torch.Tensor]:
		return data.get(key, None)

	res = {
		"lhand_contact": compute_ratio(get("lhand_contact")),
		"rhand_contact": compute_ratio(get("rhand_contact")),
		"obj_contact": compute_ratio(get("obj_contact")),
	}
	return res


def main():
	parser = argparse.ArgumentParser(
		description="统计 BEHAVE 训练集序列中手/物体接触标签占比"
	)
	parser.add_argument(
		"--root",
		type=str,
		default=os.path.join("process", "processed_data_OMOMO", "train"),
		help="包含 .pt 序列文件的目录 (默认: process/processed_data_BEHAVE/train)",
	)
	parser.add_argument(
		"--ext",
		type=str,
		default=".pt",
		help="文件后缀 (默认: .pt)",
	)
	args = parser.parse_args()

	root = args.root
	ext = args.ext
	if not os.path.isdir(root):
		raise FileNotFoundError(f"目录不存在: {root}")

	files = [
		os.path.join(root, f)
		for f in sorted(os.listdir(root))
		if f.endswith(ext)
	]
	if not files:
		print("未找到任何 .pt 文件。")
		return

	# 汇总统计
	count = 0
	acc = {"lhand_contact": 0.0, "rhand_contact": 0.0, "obj_contact": 0.0}
	valid = {"lhand_contact": 0, "rhand_contact": 0, "obj_contact": 0}

	print("file,lhand_ratio,rhand_ratio,obj_ratio")
	for fp in files:
		try:
			res = analyze_file(fp)
		except Exception as e:
			print(f"{os.path.basename(fp)},ERROR:{str(e)}")
			continue

		l = res["lhand_contact"]
		r = res["rhand_contact"]
		o = res["obj_contact"]

		def _fmt(x: Optional[float]) -> str:
			return f"{x:.6f}" if isinstance(x, float) else "NA"

		# print(f"{os.path.basename(fp)},{_fmt(l)},{_fmt(r)},{_fmt(o)}")
		count += 1
		if isinstance(l, float):
			acc["lhand_contact"] += l
			valid["lhand_contact"] += 1
		if isinstance(r, float):
			acc["rhand_contact"] += r
			valid["rhand_contact"] += 1
		if isinstance(o, float):
			acc["obj_contact"] += o
			valid["obj_contact"] += 1

	# 打印均值
	def _mean(key: str) -> str:
		v = valid[key]
		return f"{(acc[key] / v):.6f}" if v > 0 else "NA"

	print("ALL_MEAN,", end="")
	print(
		"{}{}{}".format(
			_mean("lhand_contact"),
			"," + _mean("rhand_contact"),
			"," + _mean("obj_contact"),
		)
	)


if __name__ == "__main__":
	main()


