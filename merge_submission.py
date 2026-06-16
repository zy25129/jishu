"""
合并正演和反演的 submission.csv，还原为原始 test.csv 格式
正演输出: yindex, Xindex, Height, R1-R100 → 补 resLayer1-100 = 9999
反演输出: yindex, Xindex, Height, resLayer1-100 → 补 R1-100 = 9999
"""
import pandas as pd

FORWARD_PATH = 'results/forward/LSTM/submission.csv'
INVERSE_PATH = 'results/inverse/LSTM/submission.csv'
OUTPUT_PATH = 'submission.csv'

# 加载
forward_df = pd.read_csv(FORWARD_PATH)
inverse_df = pd.read_csv(INVERSE_PATH)

# 统一列顺序: yindex, Xindex, Height, resLayer1-100, R1-100
res_cols = [f'resLayer{i}' for i in range(1, 101)]
r_cols = [f'R{i}' for i in range(1, 101)]

# 正演: 缺少 resLayer 列，用 9999 填充
res_placeholder = pd.DataFrame(9999, index=forward_df.index, columns=res_cols)
forward_df = pd.concat([forward_df, res_placeholder], axis=1)

# 反演: 缺少 R 列，用 9999 填充
r_placeholder = pd.DataFrame(9999, index=inverse_df.index, columns=r_cols)
inverse_df = pd.concat([inverse_df, r_placeholder], axis=1)

target_cols = ['yindex', 'Xindex', 'Height'] + res_cols + r_cols

forward_df = forward_df[target_cols]
inverse_df = inverse_df[target_cols]

# 合并
merged = pd.concat([forward_df, inverse_df], ignore_index=True)
merged.to_csv(OUTPUT_PATH, index=False)
print(f"正演: {len(forward_df)} 条")
print(f"反演: {len(inverse_df)} 条")
print(f"合并: {len(merged)} 条 -> {OUTPUT_PATH}")
