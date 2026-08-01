# LGSPSO_ML_kfold

LGSPSO（遺傳演算法+粒子群優化）結合 k-fold 交叉驗證的機器學習工具。

---

## 程式功能說明

| 腳本 | 功能 |
|---|---|
| `LGSPSO_kfold_SVR.py` | LGSPSO 優化 SVR |
| `MLP_adam.py` | MLP 神經網絡 |
| `submit_ML.sh` | Slurm 提交 |

## 依賴環境

| 項目 | 需求 |
|---|---|
| Python | 3.x + numpy, scikit-learn |

## AI Agent 操控指南

```
任務: SVR 超參數優化
步驟:
1. python LGSPSO_kfold_SVR.py 執行優化
2. 或使用 sbatch submit_ML.sh
```
