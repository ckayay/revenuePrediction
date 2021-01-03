import pandas as pd

from korr import pearson, corrgram, corr_vs_pval, spearman, kendall

df = pd.read_csv("data/olist_order_metrics_dataset.csv")
df.drop(['order_purchase_timestamp'], axis=1, inplace=True)
df = df[df['total_order_value'] != 0]
rp, pp = pearson(df.values)  # parametric correlation where data is normally distributed & linear relationship
rs, ps = spearman(
    df.values)  # non-parametric correlation with data is not normally distributed & non linear relationship
rk, pk = kendall(df.values)  # same like spearman bu alternative to it

print("sample correlation pearson\n", rp)
print("\np-values pearson\n", pp)

print("sample correlation spearman\n", rs)
print("\np-values spearman spearman\n", ps)

print("sample correlation kendall\n", rk)
print("\np-values spearman kendall\n", pk)

print("\nsubstantial corr pearson?\n", pp < 0.05)
print("\nsubstantial corr spearman?\n", ps < 0.05)
print("\nsubstantial corr kendall ?\n", pk < 0.05)

# corr_vs_pval(rho, pval, plim=0.0001, dpi=120)
# corrgram(rho, pval)

