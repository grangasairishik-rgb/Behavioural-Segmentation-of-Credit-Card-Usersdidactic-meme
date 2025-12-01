# robust_customer_clustering.py
import os
import glob
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ---------- Helper functions ----------
def find_excel_file(pref_name="project2.1"):
    # try exact common names, then search
    candidates = [
        pref_name + ".xlsx",
        pref_name + ".xls",
        pref_name + ".xlxs",  # user typo
        pref_name + ".csv"
    ]
    for c in candidates:
        if os.path.exists(c):
            print(f"Found file: {c}")
            return c
    # fallback: find any excel/csv in cwd with pref_name in filename
    for ext in ("*.xlsx","*.xls","*.csv"):
        for path in glob.glob(ext):
            if pref_name in os.path.basename(path):
                print(f"Found file by pattern: {path}")
                return path
    # final fallback: take the first excel/csv in folder
    for ext in ("*.xlsx","*.xls","*.csv"):
        files = glob.glob(ext)
        if files:
            print(f"No exact match. Using first found file: {files[0]}")
            return files[0]
    raise FileNotFoundError("No data file found in current folder. Put your file in the project folder.")

def load_data(path):
    print(f"Loading data from: {path}")
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    print(f"Shape: {df.shape}")
    return df

def prepare_features(df, drop_cols=None, max_onehot_card=10):
    # drop columns user doesn't want (if provided)
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    # remove id-like columns automatically (common patterns)
    auto_drop = [c for c in df.columns if c.lower() in ("id","customer_id","cust_id","index")]
    if auto_drop:
        print("Dropping id-like columns:", auto_drop)
        df = df.drop(columns=auto_drop, errors='ignore')
    # separate numeric and categorical
    num = df.select_dtypes(include=[np.number]).copy()
    cat = df.select_dtypes(exclude=[np.number]).copy()
    print(f"Numeric cols: {list(num.columns)}")
    print(f"Categorical cols: {list(cat.columns)}")
    # impute numeric with median
    if not num.empty:
        num = num.fillna(num.median())
    # encode categorical
    enc_frames = []
    for c in cat.columns:
        nunique = cat[c].nunique(dropna=True)
        if nunique == 0:
            continue
        if nunique <= max_onehot_card:
            # one-hot
            dummies = pd.get_dummies(cat[c].fillna("MISSING"), prefix=c, drop_first=True)
            enc_frames.append(dummies)
        else:
            # frequency encoding
            freq = cat[c].fillna("MISSING").value_counts(normalize=True).to_dict()
            enc = cat[c].fillna("MISSING").map(freq).rename(c + "_freq")
            enc_frames.append(enc.to_frame())
    if enc_frames:
        enc_df = pd.concat(enc_frames, axis=1)
    else:
        enc_df = pd.DataFrame(index=df.index)
    # combine
    final = pd.concat([num.reset_index(drop=True), enc_df.reset_index(drop=True)], axis=1)
    # if still empty -> raise
    if final.shape[1] == 0:
        raise ValueError("No features available for clustering after preprocessing. Check your data.")
    return final

def scale_features(X_df):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)
    return X_scaled, scaler

def compute_elbow_silhouette(X, k_min=2, k_max=10, random_state=42):
    ks = []
    inertias = []
    sil_scores = []
    n = X.shape[0]
    upper = min(k_max, max(2, n-1))
    for k in range(k_min, upper+1):
        km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        # silhouette valid only if k>1 and less than n
        try:
            sil = silhouette_score(X, labels) if (k > 1 and k < n) else np.nan
        except Exception:
            sil = np.nan
        sil_scores.append(sil)
        ks.append(k)
    return ks, inertias, sil_scores

def plot_elbow_sil(ks, inertias, sil_scores, out_prefix="plot"):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(ks, inertias, marker='o')
    plt.title("Elbow (K vs Inertia)")
    plt.xlabel("K")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(ks, sil_scores, marker='o')
    plt.title("Average Silhouette Score by K")
    plt.xlabel("K")
    plt.ylabel("Silhouette")
    plt.grid(True)
    plt.tight_layout()
    fname = out_prefix + "_elbow_silhouette.png"
    plt.savefig(fname, dpi=150)
    print("Saved plot:", fname)
    plt.show()

def plot_pca_scatter(X_scaled, labels, out_prefix="plot"):
    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(X_scaled)
    plt.figure(figsize=(7,6))
    unique = np.unique(labels)
    for lbl in unique:
        idx = labels == lbl
        plt.scatter(proj[idx,0], proj[idx,1], label=f"C{lbl}", s=40, alpha=0.6)
    # centroids in PCA space
    centroids = np.array([proj[labels==lbl].mean(axis=0) for lbl in unique])
    plt.scatter(centroids[:,0], centroids[:,1], marker='X', s=120, c='k', label='centroid')
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.title("Clusters (PCA 2D projection)")
    plt.legend()
    fname = out_prefix + "_pca_scatter.png"
    plt.savefig(fname, dpi=150)
    print("Saved plot:", fname)
    plt.show()
    return proj, pca

def plot_silhouette_detailed(X, labels, n_clusters, out_prefix="plot"):
    sil_vals = silhouette_samples(X, labels)
    y_lower = 10
    plt.figure(figsize=(8,6))
    for i in range(n_clusters):
        ith_sil = np.sort(sil_vals[labels == i])
        size = ith_sil.shape[0]
        y_upper = y_lower + size
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_sil)
        plt.text(-0.05, y_lower + 0.5*size, str(i))
        y_lower = y_upper + 10
    plt.axvline(x=np.mean(sil_vals), color="red", linestyle="--")
    plt.title(f"Silhouette plot for K={n_clusters}")
    plt.xlabel("Silhouette coefficient")
    plt.ylabel("Cluster")
    fname = out_prefix + f"_silhouette_k{n_clusters}.png"
    plt.savefig(fname, dpi=150)
    print("Saved plot:", fname)
    plt.show()

# ---------- Main pipeline ----------
def main(preferred_name="project2.1"):
    file_path = find_excel_file(preferred_name)
    df = load_data(file_path)

    # quick diagnostics
    print("\nColumn types summary:")
    print(df.dtypes.value_counts())

    # prepare features
    X_df = prepare_features(df)
    print("Prepared feature matrix with shape:", X_df.shape)

    # scale
    X_scaled, scaler = scale_features(X_df)

    # Elbow + silhouette
    ks, inertias, sil_scores = compute_elbow_silhouette(X_scaled, k_min=2, k_max=10)
    print("Ks:", ks)
    print("Inertias:", inertias)
    print("Silhouettes:", sil_scores)
    plot_elbow_sil(ks, inertias, sil_scores, out_prefix="results")

    # choose candidate K: top silhouette peaks (up to 3) and elbow heuristic
    sil_arr = np.array([s if not np.isnan(s) else -1 for s in sil_scores])
    top_idxs = sil_arr.argsort()[::-1][:3]
    cand_k = sorted({ks[i] for i in top_idxs})
    # elbow heuristic: where relative drop reduces significantly
    rel_drop = np.diff(inertias) / np.array(inertias)[:-1]
    for i, rd in enumerate(rel_drop):
        if abs(rd) < 0.15:
            cand_k.append(ks[i+1])
    cand_k = sorted(set(cand_k))
    print("Candidate K values (from silhouette peaks + elbow heuristic):", cand_k)

    # pick final K - prefer highest silhouette among candidates
    best_k = cand_k[0] if cand_k else ks[np.argmax(sil_arr)]
    if len(cand_k)>0:
        best_k = cand_k[np.argmax([sil_arr[ks.index(k)] if k in ks else -1 for k in cand_k])]
    print("Selected K =", best_k)

    # final clustering
    kmeans = KMeans(n_clusters=best_k, n_init=30, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    df_out = df.copy()
    df_out["Cluster"] = labels
    df_out.to_csv("data_with_clusters.csv", index=False)
    print("Saved labeled data -> data_with_clusters.csv")

    # PCA scatter
    proj, pca = plot_pca_scatter(X_scaled, labels, out_prefix="results")

    # silhouette detailed
    try:
        if best_k > 1:
            plot_silhouette_detailed(X_scaled, labels, best_k, out_prefix="results")
    except Exception as e:
        print("Could not produce silhouette detailed plot:", e)

    # cluster profiling: use original features (raw), but include counts and mean of numeric cols
    profile_num = df_out.select_dtypes(include=[np.number]).groupby(df_out['Cluster']).agg(['mean','median','std','count'])
    # flatten columns
    profile_num.columns = ['_'.join(col).strip() for col in profile_num.columns.values]
    # percentage of total
    sizes = df_out['Cluster'].value_counts().sort_index()
    profile_summary = profile_num.copy()
    profile_summary['Cluster_size'] = sizes.values
    profile_summary['Cluster_pct'] = (sizes.values / len(df_out) * 100).round(2)
    profile_summary.to_csv("cluster_profile.csv")
    print("Saved cluster profile -> cluster_profile.csv")
    print("\nDone. Outputs: data_with_clusters.csv, cluster_profile.csv, results_elbow_silhouette.png, results_pca_scatter.png, results_silhouette_k*.png")

if __name__ == "__main__":
    main("project2.1")
