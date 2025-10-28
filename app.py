# app_gene_clustering.py
# ------------------------------------------------------------
# Clusterização de expressão gênica com PyCaret + Streamlit (v3.x)
# Base padrão: data/gene_expression.csv (coloque aí o arquivo do Kaggle)
# Melhorias integradas: detecção/transposição automática, UMAP para visualização,
# top genes por ANOVA e caching no carregamento.
# ------------------------------------------------------------

import os
from typing import List, Optional

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
from scipy.cluster.hierarchy import linkage, dendrogram

# PyCaret
from pycaret.clustering import setup, create_model, assign_model, plot_model, save_model

# scikit-learn
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.feature_selection import f_classif

# UMAP (opcional)
try:
    import umap
except Exception:
    umap = None

# Streamlit caching
@st.cache_data
def load_csv(path, sep=',', index_col=None):
    return pd.read_csv(path, sep=sep, index_col=index_col)

# Página
st.set_page_config(page_title="Clustering: Gene Expression (Bioinformatics)", layout="wide")
st.title("Clusterização de Expressão Gênica — PyCaret + Streamlit")

st.markdown("Esse app faz uma exploração dos dados e agrupa amostras parecidas em 'clusters'. Primeiro mostramos resumos simples (média, variação) e gráficos para entender como os dados se comportam. Depois reduzimos para dois eixos para ver visualmente se há grupos. Em seguida rodamos vários algoritmos e comparamos usando medidas de qualidade (como a Silhouette). Finalmente mostramos quais variáveis mais diferenciam os grupos (heatmaps, ANOVA e um classificador proxy). Em resumo: olhamos os dados, geramos grupos e apresentamos por que esses grupos fazem sentido. O dataset escolhido e usado como exemplo ao abrir o app é o Gene Expression - Bioinformatics(https://www.kaggle.com/datasets/samira1992/gene-expression-bioinformatics-dataset).")

st.title("Sobre o dataset")

st.markdown("""Este conjunto de dados mostra como milhares de genes “se comportam” ao longo do tempo.
Cada linha representa um gene, e cada coluna mostra o quanto esse gene estava “ativo” em um determinado momento.

No total, são 4.381 genes medidos em 23 momentos diferentes.
Esses valores indicam o nível de atividade de cada gene, o que ajuda pesquisadores a descobrir padrões, como por exemplo:

Genes que aumentam ou diminuem sua atividade juntos.

Momentos em que acontecem mudanças importantes no funcionamento das células.

Grupos de genes que podem estar ligados a um mesmo processo biológico.

Em resumo, o dataset é como um “gráfico gigante” mostrando como os genes mudam com o tempo, e serve para fazer análises, agrupamentos e visualizações que ajudam a entender melhor o comportamento biológico das células.""")

# -------------------- Utilitários --------------------

# substitua a definição antiga por esta
def safe_numeric_df(df: pd.DataFrame, cols: Optional[List[str]] = None, cols_num: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Retorna um DataFrame apenas com colunas numéricas úteis (nuniques > 1).
    Aceita tanto o parâmetro `cols` quanto `cols_num` (compatibilidade).
    """
    # aceita qualquer um dos dois nomes de argumento; prioridade para cols, depois cols_num
    use_cols = cols if cols is not None else cols_num

    if use_cols:
        # filtra apenas as colunas que existem no df
        valid = [c for c in use_cols if c in df.columns]
        num_df = df[valid].copy()
        # ainda garante que sejam numéricas (converte se possível)
        num_df = num_df.select_dtypes(include="number").copy()
    else:
        num_df = df.select_dtypes(include="number").copy()

    # remove colunas constantes (nunique <= 1)
    nunique = num_df.nunique(dropna=False)
    keep = nunique[nunique > 1].index.tolist()
    return num_df[keep]

def compute_metrics(X: pd.DataFrame, labels: pd.Series) -> tuple:
    if len(np.unique(labels)) <= 1:
        return (np.nan, np.nan, np.nan)
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)
    return (sil, ch, db)


def interpret_row(model_name: str, sil: float, ch: float, db: float) -> str:
    if np.isnan(sil):
        return f"{model_name}: não conseguiu formar múltiplos clusters ou falhou na avaliação."
    parts = []
    if sil > 0.5:
        parts.append(f"Silhouette = {sil:.2f} (boa separação)")
    elif sil > 0.25:
        parts.append(f"Silhouette = {sil:.2f} (moderada, pode melhorar)")
    else:
        parts.append(f"Silhouette = {sil:.2f} (clusters fracos/ruído)")
    parts.append(f"CH = {ch:.1f} (quanto maior, melhor)")
    if db < 0.5:
        parts.append(f"DB = {db:.2f} (excelente compactação)")
    elif db < 1.0:
        parts.append(f"DB = {db:.2f} (bom resultado)")
    else:
        parts.append(f"DB = {db:.2f} (sobreposição entre clusters)")
    return f"{model_name} → " + "; ".join(parts) + "."


def cluster_profiles_table(labeled: pd.DataFrame, cluster_col: str = "Cluster") -> pd.DataFrame:
    num_cols = labeled.select_dtypes(include="number").columns.tolist()
    num_cols = [c for c in num_cols if c != cluster_col]
    if not num_cols:
        return pd.DataFrame()
    prof = labeled.groupby(cluster_col)[num_cols].agg(["mean", "median", "std", "min", "max"])
    return prof


def make_dendrogram(X: pd.DataFrame, sample_cap: int = 250, method: str = "ward"):
    if len(X) > sample_cap:
        X_plot = X.sample(n=sample_cap, random_state=42)
        st.caption(f"Dendrograma com amostra de {sample_cap} linhas (de {len(X)}) para desempenho.")
    else:
        X_plot = X
    Z = linkage(X_plot.values, method=method)
    fig, ax = plt.subplots(figsize=(10, 4))
    dendrogram(Z, truncate_mode="level", p=5, no_labels=True, color_threshold=None, ax=ax)
    ax.set_title(f"Dendrograma (método: {method})")
    ax.set_ylabel("Distância de ligação")
    st.pyplot(fig)


def plot_cluster_means_heatmap(labeled: pd.DataFrame, cluster_col: str = "Cluster",
                               zscore: bool = True, top_n_features: Optional[int] = None):
    num_cols = labeled.select_dtypes(include="number").columns.tolist()
    num_cols = [c for c in num_cols if c != cluster_col]
    if not num_cols:
        st.info("Sem colunas numéricas para heatmap.")
        return
    means = labeled.groupby(cluster_col)[num_cols].mean()
    if top_n_features is not None and top_n_features > 0 and top_n_features < len(num_cols):
        var_between = means.var(axis=0)
        selected = var_between.sort_values(ascending=False).head(top_n_features).index
        means = means[selected]
    data_plot = means.copy()
    title_suffix = ""
    if zscore:
        data_plot = (means - means.mean(axis=0)) / (means.std(axis=0).replace(0, np.nan))
        title_suffix = " (z-score)"
        data_plot = data_plot.fillna(0.0)
    def _fmt_cluster_label(v):
        if isinstance(v, (int, np.integer)):
            return f"Cluster {int(v)}"
        if isinstance(v, (float, np.floating)) and float(v).is_integer():
            return f"Cluster {int(v)}"
        s = str(v)
        return s if s.lower().startswith("cluster") else f"{s}"
    y_labels = [_fmt_cluster_label(i) for i in data_plot.index]
    fig = px.imshow(
        data_plot,
        x=[str(c) for c in data_plot.columns],
        y=y_labels,
        color_continuous_midpoint=0.0 if zscore else None,
        aspect="auto",
        labels=dict(color="intensidade")
    )
    fig.update_layout(title=f"Heatmap das médias por cluster{title_suffix}", height=500)
    st.plotly_chart(fig, use_container_width=True)

# -------------------- Fonte de Dados --------------------
st.sidebar.header("Fonte de Dados")
fonte = st.sidebar.radio("Selecione a fonte", ["Gene Expression (Exemplo)", "Upload CSV"])

if fonte == "Gene Expression (Exemplo)":
    # tenta carregar automaticamente o dataset local
    caminho_csv = "C:/Users/gabri/OneDrive/Área de Trabalho/Faculdade/1 bimestre/Matemática computacional/Unsupervised_Learning_ML/dataset/Spellman.csv"
    caminho_tsv = "C:/Users/gabri/OneDrive/Área de Trabalho/Faculdade/1 bimestre/Matemática computacional/Unsupervised_Learning_ML/dataset/Spellman.csv"

    if os.path.exists(caminho_csv):
        df = pd.read_csv(caminho_csv)
        st.write(f"Usando dataset local: `{caminho_csv}` ({df.shape[0]} linhas, {df.shape[1]} colunas)")
        st.dataframe(df.head())
    elif os.path.exists(caminho_tsv):
        df = pd.read_csv(caminho_tsv, sep="\t")
        st.write(f"Usando dataset local: `{caminho_tsv}` ({df.shape[0]} linhas, {df.shape[1]} colunas)")
        st.dataframe(df.head())
    else:
        st.warning("Nenhum arquivo encontrado em `dataset\\Spellman.csv` ou `.tsv`.")
        st.stop()
else:
    # permite upload de CSV pelo usuário
    file = st.sidebar.file_uploader("Carregue seu CSV", type=["csv", "tsv"])
    if file:
        if str(file.name).lower().endswith(".tsv"):
            df = pd.read_csv(file, sep="\t")
        else:
            df = pd.read_csv(file)
        st.write("Prévia dos dados carregados:")
        st.dataframe(df.head())
    else:
        st.stop()


# A — Heurística de transpose sugerida
def suggest_transpose(df: pd.DataFrame, threshold_ratio: float = 3.0) -> bool:
    if df is None:
        return False
    n_rows, n_cols = df.shape
    return (n_cols / max(1, n_rows)) > threshold_ratio

if 'df' in locals():
    if suggest_transpose(df):
        if st.sidebar.button("Detectei colunas >> linhas: transpor automaticamente"):
            df = df.transpose()
            st.info("Dados transpostos automaticamente (linhas = amostras, colunas = genes).")
            st.dataframe(df.head())

# F — warning para datasets grandes
MAX_FEATURES = 5000
if df is not None and df.shape[1] > MAX_FEATURES:
    st.warning(f"O dataset tem {df.shape[1]} features — isso pode ser pesado. Use PCA/UMAP ou limite as features para performance.")

# seleção de features
st.sidebar.header("Configuração das Features")
cols_num = st.sidebar.multiselect("Selecione features numéricas (se vazio usamos automaticamente)", df.columns.tolist())
if not cols_num:
    cols_num = df.select_dtypes(include="number").columns.tolist()
    st.sidebar.info("Usando automaticamente colunas numéricas não constantes.")

# pré-processamento
st.sidebar.header("Pré-processamento")
normalize = st.sidebar.checkbox("Normalizar", value=True)
pca = st.sidebar.checkbox("Aplicar PCA", value=True)
pca_comp = st.sidebar.slider("Componentes PCA", 2, 50, 10, disabled=not pca)

# adicionar opção de DR (A/B)
st.sidebar.header("Visualização 2D")
dr_method = st.sidebar.selectbox("Redução para visualização (2D)", ["PCA", "UMAP", "t-SNE"], index=0)

# parâmetros de cluster
st.sidebar.header("Parâmetros de cluster")
k_clusters = st.sidebar.number_input("Número de clusters (k) (0 = auto)", min_value=0, value=0, step=1)
dbscan_eps = st.sidebar.slider("DBSCAN eps", min_value=0.05, max_value=5.0, value=0.5, step=0.05)
dbscan_min_samples = st.sidebar.number_input("DBSCAN min_samples", min_value=1, value=5, step=1)
optics_min_samples = st.sidebar.number_input("OPTICS min_samples", min_value=1, value=5, step=1)

st.sidebar.header("Desempenho")
limit_rows = st.sidebar.number_input("Limitar amostras (0 = sem limite)", min_value=0, value=0, step=100)
models_to_try = st.sidebar.multiselect(
    "Algoritmos a testar",
    ["kmeans", "hclust", "dbscan", "optics", "birch", "spectral"],
    default=["kmeans", "hclust", "dbscan"]
)

st.sidebar.header("Exibição avançada")
show_dendro_anyway = st.sidebar.checkbox("Mostrar dendrograma mesmo se não for hclust", value=False)
heatmap_zscore = st.sidebar.checkbox("Heatmap com z-score (recomendado)", value=True)
heatmap_topn = st.sidebar.number_input("Heatmap: limitar às top-N variáveis (0 = todas)", min_value=0, value=20, step=1)

if "objetos" not in st.session_state:
    st.session_state["objetos"] = {}

if "res_df" not in st.session_state:
    st.session_state["res_df"] = pd.DataFrame(
        columns=["Modelo", "Silhouette", "Calinski-Harabasz", "Davies-Bouldin"]
    )

# -------------------- EDA (Exploratory Data Analysis) --------------------
st.header("EDA — Análise Exploratória de Dados")

# pequena função utilitária
def ensure_samples_by_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garante que linhas = amostras. Nosso dataset de expressão pode vir com linhas = genes.
    Heurística: se maioria das colunas for numérica e n_cols >> n_rows, talvez precise transpor.
    """
    if df is None:
        return df
    n_rows, n_cols = df.shape
    num_cols = df.select_dtypes(include="number").shape[1]
    # se houver muitos mais colunas numéricas que linhas, sugerir transpor
    if n_cols > n_rows and (n_cols / max(1, n_rows)) > 2.5:
        if st.checkbox("Detectei formato genes x amostras (transpor para amostras x features)? (marque para transpor)", value=False, key="eda_transpose"):
            df = df.transpose()
            st.info("Dados transpostos para que cada linha represente uma amostra.")
    return df

# aplica verificação (não sobrescreve df original sem confirmação)
df = ensure_samples_by_rows(df)

# selecionar automaticamente colunas numéricas seguras para EDA
num_df_full = safe_numeric_df(df, cols_num=None)
st.subheader("Resumo rápido dos dados")
st.markdown(
    "Visão inicial: número de amostras (linhas) e features (colunas numéricas). "
    "Se houver features sem variância ou não-numéricas, elas serão ignoradas nos gráficos."
)
st.metric("Amostras (linhas)", f"{df.shape[0] if df is not None else 0}")
st.metric("Features numéricas", f"{num_df_full.shape[1] if df is not None else 0}")

# abas para organizar EDA
tab1, tab2, tab3, tab4 = st.tabs(["Resumo & Estatísticas", "Distribuições & Boxplots", "Correlação / Heatmap", "Projeções (PCA)"])

# ---------- TAB 1: Estatísticas Descritivas ----------
with tab1:
    st.markdown("#### Estatísticas descritivas (features numéricas selecionadas)")
    if num_df_full.shape[1] == 0:
        st.warning("Nenhuma coluna numérica disponível para EDA.")
    else:
        desc = num_df_full.describe().T
        desc["missing_ratio"] = (num_df_full.isna().sum() / num_df_full.shape[0]).round(3)
        desc = desc.rename(columns={"50%": "median"})
        st.markdown("""Esta parte mostra um resumo numérico dos dados.
Cada coluna representa uma medida (ou “gene”) e o sistema calcula valores como:

Média: valor médio de expressão dos genes.

Mediana: valor do meio — ajuda a ver se os dados estão equilibrados.

Desvio padrão (std): mostra quanto os valores variam (se são estáveis ou mudam muito).

Valores ausentes (missing): indica se há dados faltando.

💡 Por que isso importa:
Essas medidas ajudam a entender se o conjunto de dados está “limpo” e se há genes que variam bastante (esses geralmente são mais interessantes para análises e agrupamentos).""")
        st.dataframe(desc.style.format({
            "mean": "{:.4f}", "std": "{:.4f}", "min": "{:.4f}", "max": "{:.4f}", "missing_ratio": "{:.3f}"
        }), use_container_width=True)
# ---------- TAB 2: Histogramas e Boxplots ----------
with tab2:
    st.markdown("#### Distribuição global dos valores")
    st.markdown("Aqui temos duas formas de visualizar os valores dos genes:")
    st.markdown("""
                🔹 Histograma

                Mostra quantas vezes certos valores aparecem no dataset.
                Se a maior parte dos valores está perto de zero, significa que muitos genes têm baixa expressão.
                Se houver “caudas” longas, quer dizer que alguns genes têm valores muito altos — podem ser casos especiais ou “outliers”

                Ou seja, se o gráfico mostra muitos valores perto de zero, isso quer dizer que a maioria dos genes está “calma” (baixa atividade).
                Mas se houver alguns valores bem altos (umas “barrinhas” distantes), isso indica que alguns genes estão muito ativos — o que pode ser interessante, pois esses genes podem estar fazendo algo importante..
                """)
    if num_df_full.shape[1] == 0:
        st.info("Sem dados numéricos.")
    else:
        # histograma global (todos os valores das features concatenados)
        hist_bins = st.slider("Bins do histograma", min_value=10, max_value=200, value=60, key="eda_bins")
        flat_vals = num_df_full.values.flatten()
        flat_vals = flat_vals[~np.isnan(flat_vals)]
        fig_hist = px.histogram(flat_vals, nbins=hist_bins, labels={"value": "valor de expressão"}, title="Distribuição global dos valores de expressão (todas as features)")
        fig_hist.update_layout(height=350)
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("#### Boxplots das features mais variáveis")
        st.markdown("""
                    🔹 Boxplot

                    O boxplot (gráfico de caixa) resume visualmente a variação dos valores.

                    Cada “caixinha” representa um gene, e mostra:

                    A linha do meio é o valor “normal” (a mediana).

                    O tamanho da caixa é o quanto os valores do gene variam.

                    Os pontos fora da caixa são os valores muito diferentes dos outros (os chamados outliers).""")
        top_n = st.number_input("Top-N features por variância (boxplots)", min_value=2, max_value=50, value=8, key="eda_topn")
        # calcula top features por variância
        var_series = num_df_full.var(axis=0).sort_values(ascending=False)
        top_feats = var_series.head(top_n).index.tolist()
        if not top_feats:
            st.info("Nenhuma feature para exibir.")
        else:
            sample_n = min(1000, num_df_full.shape[0])
            sample_df = num_df_full[top_feats].sample(n=sample_n, random_state=42) if num_df_full.shape[0] > sample_n else num_df_full[top_feats]
            # melt para boxplot agrupado
            df_melt = sample_df.reset_index().melt(id_vars=["index"], value_vars=top_feats, var_name="feature", value_name="value")
            fig_box = px.box(df_melt, x="feature", y="value", points="all", title=f"Boxplots — top {len(top_feats)} features por variância (sample)")
            fig_box.update_layout(xaxis_tickangle=-45, height=450)
            st.plotly_chart(fig_box, use_container_width=True)
            st.markdown("""
            💡 Por que isso importa:
            Se um gene tem uma caixa bem alta, significa que ele muda bastante ao longo do tempo (ou entre amostras).
            Isso é bom de saber, porque genes que mudam muito costumam estar ligados a processos biológicos importantes — eles “reagem” mais às mudanças no ambiente da célula.""")

# ---------- TAB 3: Correlação / Heatmap ----------
with tab3:
    st.markdown("#### Correlograma (matriz de correlação entre features)")
    if num_df_full.shape[1] < 2:
        st.info("Pelo menos duas features são necessárias para correlação.")
    else:
        max_corr_feats = st.number_input("Max features para correlograma (por amostragem)", min_value=5, max_value=200, value=50, key="eda_corr_max")
        if num_df_full.shape[1] > max_corr_feats:
            st.info(f"Selecionando top {max_corr_feats} features por variância para correlograma.")
            corr_feats = var_series.head(max_corr_feats).index.tolist()
        else:
            corr_feats = num_df_full.columns.tolist()
        corr_sample_n = min(500, num_df_full.shape[0])
        corr_df = num_df_full[corr_feats].sample(n=corr_sample_n, random_state=42) if num_df_full.shape[0] > corr_sample_n else num_df_full[corr_feats]
        corr_mat = corr_df.corr()
        st.markdown("""
                    Este gráfico mostra o quanto os genes estão relacionados entre si.
                    Cada célula colorida representa a força da relação entre dois genes:

                    Cores mais próximas de azul forte indicam correlação alta (os genes variam juntos).

                    Cores próximas do branco indicam pouca relação.

                    💡 Por que isso importa:
                    Genes muito parecidos podem estar trabalhando em conjunto (co-regulados) ou podem ser redundantes (mesma informação).
                    Entender essas relações ajuda a reduzir a dimensionalidade do dataset e evitar dados repetidos na análise.
                    """)
        fig_corr = px.imshow(corr_mat, labels=dict(x="features", y="features", color="pearson r"), title=f"Correlograma — {len(corr_feats)} features (sample {corr_sample_n})")
        fig_corr.update_layout(height=600)
        st.plotly_chart(fig_corr, use_container_width=True)
# ---------- TAB 4: Projeção 2D com PCA ----------  
with tab4:
    st.markdown("#### Projeção 2D — reduza a dimensionalidade para inspecionar agrupamentos globais")
    st.markdown("""
        🔸 PCA (Análise de Componentes Principais)

        O PCA tenta resumir o máximo possível da variação dos dados em poucos eixos.

        Ele “comprime” as 23 colunas em apenas 2 direções principais, mostrando quem é parecido ou diferente.

        Se dois genes aparecem perto no gráfico de PCA, significa que eles se comportam de forma parecida ao longo do tempo.
        Se estão longe, é porque têm padrões diferentes.
    """)

    n_components = 2
    sample_for_dr = st.number_input(
        "Amostras para projeção (0 = usar todas)", 
        min_value=0, max_value=2000, value=500, key="eda_dr_sample"
    )

    if num_df_full.shape[0] == 0:
        st.info("Sem dados para projeção.")
    else:
        X_dr = num_df_full.copy()
        if sample_for_dr and sample_for_dr > 0 and X_dr.shape[0] > sample_for_dr:
            X_dr = X_dr.sample(n=sample_for_dr, random_state=42)
        
        # padronizar (z-score)
        X_dr = (X_dr - X_dr.mean()) / X_dr.std(ddof=0).replace(0, np.nan)
        X_dr = X_dr.fillna(0.0).values

        try:
            from sklearn.decomposition import PCA
            pca_model = PCA(n_components=n_components)
            proj = pca_model.fit_transform(X_dr)
            variance_explained = pca_model.explained_variance_ratio_.cumsum()
            st.markdown(
                f"PCA — variância acumulada pelos 2 primeiros componentes: "
                f"{variance_explained[min(1, len(variance_explained)-1)]:.3f}"
            )

            proj_df = pd.DataFrame(proj, columns=["dim1", "dim2"])

            # usar coluna Cluster se existir
            if "Cluster" in df.columns:
                proj_df["cluster"] = df["Cluster"].astype(str).values[:len(proj_df)]

            fig_sc = px.scatter(
                proj_df,
                x="dim1",
                y="dim2",
                color="cluster" if "cluster" in proj_df.columns else None,
                title="Projeção 2D (PCA)",
                hover_data=[proj_df.index]
            )
            fig_sc.update_layout(height=500)
            st.plotly_chart(fig_sc, use_container_width=True)

            st.markdown(
                "Interpretação: pontos próximos no gráfico 2D tendem a ser amostras com perfis de expressão similares, "
                "porém a projeção perde informação — ideal para inspeção visual, não para métricas finais."
            )
        except Exception as e:
            st.warning(f"Falha ao gerar projeção 2D: {e}")
# Pequeno resumo / download dos dados processados para EDA
with st.expander("Exportar / baixar dados numéricos usados na EDA"):
    try:
        csv_eda = num_df_full.to_csv(index=True).encode("utf-8")
        st.download_button("Baixar CSV com features numéricas usadas na EDA", csv_eda, "eda_numeric_features.csv")
    except Exception as e:
        st.info(f"Não foi possível preparar download: {e}")

st.markdown("---")
st.markdown(
    "Depois de explorar os dados aqui, vá para a aba/parte de `Rodar Clusterização` para gerar modelos. "
    "Use os insights do EDA (ex.: top features, variância, correlações) para ajustar parâmetros de pré-processamento."
)
# -------------------- FIM DA EDA --------------------

# -------------------- Rodar pipeline --------------------
if st.button("Rodar Clusterização"):
    data_full = safe_numeric_df(df, cols_num)
    if data_full.shape[1] == 0:
        st.error("Nenhuma coluna numérica válida encontrada. Verifique seu arquivo ou selecione as colunas manualmente.")
        st.stop()

    if limit_rows and limit_rows > 0 and limit_rows < len(data_full):
        data = data_full.sample(n=limit_rows, random_state=42).reset_index(drop=True)
    else:
        data = data_full.copy()

    if data.shape[1] > MAX_FEATURES:
        st.warning("Muitas features; considere PCA antes de rodar todos os modelos para performance.")

    # Setup PyCaret
    setup(
        data=data,
        session_id=42,
        normalize=normalize,
        pca=pca,
        pca_components=pca_comp if pca else None,
        verbose=False,
        html=False,
    )
    st.success("Setup concluído")

    resultados, objetos = [], {}
    k_models = {"kmeans", "hclust", "birch", "spectral"}

    for m in models_to_try:
        try:
            params = {}
            if (k_clusters is not None) and (int(k_clusters) > 0) and (m in k_models):
                params["num_clusters"] = int(k_clusters)
            if m == "dbscan":
                params["eps"] = float(dbscan_eps)
                params["min_samples"] = int(dbscan_min_samples)
            if m == "optics":
                params["min_samples"] = int(optics_min_samples)

            model = create_model(m, **params)
            labeled = assign_model(model, transformation=True)
            X = labeled.drop(columns=["Cluster"])
            y = labeled["Cluster"]
            sil, ch, db = compute_metrics(X, y)
            resultados.append([m, sil, ch, db])
            objetos[m] = (model, labeled)
        except Exception as e:
            resultados.append([m, np.nan, np.nan, np.nan])
            objetos[m] = (str(e), None)

    res_df = pd.DataFrame(resultados, columns=["Modelo", "Silhouette", "Calinski-Harabasz", "Davies-Bouldin"])

    # Sempre atualizar session_state
    st.session_state["objetos"] = objetos
    st.session_state["res_df"] = res_df

# -------------------- Visualização dos resultados (organizada em abas) --------------------
if "res_df" in st.session_state and not st.session_state["res_df"].empty:
    # seletor de modelo (usado por todas as abas)
    escolha = st.selectbox("Modelo", st.session_state["res_df"]["Modelo"].tolist(), key="selecionar_modelo_resultados")
    obj, labeled_final = st.session_state["objetos"].get(escolha, (None, None))

    if isinstance(obj, str) or labeled_final is None:
        st.warning(f"Não foi possível analisar {escolha}")
    else:
        # Cria abas focadas
        tab_modelos, tab_perfis, tab_vis_ad, tab_projecoes, tab_pycaret, tab_dendro = st.tabs([
            "Resumo do Modelo", "Perfis & Estatísticas", "Visualizações Adicionais", "Projeções 2D", "PyCaret Plots", "Dendrograma & Heatmap"
        ])

        # -------------------- Aba 1: Resumo do Modelo --------------------
        with tab_modelos:
            st.subheader("Amostra com clusters atribuídos (dinâmico)")
            st.markdown("💡 Ideia geral: Essa aba mostra um resumo rápido de como os grupos (clusters) foram formados, e algumas métricas de “qualidade” da clusterização.")
            st.markdown(
                "Cada linha representa uma amostra, e a coluna Cluster indica a qual grupo ela pertence. "
                "Troque o modelo no seletor para comparar diferentes resultados."
            )
            st.dataframe(labeled_final.head())
            st.markdown("Se duas amostras estão no mesmo cluster, significa que seus perfis de expressão são parecidos. Se estão em clusters diferentes, são bem diferentes")
            st.markdown("**Métricas rápidas do modelo selecionado**")
            st.markdown("""
                        Silhouette: mede o quão bem cada amostra se encaixa no seu cluster. Próximo de 1 → perfeito, próximo de 0 → borderline, negativo → pode estar no cluster errado.

                        Calinski-Harabasz: maior valor = clusters mais separados e compactos.

                        Davies-Bouldin: menor valor = clusters mais distintos entre si.

                        Essas métricas ajudam a ter uma ideia rápida se a clusterização “faz sentido” ou se os grupos estão muito misturados.""")
            try:
                counts = labeled_final["Cluster"].value_counts().sort_index()
                st.write("Tamanho por cluster:")
                st.table(pd.DataFrame({"Cluster": counts.index.astype(str), "Count": counts.values}))
                sil, ch, db = compute_metrics(labeled_final.drop(columns=["Cluster"]), labeled_final["Cluster"])
                st.write(f"- Silhouette: {sil if not np.isnan(sil) else 'N/A'}")
                st.write(f"- Calinski-Harabasz: {ch if not np.isnan(ch) else 'N/A'}")
                st.write(f"- Davies-Bouldin: {db if not np.isnan(db) else 'N/A'}")
            except Exception as e:
                st.info(f"Não foi possível calcular resumo rápido: {e}")

        # -------------------- Aba 2: Perfis & Estatísticas --------------------
        with tab_perfis:
            st.subheader("Perfis dos clusters (estatísticas)")
            st.markdown("""
                        💡 Ideia geral:
                        Mostra os valores médios, medianas e desvios de cada cluster. Basicamente, ajuda a entender como é o “perfil típico” de cada grupo.

                        🔹 Média/mediana/desvio/min/max por cluster
                        Cada cluster tem seu “perfil médio”.

                        Linha do meio(mediana) → valor médio do gene/feature.

                        Desvio/std → indica se os valores dentro do cluster são parecidos ou bem diferentes.

                        Se um cluster tem genes com média alta, esses genes provavelmente estão ativos e podem estar envolvidos em funções importantes. Se o desvio é grande, significa que nem todas as amostras dentro do cluster são iguais — ou seja, há heterogeneidade.""")
            prof = cluster_profiles_table(labeled_final, cluster_col="Cluster")
            if prof.empty:
                st.info("Sem colunas numéricas para apresentar perfis.")
            else:
                st.dataframe(prof)

            st.markdown("""
                🔹 Média

                A média de um gene dentro de um cluster mostra quão ativo esse gene costuma estar nesse grupo.

                🔹 Mediana

                A mediana é útil para ver o valor típico sem ser enganado por valores extremos.

                Se a média de um gene é alta, mas a mediana é baixa, isso indica que algumas amostras têm valores muito altos, enquanto a maioria está baixa. Ou seja, o cluster não é tão homogêneo quanto parece pela média.

                🔹 Desvio (padrão)

                O desvio serve para dizer o quanto os genes variam dentro do cluster.

                🔹 Mínimo (min)

                Mostra o valor mais baixo de expressão de um gene dentro do cluster.

                🔹 Máximo (max)

                Mostra o valor mais alto de expressão de um gene dentro do cluster.""")

        # -------------------- Aba 3: Visualizações Adicionais --------------------
        with tab_vis_ad:
            st.subheader("Visualizações adicionais")
            # 1) Tamanho dos clusters (barra)
            st.markdown("**Tamanho dos clusters**")
            st.caption("Conta de amostras por cluster — revela se algum cluster é muito pequeno (possível ruído) ou muito grande (dominante).")
            try:
                counts = labeled_final["Cluster"].value_counts().sort_index()
                fig_counts = px.bar(x=counts.index.astype(str), y=counts.values,
                                    labels={'x':'Cluster','y':'Count'}, title="Tamanho dos clusters")
                st.plotly_chart(fig_counts, use_container_width=True)
            except Exception as e:
                st.info(f"Tamanho dos clusters não disponível: {e}")

            # 2) Silhouette por amostra
            try:
                X_vis = labeled_final.drop(columns=["Cluster"])
                labels = labeled_final["Cluster"].values
                sil_samples = silhouette_samples(X_vis.values, labels)
                df_sil = pd.DataFrame({"silhouette": sil_samples, "cluster": labels.astype(str)})
                df_sil = df_sil.sort_values(["cluster", "silhouette"], ascending=[True, False])
                fig_sil = px.box(df_sil, x="cluster", y="silhouette", points="all",
                                 title="Silhouette por cluster (box + pontos)")
                st.plotly_chart(fig_sil, use_container_width=True)
                st.markdown("Boxplot das silhuetas por cluster — mostra a distribuição de qualidade de atribuição dentro de cada cluster. Valores próximos de 1 = bem classificados; negativos = possivelmente mal atribuídos.")
            except Exception as e:
                st.info(f"Silhouette por amostra não disponível: {e}")
            # 3) Scatter matrix (top-N features)
            try:
                top_n = min(6, max(2, int(st.sidebar.number_input('Top-N features para scatter matrix', 2, 12, 6))))
                X_vis = labeled_final.drop(columns=["Cluster"])
                top_feats = X_vis.var(axis=0).sort_values(ascending=False).head(top_n).index.tolist()
                sample_for_plot = labeled_final[top_feats + ["Cluster"]].sample(n=min(500, len(labeled_final)), random_state=42)
                fig_scatter = px.scatter_matrix(sample_for_plot, dimensions=top_feats, color="Cluster",
                                                title=f"Scatter matrix — top {top_n} features por variância (sample)")
                fig_scatter.update_traces(diagonal_visible=False)
                st.plotly_chart(fig_scatter, use_container_width=True)
                st.markdown("""

                            💡 Mostra gráficos de dispersão comparando pares dos genes/features que mais variam entre as amostras. Cada ponto é uma amostra, e a cor indica a qual cluster ela pertence.

                            Se os pontos de cada cluster aparecem agrupados em regiões diferentes, significa que essas features realmente ajudam a separar os grupos. Se estiverem misturados, essas variáveis não distinguem bem os clusters.""")
                
            except Exception as e:
                st.info(f"Scatter matrix não disponível: {e}")
        
            st.markdown("")  # linha em branco
            st.markdown("")

            # 5) Correlograma (sample)
            try:
                X_corr = labeled_final.drop(columns=["Cluster"]).sample(n=min(500, len(labeled_final)), random_state=42)
                corr = X_corr.corr()
                fig_corr = px.imshow(corr, labels=dict(x="features", y="features", color="correlation"),
                                     title="Correlograma (sample)")
                fig_corr.update_layout(height=700)
                st.plotly_chart(fig_corr, use_container_width=True)
                st.markdown("""
                💡 Mostra como os genes/features se comportam juntos. Cada quadradinho do gráfico indica se dois genes tendem a subir e descer juntos ou de forma oposta.

                Se dois genes estão muito correlacionados, eles podem estar ligados a funções parecidas ou reagir de forma similar em diferentes amostras.
                """)
            except Exception as e:
                st.info(f"Correlograma não disponível: {e}")

        # -------------------- Aba 4: Projeções 2D --------------------
        with tab_projecoes:
            st.subheader("Projeção 2D (PCA / t-SNE)")
            st.markdown("""
🔸 PCA (Análise de Componentes Principais)

💡 Como funciona:

O PCA encontra as direções que explicam mais variação nos dados.
Essas direções viram os eixos X e Y do gráfico.
Ou seja, ele comprime várias colunas em poucos eixos sem perder muito da informação mais importante.

Se dois genes/amostras estão muito próximos no gráfico, significa que eles têm padrões de expressão parecidos.
Se estão distantes, seus comportamentos são diferentes.

🔸 t-SNE (t-Distributed Stochastic Neighbor Embedding)

💡 Como funciona:

O t-SNE tenta manter os vizinhos próximos e destacar separações locais.
É ideal quando os clusters têm formas curvas ou complexas, que o PCA não consegue mostrar direito.

Clusters que podem parecer misturados no PCA podem ficar bem separados no t-SNE.
Ajuda a enxergar “grupos naturais” de genes/amostras que se comportam de forma parecida.
                        
""")
            try:
                X_for_vis = labeled_final.drop(columns=["Cluster"])
                if X_for_vis.shape[0] == 0 or X_for_vis.shape[1] == 0:
                    st.info("Dados insuficientes para projeção.")
                else:
                    # Remove UMAP da lista de métodos
                    method = st.selectbox("Método", ["PCA", "t-SNE"], index=0, key="proj_method")
                    sample_n = st.number_input("Amostras para projeção (0 = usar todas)", min_value=0, max_value=2000, value=500, key="proj_sample_n")
                    Xpv = X_for_vis.copy()
                    if sample_n and sample_n > 0 and Xpv.shape[0] > sample_n:
                        Xpv = Xpv.sample(n=sample_n, random_state=42)
                    # padroniza (z-score)
                    Xpv = (Xpv - Xpv.mean()) / Xpv.std(ddof=0).replace(0, np.nan)
                    Xpv = Xpv.fillna(0.0).values

                    proj = None
                    if method == "PCA":
                        pca_model = PCA(n_components=2, random_state=42)
                        proj = pca_model.fit_transform(Xpv)
                        varexpl = pca_model.explained_variance_ratio_.cumsum()
                        st.caption(f"Variância acumulada pelos 2 PCs: {varexpl[min(1, len(varexpl)-1)]:.3f}")
                    else:  # t-SNE
                        from sklearn.manifold import TSNE
                        proj = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(Xpv)

                    if proj is not None:
                        proj_df = pd.DataFrame(proj, columns=["dim1", "dim2"])
                        # alinha clusters aos indices amostrados
                        sampled_idx = X_for_vis.index[:len(proj_df)]
                        proj_df["cluster"] = labeled_final.loc[sampled_idx, "Cluster"].astype(str).values
                        fig = px.scatter(proj_df, x="dim1", y="dim2", color="cluster", title=f"Projeção 2D ({method})")
                        st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Falha ao gerar projeção 2D: {e}")

       # -------------------- Aba 5: Visualizações PyCaret --------------------
with tab_pycaret:
    st.subheader("Visualizações do PyCaret")
    st.markdown("Gráficos extras para interpretar clusters: centróides, distância entre clusters, importância de features e ANOVA.")

    if isinstance(obj, str) or labeled_final is None:
        st.warning(f"Não foi possível analisar {escolha}")
    else:
        X_all = labeled_final.drop(columns=["Cluster"])
        y_all_raw = labeled_final["Cluster"]

        try:
            y_codes, y_uniques = pd.factorize(y_all_raw)
        except Exception:
            try:
                y_codes = y_all_raw.astype(int).values
                y_uniques = np.unique(y_codes)
            except Exception:
                y_codes = np.arange(len(y_all_raw))
                y_uniques = np.unique(y_codes)

        # 1) Heatmap das médias (centróides)
        st.markdown("**Heatmap: médias/centróides por cluster**")
        st.markdown("💡 Ideia geral: Cada linha = cluster, cada coluna = gene/feature. As cores mostram os valores médios de expressão.")
        st.markdown("👉 Exemplo prático: Permite ver rapidamente quais genes estão mais ativos em cada cluster.")
        try:
            centroids = labeled_final.groupby("Cluster").mean()
            if not centroids.empty:
                fig_cent = px.imshow(
                    centroids,
                    labels=dict(x="feature", y="cluster", color="mean"),
                    title="Centróides (média por feature por cluster)",
                    aspect="auto"
                )
                fig_cent.update_layout(height=450)
                st.plotly_chart(fig_cent, use_container_width=True)
            else:
                st.info("Centróides vazios — nenhuma coluna numérica encontrada.")
        except Exception as e:
            st.info(f"Não foi possível gerar heatmap de centróides: {e}")

        # 2) Matriz de distâncias entre centróides
        st.markdown("**Matriz de distâncias entre centróides**")
        st.markdown("💡 Ideia geral: Mostra quão separados estão os clusters uns dos outros.")
        st.markdown("👉 Exemplo prático: Clusters próximos → grupos parecidos; Clusters distantes → grupos bem diferentes.")
        try:
            from sklearn.metrics import pairwise_distances
            cent = centroids.values
            if cent.size == 0:
                st.info("Centróides vazios — nada a mostrar.")
            else:
                dist_mat = pairwise_distances(cent, metric="euclidean")
                fig_dist = px.imshow(dist_mat, x=centroids.index.astype(str), y=centroids.index.astype(str),
                                    labels=dict(x="cluster", y="cluster", color="distância"),
                                    title="Distância euclidiana entre centróides")
                fig_dist.update_layout(height=420)
                st.plotly_chart(fig_dist, use_container_width=True)
        except Exception as e:
            st.info(f"Não foi possível gerar matriz de distâncias: {e}")

        # 3) Importância de features via RandomForest (proxy)
        st.markdown("**Importância de features (RandomForest proxy)**")
        st.markdown("💡 Ideia geral: Mostra quais genes/features mais ajudam a diferenciar os clusters.")
        st.markdown("👉 Exemplo prático: Top genes → bons candidatos para estudos biológicos mais detalhados.")
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split

            if X_all.shape[1] < 2:
                st.info("Poucas features para treinar RandomForest.")
            else:
                max_features_rf = min(500, X_all.shape[1])
                var_order = X_all.var(axis=0).sort_values(ascending=False).head(max_features_rf).index
                X_rf = X_all[var_order].fillna(0.0)
                y_enc = y_codes
                stratify_arg = y_enc if len(np.unique(y_enc)) > 1 and min(np.bincount(y_enc)) > 1 else None
                X_tr, X_te, y_tr, y_te = train_test_split(X_rf, y_enc, test_size=0.25, random_state=42, stratify=stratify_arg)
                rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
                with st.spinner("Treinando RandomForest para estimar importância de features (proxy)..."):
                    rf.fit(X_tr, y_tr)
                importances = pd.Series(rf.feature_importances_, index=X_rf.columns).sort_values(ascending=False)
                top_n_imp = int(st.number_input("Top-N features (Feature Importance)", min_value=3, max_value=50, value=10, key="rf_topn"))
                train_score = rf.score(X_tr, y_tr)
                test_score = rf.score(X_te, y_te)
                st.write(f"Importâncias (top {top_n_imp}) — accuracy proxy (treino/teste): {train_score:.3f} / {test_score:.3f}")
                fig_imp = px.bar(x=importances.head(top_n_imp).index, y=importances.head(top_n_imp).values,
                                labels={"x":"feature","y":"importance"}, title=f"Top {top_n_imp} features por importance (RandomForest proxy)")
                fig_imp.update_layout(xaxis_tickangle=-45, height=420)
                st.plotly_chart(fig_imp, use_container_width=True)
        except Exception as e:
            st.info(f"Importância de features indisponível: {e}")

        # 4) ANOVA (SelectKBest style) — top features por F-score
        st.markdown("**Top features por ANOVA (F-score)**")
        st.markdown("💡 Ideia geral: Identifica genes/features com maiores diferenças médias entre clusters. O heatmap mostra como essas features variam entre os grupos.")
        st.markdown("👉 Exemplo prático: Ajuda a entender quais genes separam melhor os clusters.")
        try:
            from sklearn.feature_selection import f_classif
            if X_all.shape[1] < 2 or len(np.unique(y_codes)) < 2:
                st.info("ANOVA requer pelo menos 2 features e 2 clusters distintos.")
            else:
                max_feats_anova = min(2000, X_all.shape[1])
                cols_for_anova = X_all.var(axis=0).sort_values(ascending=False).head(max_feats_anova).index
                X_anova = X_all[cols_for_anova].fillna(0.0).values
                F, pvals = f_classif(X_anova, y_codes)
                f_series = pd.Series(F, index=cols_for_anova).sort_values(ascending=False)
                top_n_anova = int(st.number_input("Top-N features (ANOVA F)", min_value=3, max_value=50, value=10, key="anova_topn"))
                st.dataframe(pd.DataFrame({"F_score": f_series.head(top_n_anova)}).style.format("{:.4f}"))
                top_an_feats = f_series.head(top_n_anova).index.tolist()
                mean_by_cluster = labeled_final.groupby("Cluster")[top_an_feats].mean()
                fig_an_heat = px.imshow(mean_by_cluster, labels=dict(x="feature", y="cluster", color="mean"),
                                        title=f"Heatmap — top {top_n_anova} features por ANOVA (média por cluster)")
                fig_an_heat.update_layout(height=420)
                st.plotly_chart(fig_an_heat, use_container_width=True)
        except Exception as e:
            st.info(f"ANOVA indisponível: {e}")

        # 5) Silhouette detalhado (scatter ordenado)
        st.markdown("**Silhouette detalhado (ordenado)**")
        st.markdown("💡 Ideia geral: Permite ver quais amostras podem estar mal atribuídas (valores negativos).")
        st.markdown("👉 Exemplo prático: Amostras problemáticas podem indicar erros de clusterização ou casos especiais.")
        try:
            from sklearn.metrics import silhouette_samples
            X_vis = X_all.fillna(0.0).values
            labels = y_codes
            if len(np.unique(labels)) > 1:
                sil_vals = silhouette_samples(X_vis, labels)
                sil_df = pd.DataFrame({"silhouette": sil_vals, "cluster": labels.astype(str)})
                sil_df = sil_df.sort_values(["cluster", "silhouette"], ascending=[True, False]).reset_index(drop=True)
                sil_df["idx"] = sil_df.index
                fig_scat = px.scatter(sil_df, x="idx", y="silhouette", color="cluster",
                                    title="Silhouette por amostra (ordenado por cluster)")
                fig_scat.update_layout(height=350, xaxis_title="amostras (ordenadas)", yaxis_title="silhouette")
                st.plotly_chart(fig_scat, use_container_width=True)
                st.markdown("Atenção a valores negativos: amostras com silhouette negativo podem estar mal atribuídas.")
            else:
                st.info("Silhouette requer pelo menos 2 clusters.")
        except Exception as e:
            st.info(f"Silhouette detalhado não disponível: {e}")

        # 6) Centróides em 2D (PCA)
        st.markdown("**Centróides projetados (PCA 2D)**")
        st.markdown("💡 Ideia geral: Mostra os clusters como pontos no espaço 2D, usando os valores médios das features.")
        st.markdown("👉 Exemplo prático: Permite ver visualmente se os clusters estão bem separados ou se há sobreposição.")
        try:
            from sklearn.decomposition import PCA as _PCA
            if 'centroids' in locals() and centroids.shape[0] >= 2 and centroids.shape[1] >= 2:
                pca_c = _PCA(n_components=2, random_state=42).fit_transform(centroids.values)
                cent_df = pd.DataFrame(pca_c, columns=["pc1", "pc2"], index=centroids.index.astype(str))
                cent_df["cluster"] = cent_df.index
                fig_cent2 = px.scatter(cent_df, x="pc1", y="pc2", text=cent_df.index, title="Centróides (PCA 2D)")
                fig_cent2.update_traces(textposition="top center")
                fig_cent2.update_layout(height=420)
                st.plotly_chart(fig_cent2, use_container_width=True)
            else:
                st.info("Centróides insuficientes para projeção PCA 2D.")
        except Exception as e:
            st.info(f"Projeção dos centróides indisponível: {e}")

        # 7) Plots nativos do PyCaret (opcional)
st.markdown("**Plots nativos do PyCaret (se disponíveis)**")
st.markdown("💡 Plots automáticos do PyCaret para análise de clusters.")

for plot_type in ["elbow", "silhouette", "tsne"]:
    try:
        st.markdown(f"**Plot PyCaret: {plot_type}**")
        
        # Explicações baseadas no tipo
        if plot_type == "elbow":
            st.markdown("💡 Gráfico de cotovelo: mostra a soma das distâncias quadradas dentro dos clusters para diferentes números de clusters.")
            st.markdown("👉 Procure o 'cotovelo' no gráfico — o ponto onde a redução da distância começa a diminuir lentamente indica um bom número de clusters.")
        elif plot_type == "silhouette":
            st.markdown("💡 Gráfico de silhouette: mostra a média da silhouette por cluster, indicando quão bem as amostras estão atribuídas aos clusters.")
            st.markdown("👉 Valores próximos de 1 → cluster bem definido; valores próximos de 0 → amostra na fronteira; negativos → amostra possivelmente no cluster errado.")
        elif plot_type == "tsne":
            st.markdown("💡 Gráfico de t-SNE: projeção 2D não linear das amostras, útil para visualizar separações complexas entre clusters.")
            st.markdown("👉 Pontos próximos → amostras com comportamento parecido; pontos distantes → amostras diferentes; útil para ver se os clusters se separam visualmente.")
        
        plot_model(obj, plot=plot_type, display_format="streamlit")
    except Exception as e:
        st.info(f"{plot_type} não disponível via PyCaret para {escolha}")



        # -------------------- Aba 6: Dendrograma & Heatmap --------------------
        with tab_dendro:
            st.subheader("Dendrograma (hierárquico)")
            st.markdown("💡 Mostra a hierarquia de fusões entre amostras — passo a passo.")
            st.markdown("👉 Clusters mais próximos → se fundem primeiro; Clusters distantes → se fundem por último.")
            if escolha == "hclust" or show_dendro_anyway:
                try:
                    make_dendrogram(labeled_final.drop(columns=["Cluster"]), sample_cap=250, method="ward")
                except Exception as e:
                    st.info(f"Falha ao gerar dendrograma: {e}")
            else:
                st.info("Dendrograma recomendado apenas para hclust. Marque 'Mostrar dendrograma mesmo se não for hclust' para forçar.")

            st.subheader("Heatmap das médias por cluster")
            st.markdown("💡 Visualiza os valores médios por cluster, destacando padrões claros de diferenciação.")
            st.markdown("👉 Cores mais fortes → genes mais ativos; cores mais claras → genes menos expressos.")
            try:
                top_n = int(heatmap_topn) if heatmap_topn and heatmap_topn > 0 else None
                plot_cluster_means_heatmap(labeled_final, cluster_col="Cluster", zscore=heatmap_zscore, top_n_features=top_n)
            except Exception as e:
                st.info(f"Falha ao gerar heatmap: {e}")

            st.markdown("### Downloads")
            try:
                st.download_button("Baixar clusters (CSV)", labeled_final.to_csv(index=False).encode("utf-8"), "clusters.csv")
            except Exception as e:
                st.info(f"Não foi possível preparar o download CSV: {e}")

            try:
                # salva modelo temporariamente com PyCaret e oferece download
                save_model(obj, "modelo_cluster")
                with open("modelo_cluster.pkl", "rb") as f:
                    st.download_button("Baixar modelo (PKL)", f, "modelo_cluster.pkl")
            except Exception as e:
                st.info(f"Não foi possível salvar/baixar o modelo: {e}")
# -------------------- Fim da visualização organizada --------------------
