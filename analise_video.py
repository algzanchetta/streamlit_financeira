import sqlite3
import hashlib
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
import plotly.express as px
import plotly.graph_objects as go

DB_PATH = Path(__file__).parent / "dbase.db"

# ==============================================================================
# CONFIGURAÇÕES DE OFUSCAÇÃO (MODO DEMONSTRAÇÃO PARA VÍDEO)
# ==============================================================================
MODO_DEMONSTRACAO = True  # Defina como False para usar dados reais
FATOR_ESCALA_VALOR = 1.37 # Multiplica valores (mantém todas as % e proporções)
DESLOCAMENTO_DIAS = 145   # Desloca datas para o futuro/passado

COLORS = {
    "azul": "#4C78A8",
    "laranja": "#F58518",
    "verde": "#54A24B",
    "vermelho": "#E45756",
    "roxo": "#B279A2",
    "teal": "#72B7B2",
    "amarelo": "#EECA3B",
}
SEQUENCE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2", "#72B7B2", "#EECA3B", "#BAB0AC"]

PDD_FAIXAS = [
    ("1-30 dias", 1, 30, 0.02),
    ("31-60 dias", 31, 60, 0.10),
    ("61-90 dias", 61, 90, 0.30),
    ("91-180 dias", 91, 180, 0.50),
    ("181+ dias", 181, 10_000, 1.00),
]

def is_streamlit_running():
    try:
        return get_script_run_ctx() is not None
    except Exception:
        return False

def parse_dt(series):
    return pd.to_datetime(series, errors="coerce")

def show(df):
    st.dataframe(df.reset_index(drop=True), hide_index=True, use_container_width=True)

def fmt_brl(v):
    return f"R$ {v:,.2f}"

def gerar_pseudonimo(texto, prefixo="Registro"):
    """Gera um nome falso, mas consistente. O mesmo nome original sempre gera o mesmo código."""
    if pd.isna(texto) or str(texto).strip() == "" or str(texto).lower() in ["nan", "none", "nat"]:
        return "N/A"
    hash_hex = hashlib.md5(str(texto).encode('utf-8')).hexdigest()[:4]
    return f"{prefixo} {int(hash_hex, 16) % 10000:04d}"

def normalize_estabelecimento(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return "OUTROS"
    x = name.upper()
    patterns = [
        ("RESTAURANT", "RESTAURANTE"), ("PIZZA", "PIZZARIA"), ("BAR", "BAR"), ("PUB", "BAR"),
        ("PADARI", "PADARIA/PANIFICADORA"), ("PANIFICAD", "PADARIA/PANIFICADORA"), ("BOULANGER", "PADARIA/PANIFICADORA"),
        ("LOJA", "LOJA"), ("SUPERMERC", "SUPERMERCADO"), ("MERCEARIA", "MERCEARIA"), ("CAF", "CAFETERIA"),
        ("HOTEL", "HOTEL"), ("ASSIST", "ASSISTENCIA"), ("BARBEARIA", "BARBEARIA"), ("ESPET", "ESPETINHO"),
        ("PEIXA", "PEIXARIA"), ("VEND", "VENDEDOR"), ("FRUTARIA", "FRUTARIA"), ("SALGADO", "SALGADOS"),
        ("SALAO", "SALAO/BELEZA"), ("BELEZA", "SALAO/BELEZA"), ("UNHA", "SALAO/BELEZA"),
        ("CELULA", "ELETRONICOS"), ("ELETRONIC", "ELETRONICOS"), ("SERRALHERIA", "SERRALHERIA"),
        ("MECANIC", "OFICINA/MECANICA"), ("BORRACHA", "OFICINA/MECANICA"), ("LAVA", "LAVA-JATO"),
        ("MERCADO", "MERCADO"), ("PENSAO", "PENSAO"), ("DEPOSITO", "DEPOSITO/ATACADO"),
        ("ATACAD", "DEPOSITO/ATACADO"), ("CONSTRU", "CONSTRUCAO"), ("MATERIAL", "CONSTRUCAO"),
        ("MOVEL", "MOVEIS"), ("ROUPA", "VESTUARIO"), ("VESTUARIO", "VESTUARIO"), ("SAPATO", "VESTUARIO"),
        ("ACOUGUE", "ACOUGUE"), ("LANCH", "LANCHONETE"), ("ACAI", "ACAI"), ("BOLO", "CONFEITARIA"),
        ("DOCE", "CONFEITARIA"), ("FARM", "FARMACIA"), ("CABEL", "SALAO/BELEZA"),
        ("MOTOR", "TRANSPORTE"), ("TRANSPORT", "TRANSPORTE"), ("MOTOTAXI", "TRANSPORTE"),
        ("PADARIA", "PADARIA/PANIFICADORA"),
    ]
    for token, label in patterns:
        if token in x:
            return label
    first = x.split()[0]
    if first in {"RESTAURANTE", "PIZZARIA", "PEIXARIA", "VENDEDOR", "CONVENIENCIA", "BAR", "ESPETINHO",
                 "BARBEARIA", "PADARIA", "LOJA", "SUPERMERCADO", "CAFETERIA", "ASSISTENCIA", "HOTEL",
                 "FRUTARIA", "SALGADOS", "LANCHONETE"}:
        return first
    return "OUTROS"

def ofuscar_dados(usuarios, clientes, contratos, movimentos):
    """Aplica ofuscação nos dataframes sem tocar no banco de dados."""
    if not MODO_DEMONSTRACAO:
        return usuarios, clientes, contratos, movimentos

    # 1. Nomes de Usuários (Agentes)
    if 'usuario' in usuarios.columns:
        usuarios['usuario'] = usuarios['usuario'].apply(lambda x: gerar_pseudonimo(x, "Agente"))

    # 2. Nomes de Clientes e Estabelecimentos
    if 'cliente' in clientes.columns:
        clientes['cliente'] = clientes['cliente'].apply(lambda x: gerar_pseudonimo(x, "Cliente"))
    if 'nome_estabelecimento' in clientes.columns:
        clientes['nome_estabelecimento'] = clientes['nome_estabelecimento'].apply(lambda x: gerar_pseudonimo(x, "Empresa"))

    # 3. Datas em Contratos
    for col in ['dtinicio', 'dtfim', 'dtatualizacao']:
        if col in contratos.columns:
            contratos[col] = pd.to_datetime(contratos[col]) + pd.Timedelta(days=DESLOCAMENTO_DIAS)

    # 4. Valores em Contratos
    for col in ['valor', 'valor_parcelado']:
        if col in contratos.columns:
            contratos[col] = pd.to_numeric(contratos[col], errors='coerce').fillna(0) * FATOR_ESCALA_VALOR
    
    # Recalcular derivados de valor para manter integridade matemática
    contratos['qtd_parcela'] = pd.to_numeric(contratos.get('qtd_parcela'), errors='coerce').fillna(0)
    contratos['parcela_esperada'] = np.where(contratos['qtd_parcela'] > 0, contratos['valor_parcelado'] / contratos['qtd_parcela'], 0)
    contratos['frac_principal'] = np.where(contratos['valor_parcelado'] > 0, contratos['valor'] / contratos['valor_parcelado'], 0)
    contratos['juros_previstos'] = contratos['valor_parcelado'] - contratos['valor']

    # 5. Datas em Movimentações
    for col in ['dtinicio', 'dtfim', 'dtvenc', 'dtrecebimento']:
        if col in movimentos.columns:
            movimentos[col] = pd.to_datetime(movimentos[col]) + pd.Timedelta(days=DESLOCAMENTO_DIAS)

    # 6. Valores em Movimentações
    for col in ['valorrecebido', 'areceber', 'valorcontrato', 'desconto']:
        if col in movimentos.columns:
            movimentos[col] = pd.to_numeric(movimentos[col], errors='coerce').fillna(0) * FATOR_ESCALA_VALOR

    return usuarios, clientes, contratos, movimentos

@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    usuarios = pd.read_sql_query("SELECT * FROM usuarios", conn)
    clientes = pd.read_sql_query("SELECT * FROM clientes", conn)
    contratos = pd.read_sql_query("SELECT * FROM contratos2", conn)
    movimentos = pd.read_sql_query("SELECT * FROM contratos_movimentacoes2", conn)
    conn.close()

    # ---------- Clientes ----------
    clientes["idade"] = pd.to_numeric(clientes.get("idade"), errors="coerce")
    clientes["idade"] = clientes["idade"].where(clientes["idade"] > 0, np.nan)
    clientes["genero"] = clientes.get("genero").astype(str).str.strip()
    clientes["genero_cat"] = clientes["genero"].map({"1": "Masculino", "0": "Feminino"}).fillna("Outro")
    idade_bins = [0, 18, 25, 35, 45, 55, 65, 200]
    idade_labels = ["<18", "18-25", "26-35", "36-45", "46-55", "56-65", ">65"]
    clientes["faixa_idade"] = pd.cut(clientes["idade"], bins=idade_bins, labels=idade_labels)
    clientes["faixa_idade"] = clientes["faixa_idade"].cat.add_categories(["Sem idade"]).fillna("Sem idade")
    clientes["avaliacao"] = clientes["avaliacao"].astype(str).fillna("Nao avaliado")
    clientes["nome_estabelecimento"] = clientes["nome_estabelecimento"].astype(str).fillna("Desconhecido")

    # ---------- Contratos ----------
    for col in ["dtinicio", "dtfim", "dtatualizacao"]:
        contratos[col] = parse_dt(contratos.get(col))
    contratos["valor"] = pd.to_numeric(contratos.get("valor"), errors="coerce").fillna(0)
    contratos["valor_parcelado"] = pd.to_numeric(contratos.get("valor_parcelado"), errors="coerce")
    contratos["valor_parcelado"] = contratos["valor_parcelado"].fillna(contratos["valor"])
    contratos["qtd_parcela"] = pd.to_numeric(contratos.get("qtd_parcela"), errors="coerce").fillna(0)
    contratos["parcela_esperada"] = np.where(contratos["qtd_parcela"] > 0, contratos["valor_parcelado"] / contratos["qtd_parcela"], 0)
    contratos["frac_principal"] = np.where(contratos["valor_parcelado"] > 0, contratos["valor"] / contratos["valor_parcelado"], 0)
    contratos["juros_previstos"] = contratos["valor_parcelado"] - contratos["valor"]

    contratos = contratos.merge(
        clientes[["id", "cliente", "genero_cat", "faixa_idade", "avaliacao", "nome_estabelecimento"]],
        left_on="idcliente", right_on="id", how="left", suffixes=("", "_cliente"),
    )
    contratos["usuario_nome"] = contratos["idusuario"].map(usuarios.set_index("id")["usuario"])

    # ---------- Movimentacoes ----------
    for col in ["dtinicio", "dtfim", "dtvenc", "dtrecebimento"]:
        movimentos[col] = parse_dt(movimentos.get(col))
    movimentos["valorrecebido"] = pd.to_numeric(movimentos.get("valorrecebido"), errors="coerce").fillna(0)
    movimentos["areceber"] = pd.to_numeric(movimentos.get("areceber"), errors="coerce").fillna(0)
    movimentos["valorcontrato"] = pd.to_numeric(movimentos.get("valorcontrato"), errors="coerce").fillna(0)
    movimentos["desconto"] = pd.to_numeric(movimentos.get("desconto"), errors="coerce").fillna(0)
    movimentos["recebido"] = pd.to_numeric(movimentos.get("recebido"), errors="coerce")
    movimentos["paid"] = movimentos["recebido"] == 1
    movimentos["status_pago"] = movimentos["paid"] | (movimentos["ok"].astype(str).str.lower() == "sim")

    movimentos = movimentos.merge(
        contratos[["id", "parcela_esperada", "frac_principal", "valor", "valor_parcelado", "juros_previstos"]],
        left_on="idcontrato", right_on="id", how="left", suffixes=("", "_c"),
    )
    movimentos["parcela"] = movimentos["parcela_esperada"].fillna(0)
    movimentos["frac_principal"] = movimentos["frac_principal"].fillna(0)
    movimentos["juros_frac"] = 1.0 - movimentos["frac_principal"]

    today = pd.Timestamp(date.today())

    def compute_delay(row):
        if pd.isna(row["dtvenc"]):
            return 0
        if pd.notna(row["dtrecebimento"]):
            return max((row["dtrecebimento"] - row["dtvenc"]).days, 0)
        return max((today - row["dtvenc"]).days, 0)

    movimentos["dias_atraso"] = movimentos.apply(compute_delay, axis=1)
    movimentos["vencido"] = (~movimentos["status_pago"]) & movimentos["dtvenc"].notna() & (movimentos["dtvenc"] < today)
    movimentos["a_vencer"] = (~movimentos["status_pago"]) & movimentos["dtvenc"].notna() & (movimentos["dtvenc"] >= today)
    movimentos["atraso_90"] = movimentos["vencido"] & (movimentos["dias_atraso"] >= 90)

    movimentos = movimentos.merge(
        clientes[["id", "cliente", "genero_cat", "faixa_idade", "avaliacao", "nome_estabelecimento"]],
        left_on="idcliente", right_on="id", how="left", suffixes=("", "_cliente"),
    )
    movimentos["nome_estabelecimento"] = movimentos["nome_estabelecimento"].fillna("Desconhecido")
    movimentos["nome_estabelecimento_norm"] = movimentos["nome_estabelecimento"].astype(str).apply(normalize_estabelecimento)
    movimentos["usuario_nome"] = movimentos["idusuario"].map(usuarios.set_index("id")["usuario"])

    # ---------- Agregacao por contrato ----------
    contrato_agg = movimentos.groupby("idcontrato").agg(
        total_recebido=("valorrecebido", "sum"),
        total_desconto=("desconto", "sum"),
        total_aberto=("areceber", "sum"),
        aberto_nao_pago=("areceber", lambda s: s[movimentos.loc[s.index, "status_pago"] == False].sum()),
        parcelas_pagas=("status_pago", "sum"),
        parcelas_total=("id", "count"),
        vencido_valor=("areceber", lambda s: s[movimentos.loc[s.index, "vencido"]].sum()),
    ).reset_index()
    
    contratos = contratos.merge(contrato_agg, left_on="id", right_on="idcontrato", how="left").fillna(
        {c: 0 for c in ["total_recebido", "total_desconto", "total_aberto", "aberto_nao_pago",
                        "parcelas_pagas", "parcelas_total", "vencido_valor"]}
    )
    contratos["percentual_recebido"] = (
        contratos["total_recebido"] / contratos["valor_parcelado"].replace(0, np.nan)
    ).fillna(0).clip(0, 1)
    contratos["principal_realizado"] = contratos["total_recebido"] * contratos["frac_principal"]
    contratos["juros_realizados"] = contratos["total_recebido"] - contratos["principal_realizado"]

    # ==========================================================================
    # APLICAÇÃO DA OFUSCAÇÃO (MODO DEMONSTRAÇÃO)
    # ==========================================================================
    usuarios, clientes, contratos, movimentos = ofuscar_dados(usuarios, clientes, contratos, movimentos)

    return usuarios, clientes, contratos, movimentos


def filter_cancelled_contracts(contratos, movimentos):
    mov_count = movimentos.groupby("idcontrato").size().rename("mov_count")
    contratos = contratos.merge(mov_count, left_on="id", right_index=True, how="left").fillna({"mov_count": 0})
    contratos["mov_count"] = contratos["mov_count"].astype(int)
    contratos["cancelado_sem_movimento"] = (contratos["status"] == "Finalizado") & (contratos["mov_count"] == 0)
    valid = contratos[~contratos["cancelado_sem_movimento"]].copy()
    valid_mov = movimentos[movimentos["idcontrato"].isin(valid["id"])].copy()
    excluded = contratos[contratos["cancelado_sem_movimento"]].copy()
    return valid, valid_mov, excluded


def apply_period_filter(contratos, movimentos, start_date, end_date):
    st_ts = pd.Timestamp(start_date)
    en_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    movimentos_f = movimentos[
        movimentos["dtvenc"].notna() &
        (movimentos["dtvenc"] >= st_ts) &
        (movimentos["dtvenc"] < en_ts)
    ].copy()
    contratos_ids = movimentos_f["idcontrato"].unique()
    contratos_f = contratos[contratos["id"].isin(contratos_ids)].copy()
    return contratos_f, movimentos_f


def build_cashflow(movimentos, contratos, today, n_future=6):
    mo = movimentos.copy()
    sched = mo.dropna(subset=["dtvenc"]).assign(mes=lambda d: d["dtvenc"].dt.to_period("M")).groupby("mes")["parcela"].sum()
    real = mo.dropna(subset=["dtrecebimento"]).assign(mes=lambda d: d["dtrecebimento"].dt.to_period("M")).groupby("mes")["valorrecebido"].sum()

    all_meses = sorted(set(sched.index) | set(real.index))
    if all_meses:
        start = min(all_meses)
        end = max(start + n_future, max(all_meses))
        idx = pd.period_range(start, end, freq="M")
    else:
        idx = pd.period_range(pd.Period(today, freq="M"), periods=n_future, freq="M")

    df = pd.DataFrame({"mes": idx})
    df["programado"] = df["mes"].map(sched).fillna(0)
    df["realizado"] = df["mes"].map(real).fillna(0)

    recebido_total = mo["valorrecebido"].sum()
    vencido_aberto = mo.loc[mo["vencido"], "areceber"].sum()
    eficiencia = recebido_total / (recebido_total + vencido_aberto) if (recebido_total + vencido_aberto) > 0 else 0.0

    cut = today - pd.Timedelta(days=90)
    rec_recente = mo.loc[mo["dtrecebimento"] >= cut, "valorrecebido"].sum()
    vencido_recente = mo.loc[mo["vencido"] & (mo["dtvenc"] >= cut), "areceber"].sum()
    eficiencia_recente = rec_recente / (rec_recente + vencido_recente) if (rec_recente + vencido_recente) > 0 else eficiencia

    future_open = mo[mo["a_vencer"]].dropna(subset=["dtvenc"]).assign(mes=lambda d: d["dtvenc"].dt.to_period("M")).groupby("mes")["areceber"].sum()
    df["recebivel_programado"] = df["mes"].map(future_open).fillna(0)
    df["projetado"] = df["recebivel_programado"] * eficiencia_recente

    df["mes_label"] = df["mes"].astype(str)
    df["mes_ts"] = df["mes"].dt.to_timestamp()
    return df, eficiencia, eficiencia_recente


def build_backlog(movimentos, today):
    venc = movimentos[movimentos["vencido"]].copy()
    if venc.empty:
        return pd.DataFrame(columns=["faixa", "valor", "parcelas"])
    venc["dias"] = venc["dias_atraso"].clip(lower=1)
    bins = [0, 30, 60, 90, 180, 10_000_000]
    labels = ["1-30 dias", "31-60 dias", "61-90 dias", "91-180 dias", "181+ dias"]
    venc["faixa"] = pd.cut(venc["dias"], bins=bins, labels=labels, include_lowest=True)
    return venc.groupby("faixa", observed=False).agg(valor=("areceber", "sum"), parcelas=("id", "count")).reset_index()


def build_pdd(backlog_df):
    if backlog_df.empty:
        return pd.DataFrame(columns=["Faixa", "Valor em aberto", "% Provisao", "PDD"])
    rows = []
    for label, lo, hi, rate in PDD_FAIXAS:
        val = backlog_df.loc[backlog_df["faixa"] == label, "valor"].sum()
        if val > 0:
            rows.append({"Faixa": label, "Valor em aberto": val, "% Provisao": rate, "PDD": val * rate})
    return pd.DataFrame(rows)


def build_recovery_curve(movimentos):
    paid = movimentos[movimentos["status_pago"] & movimentos["dtvenc"].notna() & movimentos["dtrecebimento"].notna()].copy()
    if paid.empty:
        return pd.DataFrame(columns=["janela", "valor", "pct_acumulado"])
    paid["atraso"] = (paid["dtrecebimento"] - paid["dtvenc"]).dt.days.clip(lower=0)
    total = paid["valorrecebido"].sum()
    janelas = [0, 3, 7, 15, 30, 60, 90, 10000]
    nomes = ["em dia (0d)", "ate 3d", "ate 7d", "ate 15d", "ate 30d", "ate 60d", "ate 90d", "90+"]
    rows = []
    acc = 0.0
    for lo, hi, nome in zip(janelas[:-1], janelas[1:], nomes):
        v = paid[(paid["atraso"] >= lo) & (paid["atraso"] < hi)]["valorrecebido"].sum()
        acc += v
        rows.append({"janela": nome, "valor": v, "pct_acumulado": acc / total if total else 0})
    return pd.DataFrame(rows)


def build_dow_analysis(movimentos):
    rec = movimentos.dropna(subset=["dtrecebimento"]).copy()
    if rec.empty:
        return pd.DataFrame(columns=["dia", "valor", "pct", "parcelas"])
    rec["dow"] = rec["dtrecebimento"].dt.dayofweek
    nomes = ["Segunda", "Terca", "Quarta", "Quinta", "Sexta", "Sabado", "Domingo"]
    out = rec.groupby("dow").agg(valor=("valorrecebido", "sum"), parcelas=("id", "count")).reset_index()
    out["dia"] = out["dow"].map({i: n for i, n in enumerate(nomes)})
    out = out.sort_values("dow")
    out["pct"] = out["valor"] / out["valor"].sum()
    return out[["dia", "valor", "pct", "parcelas"]]


def build_monthly_return(movimentos):
    mo = movimentos.copy()
    if "juros_frac" not in mo.columns or "frac_principal" not in mo.columns:
        return pd.DataFrame()
    mo = mo[mo["dtvenc"].notna()].copy()
    mo["mes_venc"] = mo["dtvenc"].dt.to_period("M")

    g = mo.groupby("mes_venc").agg(programado=("parcela", "sum"), total_recebido=("valorrecebido", "sum"), parcelas=("id", "count")).reset_index()

    rec = mo[mo["dtrecebimento"].notna()].copy()
    rec["juros_rec"] = rec["valorrecebido"] * rec["juros_frac"]
    rec["princ_rec"] = rec["valorrecebido"] * rec["frac_principal"]
    rec_agg = rec.groupby("mes_venc")["juros_rec"].sum()
    princ_agg = rec.groupby("mes_venc")["princ_rec"].sum()
    g["juros_recebidos"] = g["mes_venc"].map(rec_agg).fillna(0)
    g["principal_recebido"] = g["mes_venc"].map(princ_agg).fillna(0)

    venc = mo[~mo["status_pago"]].copy()
    venc["juros_ab"] = venc["areceber"] * venc["juros_frac"]
    juros_ab = venc.groupby("mes_venc")["juros_ab"].sum()
    g["juros_aberto"] = g["mes_venc"].map(juros_ab).fillna(0)
    g["desconto"] = g["mes_venc"].map(mo.groupby("mes_venc")["desconto"].sum()).fillna(0)
    g["mes_ts"] = g["mes_venc"].dt.to_timestamp()
    g["mes_label"] = g["mes_venc"].astype(str)
    keep = ["mes_venc", "mes_ts", "mes_label", "programado", "principal_recebido", "juros_recebidos", "juros_aberto", "desconto", "parcelas"]
    return g[keep]


def build_monthly_efficiency(movimentos):
    mo = movimentos.copy()
    mo["mes_venc"] = mo["dtvenc"].dt.to_period("M")
    g = mo.dropna(subset=["mes_venc"]).groupby("mes_venc").agg(programado=("parcela", "sum"), recebido=("valorrecebido", "sum"), parcelas=("id", "count")).reset_index()
    g["eficiencia"] = g["recebido"] / g["programado"].replace(0, np.nan)
    g["mes_ts"] = g["mes_venc"].dt.to_timestamp()
    g["mes_label"] = g["mes_venc"].astype(str)
    return g


def build_new_contract_stats(contratos, start_date, end_date):
    """Calcula media, maximo e minimo por periodicidade no periodo analisado."""
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    recent = contratos[
        contratos["dtinicio"].notna()
        & (contratos["dtinicio"] >= start)
        & (contratos["dtinicio"] <= end)
        & (contratos["dtinicio"].dt.dayofweek < 6)
    ]

    def summarize(counts):
        counts = counts.astype(float)
        return {"Media": counts.mean(), "Maximo": int(counts.max()), "Minimo": int(counts.min())}

    daily_index = pd.date_range(start, end, freq="D")
    daily_index = daily_index[daily_index.dayofweek < 6]
    daily = recent["dtinicio"].dt.normalize().value_counts().reindex(daily_index, fill_value=0)

    week_start = start - pd.Timedelta(days=start.dayofweek)
    week_end = end - pd.Timedelta(days=end.dayofweek)
    weeks = pd.date_range(week_start, week_end, freq="7D")
    recent_week = recent["dtinicio"].dt.normalize() - pd.to_timedelta(recent["dtinicio"].dt.dayofweek, unit="D")
    weekly = recent_week.value_counts().reindex(weeks, fill_value=0)

    months = pd.period_range(start.to_period("M"), end.to_period("M"), freq="M")
    monthly = recent["dtinicio"].dt.to_period("M").value_counts().reindex(months, fill_value=0)

    def fortnight_start(values):
        normalized = values.dt.normalize()
        return normalized - pd.to_timedelta((normalized.dt.day > 15).astype(int) * 15, unit="D")

    first_fortnight = pd.date_range(start.replace(day=1), end.replace(day=1), freq="MS")
    fortnight_index = pd.DatetimeIndex([value for month in first_fortnight for value in (month, month + pd.Timedelta(days=15))])
    fortnight = fortnight_start(recent["dtinicio"]).value_counts().reindex(fortnight_index, fill_value=0)

    return {
        "Dia": summarize(daily),
        "Semana": summarize(weekly),
        "Quinzena": summarize(fortnight),
        "Mes": summarize(monthly),
    }


def build_agent_performance(contratos, movimentos, usuarios):
    co = contratos.copy()
    mo = movimentos.copy()
    rows = []
    for nome, g in co.groupby("usuario_nome"):
        mv = mo[mo["idcontrato"].isin(g["id"])]
        aberto = mv.loc[~mv["status_pago"], "areceber"].sum()
        recebido = mv["valorrecebido"].sum()
        venc_90 = mv.loc[mv["atraso_90"], "areceber"].sum()
        desconto = mv["desconto"].sum()
        pricipal = g["valor"].sum()
        a_receber = g["valor_parcelado"].sum()
        juros_prev = g["juros_previstos"].sum()
        juros_real = g["juros_realizados"].sum()
        rows.append({
            "Agente": nome, "Contratos": g["id"].nunique(), "Clientes": mv["idcliente"].nunique(),
            "Principal (R$)": pricipal, "A receber (R$)": a_receber, "Recebido (R$)": recebido,
            "Em aberto (R$)": aberto, "Desconto (R$)": desconto,
            "Eficiencia %": recebido / (recebido + aberto) if (recebido + aberto) else 0,
            "Vencido 90+ (R$)": venc_90, "Juros previstos (R$)": juros_prev, "Juros realizados (R$)": juros_real,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["Ticket medio (R$)"] = df["Principal (R$)"] / df["Contratos"].replace(0, 1)
        df = df.sort_values("Eficiencia %", ascending=False)
    return df


def build_vintage(movimentos, contratos):
    co = contratos.copy()
    if co["dtinicio"].notna().sum() == 0:
        return pd.DataFrame()
    co = co[co["dtinicio"].notna()].copy()
    co["coorte"] = co["dtinicio"].dt.to_period("M")

    mo = movimentos.dropna(subset=["dtinicio"]).copy()
    mo["coorte"] = mo["dtinicio"].dt.to_period("M")
    mo["dias_inicio"] = (mo["dtvenc"] - mo["dtinicio"]).dt.days.clip(lower=0)
    hoje = pd.Timestamp(date.today())

    janelas = [30, 45, 60, 75, 90]
    coortes = sorted(co["coorte"].unique())

    prog = mo[mo["dtvenc"].notna() & (mo["dtvenc"] <= hoje)].groupby(["coorte", "dias_inicio"])["parcela"].sum().reset_index()
    p_tot = mo[mo["dtvenc"].notna() & (mo["dtvenc"] <= hoje)].groupby("coorte")["parcela"].sum()
    rec = mo.dropna(subset=["dtrecebimento"]).assign(dias_rec=lambda d: (d["dtrecebimento"] - d["dtinicio"]).dt.days.clip(lower=0)).groupby(["coorte", "dias_rec"])["valorrecebido"].sum().reset_index().rename(columns={"dias_rec": "dias_inicio"})

    rows = []
    for c in coortes:
        p = prog[prog["coorte"] == c]
        r = rec[rec["coorte"] == c]
        base = p_tot.get(c, 0)
        for j in janelas:
            prog_j = p.loc[p["dias_inicio"] <= j, "parcela"].sum()
            rec_j = r.loc[r["dias_inicio"] <= j, "valorrecebido"].sum()
            rows.append({
                "coorte": str(c), "janela_dias": j, "programado_ate": prog_j, "recebido_ate": rec_j,
                "pct_programado": (prog_j / base) if base > 0 else 0,
                "pct_realizado_do_programado": (rec_j / prog_j) if prog_j > 0 else 0,
            })
    return pd.DataFrame(rows)


def build_priority_clients(movimentos):
    pend = movimentos[movimentos["vencido"] | movimentos["a_vencer"]].copy()
    if pend.empty:
        return pd.DataFrame()
    pend["dias"] = pend["dias_atraso"].clip(lower=0)
    out = pend.groupby("idcliente", as_index=False).agg(
        Cliente=("cliente", "first"), Agente=("usuario_nome", "first"), Segmento=("nome_estabelecimento_norm", "first"),
        Avaliacao=("avaliacao", "first"), Valor_em_aberto=("areceber", "sum"), Parcelas_em_aberto=("id", "count"), Maior_atraso=("dias", "max"),
    ).sort_values(["Valor_em_aberto", "Maior_atraso"], ascending=[False, False])
    
    out["Prioridade"] = np.select(
        [out["Maior_atraso"] >= 90, out["Maior_atraso"] >= 60, out["Maior_atraso"] >= 30],
        ["Critica", "Alta", "Media"], default="Baixa",
    )
    return out.rename(columns={"Valor_em_aberto": "Valor em aberto", "Parcelas_em_aberto": "Parcelas em aberto", "Maior_atraso": "Maior atraso"})


def build_action_plan(open_futuro, open_vencido, open_80_89, aberto_90, pdd):
    rows = []
    if aberto_90 > 0:
        rows.append(("Acompanhar inadimplencia critica (90+)", "Critica", f"{fmt_brl(aberto_90)} ja acima de 90 dias"))
    if open_80_89 > 0:
        rows.append(("Revisar contas proximas de 90+", "Alta", f"{fmt_brl(open_80_89)} em 80-89 dias"))
    if open_vencido > 0:
        rows.append(("Cobrar saldo vencido (backlog)", "Alta", f"{fmt_brl(open_vencido)} vencido em aberto"))
    if open_futuro > 0:
        rows.append(("Preparar recebiveis de curto prazo", "Media", f"{fmt_brl(open_futuro)} a vencer nos proximos 30 dias"))
    if pdd > 0:
        rows.append(("Constituir provisao (PDD)", "Media", f"PDD estimada de {fmt_brl(pdd)} sobre saldo vencido"))
    if not rows:
        rows.append(("Nenhuma acao urgente", "Baixa", "Sem valores criticos identificados"))
    return pd.DataFrame(rows, columns=["Acao", "Prioridade", "Detalhe"])


def build_insights(agente, backlog, eficiencia, recebido, aberto, aberto_90, pdd, best_agente, pior_agente, best_dow, worst_dow, contratos_novos_30d):
    ins = []
    if recebido > 0:
        ins.append(f"**Conversao de caixa:** {eficiencia:.1%} do valor que ja venceu foi recebido ({fmt_brl(recebido)} recebidos de {fmt_brl(recebido + aberto)} liquidados).")
    if aberto_90 > 0:
        ins.append(f"**Inadimplencia critica:** {fmt_brl(aberto_90)} ja esta acima de 90 dias de atraso, representando risco de perda relevante.")
    if pdd > 0:
        ins.append(f"**Provisao recomendada (PDD):** {fmt_brl(pdd)} deveria ser provisionado sobre o saldo vencido ({pdd / aberto:.1%} do total em aberto).")
    if not backlog.empty:
        venc_60 = backlog.loc[backlog["faixa"].isin(["61-90 dias", "91-180 dias", "181+ dias"]), "valor"].sum()
        if venc_60 > 0:
            ins.append(f"**Risco de envelhecimento:** {fmt_brl(venc_60)} do saldo vencido ja esta acima de 60 dias.")
    if best_agente is not None:
        ins.append(f"**Melhor eficiencia:** agente **{best_agente}** lidera a cobranca; use suas praticas como referencia.")
    if pior_agente is not None:
        ins.append(f"**Atencao operacional:** agente **{pior_agente}** apresenta a menor eficiencia de cobranca; avalie acompanhamento.")
    if best_dow is not None and worst_dow is not None:
        ins.append(f"**Sazonalidade de caixa:** {best_dow} concentra a maior entrada de recursos; {worst_dow} e o dia mais fraco.")
    if contratos_novos_30d > 0:
        ins.append(f"**Origemacao:** {contratos_novos_30d} contratos novos nos ultimos 30 dias; monitore a maturacao.")
    return ins


def main():
    st.set_page_config(page_title="Painel Financeiro - Microcredito", layout="wide", initial_sidebar_state="expanded")

    # Banner de Aviso de Demonstração
    if MODO_DEMONSTRACAO:
        st.warning("⚠️ **MODO DEMONSTRAÇÃO ATIVO:** Todos os nomes, valores financeiros e datas foram ofuscados/escalados para proteção de dados (LGPD). As proporções, eficiências e lógicas de análise permanecem matematicamente reais.")

    # ---------------- Sidebar ----------------
    st.sidebar.header("Filtros")
    today = date.today()
    start_date = st.sidebar.date_input("Data inicial", value=date(today.year, 1, 1), max_value=today, key="start_date")
    end_date = st.sidebar.date_input("Data final", value=today, max_value=today, key="end_date")
    apply_period = st.sidebar.checkbox("Aplicar filtro de período", value=True, help="Habilita aplicação do filtro por datas")
    st.sidebar.caption("Foca em contratos que estiveram ativos e parcelas com vencimento no período.")

    usuarios, clientes, contratos, movimentos = load_data()
    usuario_choices = sorted(usuarios["usuario"].dropna().unique())
    selected_users = st.sidebar.multiselect("Filtrar por agente", usuario_choices, default=usuario_choices)

    contratos, movimentos, contratos_excluidos = filter_cancelled_contracts(contratos, movimentos)
    if selected_users and len(selected_users) < len(usuario_choices):
        contratos = contratos[contratos["usuario_nome"].isin(selected_users)]
        movimentos = movimentos[movimentos["usuario_nome"].isin(selected_users)]
    
    contratos_total = contratos.copy()
    movimentos_total = movimentos.copy()
    
    period_start_ts = None
    period_end_ts = None
    if start_date and end_date:
        period_start_ts = pd.Timestamp(start_date)
        period_end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        if apply_period:
            contratos, movimentos = apply_period_filter(contratos, movimentos, start_date, end_date)

    period_excluded_ids = sorted(contratos_total.loc[~contratos_total["id"].isin(contratos["id"]), "id"].tolist()) if apply_period and period_start_ts is not None else []

    if len(contratos_excluidos) > 0:
        excl_ids = sorted(contratos_excluidos["id"].tolist())
        st.sidebar.info(f"Contratos excluidos (cancelados sem movimento): {len(contratos_excluidos)}\nIDs: {excl_ids}")
    if period_excluded_ids:
        st.sidebar.warning(f"Contratos fora do periodo ({len(period_excluded_ids)}): {period_excluded_ids}")

    # ---------------- Dados derivados ----------------
    hoje_ts = pd.Timestamp(today)

    aberto = movimentos.loc[~movimentos["status_pago"], "areceber"].sum()
    vencido = movimentos.loc[movimentos["vencido"], "areceber"].sum()
    a_vencer = movimentos.loc[movimentos["a_vencer"], "areceber"].sum()
    aberto_90 = movimentos.loc[movimentos["atraso_90"], "areceber"].sum()

    recebido = movimentos["valorrecebido"].sum()
    desconto_total = movimentos["desconto"].sum()

    parcelas_vencidas = len(movimentos.loc[movimentos["vencido"]])
    parcelas_pagas = len(movimentos.loc[movimentos["status_pago"]])
    parcelas_abertas = len(movimentos.loc[~movimentos["status_pago"]])
    programado = movimentos["areceber"].sum()

    contratos_ids = movimentos["idcontrato"].unique()
    contratos_periodo = contratos[contratos["id"].isin(contratos_ids)]

    principal = contratos_periodo["valor"].sum()
    principal_total = contratos_total["valor"].sum()
    a_receber = contratos_periodo["valor_parcelado"].sum()
    juros_previstos = contratos_periodo["juros_previstos"].sum()
    juros_realizados = contratos_periodo["juros_realizados"].sum()

    programado_total = movimentos_total["areceber"].sum()
    recebido_total = movimentos_total["valorrecebido"].sum()
    aberto_total = movimentos_total.loc[~movimentos_total["status_pago"], "areceber"].sum()
    backlog_total = movimentos_total.loc[movimentos_total["vencido"], "areceber"].sum()
    a_vencer_total = movimentos_total.loc[movimentos_total["a_vencer"], "areceber"].sum()
    aberto_90_total = movimentos_total.loc[movimentos_total["atraso_90"], "areceber"].sum()

    backlog_df_total = build_backlog(movimentos_total, hoje_ts)
    pdd_df_total = build_pdd(backlog_df_total)
    pdd_total = pdd_df_total["PDD"].sum() if not pdd_df_total.empty else 0.0

    backlog_df = build_backlog(movimentos, hoje_ts)
    pdd_df = build_pdd(backlog_df)
    pdd = pdd_df["PDD"].sum() if not pdd_df.empty else 0.0

    cf, eficiencia, eficiencia_recente = build_cashflow(movimentos, contratos, hoje_ts)
    recovery = build_recovery_curve(movimentos)
    dow = build_dow_analysis(movimentos)
    agentes = build_agent_performance(contratos, movimentos, usuarios)
    monthly_return = build_monthly_return(movimentos)
    monthly_eff = build_monthly_efficiency(movimentos)

    aberto_80_89 = movimentos.loc[movimentos["vencido"] & movimentos["dias_atraso"].between(80, 89), "areceber"].sum()
    futuro_30d = hoje_ts + pd.Timedelta(days=30)
    open_next_30 = movimentos.loc[movimentos["a_vencer"] & (movimentos["dtvenc"] <= futuro_30d), "areceber"].sum()

    if not agentes.empty:
        best_agente = agentes.loc[agentes["Eficiencia %"].idxmax(), "Agente"]
        pior_agente = agentes.loc[agentes["Eficiencia %"].idxmin(), "Agente"]
    else:
        best_agente = pior_agente = None
    if not dow.empty:
        best_dow = dow.loc[dow["valor"].idxmax(), "dia"]
        worst_dow = dow.loc[dow["valor"].idxmin(), "dia"]
    else:
        best_dow = worst_dow = None
    contratos_novos_30d = contratos.loc[contratos["dtinicio"] >= hoje_ts - pd.Timedelta(days=30), "id"].nunique()

    insights = build_insights(agentes, backlog_df, eficiencia, recebido, aberto, aberto_90, pdd, best_agente, pior_agente, best_dow, worst_dow, contratos_novos_30d)

    st.title("Painel Financeiro - Microcredito Diario")
    st.caption("Carteira atual (contratos2/movimentacoes2): microcredito de 90 parcelas diarias (Pix). Fonte: dbase.db.")

    tab_geral, tab_caixa, tab_agentes, tab_risco, tab_carteira, tab_controle, tab_rent, tab_dados = st.tabs(
        ["Visao geral", "Fluxo de caixa", "Agentes", "Riscos e cobranca", "Carteira", "Controle de Carteira", "Rentabilidade", "Dados"]
    )

    # =======================================================================
    # TAB 1 - VISAO GERAL
    # =======================================================================
    with tab_geral:
        st.subheader("Resumo executivo")
        with st.expander("Como ler esta aba", expanded=False):
            st.markdown("**Esta aba apresenta o resumo geral da carteira no período selecionado.**\n\n- **Programado**: valor total das parcelas com vencimento no período\n- **Recebido**: valor efetivamente coletado (em caixa)\n- **Em aberto**: parcelas que venceram mas ainda não foram pagas\n- **Inadimplência 90+**: saldo vencido há mais de 90 dias\n- **Vencido (backlog)**: total do saldo em atraso sobre o em aberto\n- **A vencer**: parcelas com vencimento futuro ainda não pagas\n- **PDD**: provisão para devedores duvidosos\n- **Portfólio completo**: métricas sem filtro de período")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Programado no período", fmt_brl(programado), f"{parcelas_vencidas} parcelas")
        c2.metric("Recebido no período", fmt_brl(recebido), f"{recebido/programado:.1%} do programado" if programado else "0")
        c3.metric("Em aberto no período", fmt_brl(aberto), f"{parcelas_abertas} parcelas")
        c4.metric("Inadimplência 90+", fmt_brl(aberto_90), f"{aberto_90 / aberto:.1%} do aberto" if aberto else "0")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Vencido (backlog)", fmt_brl(vencido), f"{vencido / aberto:.1%} do aberto" if aberto else "0")
        c6.metric("A vencer", fmt_brl(a_vencer), f"{a_vencer / aberto:.1%} do aberto" if aberto else "0")
        c7.metric("Contratos no período", len(contratos_periodo))
        c8.metric("PDD (provisão)", fmt_brl(pdd), f"{pdd / aberto:.1%} do aberto" if aberto else "0")

        st.markdown("---")
        st.markdown("#### Portfólio completo (sem filtro de período)")
        c9, c10, c11, c12 = st.columns(4)
        c9.metric("Total programado", fmt_brl(programado_total), f"{len(contratos_total)} contratos")
        c10.metric("Total recebido", fmt_brl(recebido_total), f"{recebido_total/programado_total:.1%} do total" if programado_total else "0")
        c11.metric("Total em aberto", fmt_brl(aberto_total), f"{aberto_total / programado_total:.1%} do programado" if programado_total else "0")
        c12.metric("PDD total", fmt_brl(pdd_total), f"{pdd_total/aberto_total:.1%} do aberto" if aberto_total else "0")

        eficiencia_periodo = recebido / programado if programado else 0
        st.info(f"**Período selecionado:** {start_date} a {end_date} | **Eficiência de cobrança:** {eficiencia_periodo:.1%} | **Parcelas:** {parcelas_vencidas} vencidas, {parcelas_pagas} pagas, {parcelas_abertas} em aberto")

        st.markdown("##### Composição do saldo em aberto por situação")
        sit_df = pd.DataFrame({"situacao": ["A vencer 0-30d", "A vencer 31-60d", "A vencer 60+", "Vencido 1-30d", "Vencido 31-60d", "Vencido 61-90d", "Vencido 90+d"], "valor": [0.0] * 7})
        if aberto > 0:
            pend = movimentos[~movimentos["status_pago"] & movimentos["dtvenc"].notna()].copy()
            pend["dias_situacao"] = np.where(pend["a_vencer"], (pend["dtvenc"] - hoje_ts).dt.days, pend["dias_atraso"].clip(lower=1))
            pend["bucket"] = np.select(
                [pend["a_vencer"] & (pend["dias_situacao"] <= 30), pend["a_vencer"] & (pend["dias_situacao"] <= 60), pend["a_vencer"],
                 ~pend["a_vencer"] & (pend["dias_situacao"] <= 30), ~pend["a_vencer"] & (pend["dias_situacao"] <= 60), ~pend["a_vencer"] & (pend["dias_situacao"] < 90), ~pend["a_vencer"]],
                ["A vencer 0-30d", "A vencer 31-60d", "A vencer 60+", "Vencido 1-30d", "Vencido 31-60d", "Vencido 61-90d", "Vencido 90+d"], default="Vencido 90+d",
            )
            sit = pend.groupby("bucket", as_index=False)["areceber"].sum()
            sit_df = sit_df.merge(sit, left_on="situacao", right_on="bucket", how="left", suffixes=("", "_v"))
            sit_df["valor"] = sit_df["valor"].fillna(0) + sit_df["areceber"].fillna(0)
            sit_df = sit_df[["situacao", "valor"]]

        sit_df["pct"] = (sit_df["valor"] / sit_df["valor"].sum() * 100).round(1)
        sit_df["label"] = sit_df.apply(lambda r: f"R$ {r['valor']:,.0f}\n({r['pct']}%)" if r['valor'] > 0 else "", axis=1)
        sit_df["cor"] = ["#72B7B2", "#4C78A8", "#BAB0AC", "#F58518", "#EECA3B", "#B279A2", "#E45756"]
        fig = px.bar(sit_df, x="situacao", y="valor", text="label", color="situacao", color_discrete_map=dict(zip(sit_df["situacao"], sit_df["cor"])), title="Composição do saldo em aberto no período", labels={"situacao": "", "valor": "Valor (R$)"})
        fig.update_traces(texttemplate="%{text}", textposition="outside")
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20), showlegend=False, xaxis_tickangle=-25, font=dict(size=12))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        st.markdown("##### Insights automaticos")
        for ins in insights:
            st.markdown(f"- {ins}")

        st.markdown("---")
        st.markdown("##### Análise sênior: panorama do mercado")
        with st.expander("Ver análise completa", expanded=False):
            st.markdown(f"""
            **Panorama da operação vs mercado (microcrédito brasileiro):**

            | Indicador | Sua Operação | Mercado (referência) | Avaliação |
            |-----------|-------------|---------------------|-----------|
            | Eficiência de cobrança | ~{eficiencia_periodo:.0%} | 60-75% | {'✅ Dentro da média' if 0.60 <= eficiencia_periodo <= 0.75 else '⚠️ Atenção'} |
            | Inadimplência 90+ | ~{(aberto_90/programado):.0%} do programado | 10-15% | {'✅ Adequado' if (aberto_90/programado) < 0.15 else '⚠️ Acima do mercado'} |
            | Ticket médio | {fmt_brl(contratos['valor'].mean() if len(contratos) else 0)} | R$ 1.500-3.000 | ✅ Adequado |
            | Prazo | 90 parcelas diárias | 30-90 parcelas | ✅ Padrão |

            **Perfis para INVESTIR (maior eficiência de retorno):**
            - Clientes do agente com maior eficiência atual.
            - Parcelas pagas **em dia ou até 3 dias** após vencimento.
            - Segmentos com alta eficiência de recebimento.

            **Perfis para EVITAR (maior risco de perda):**
            - Agentes com eficiência de cobrança abaixo de 60%.
            - Concentração alta em poucos clientes.
            - Faixa etária com menor eficiência de pagamento.

            **Recomendações:**
            1. **Urgente**: investigar agentes com eficiência crítica.
            2. **Reforçar cobrança**: clientes com 90+ dias.
            3. **Diversificar**: reduzir concentração nos top clientes.
            """)

    # =======================================================================
    # TAB 2 - FLUXO DE CAIXA
    # =======================================================================
    with tab_caixa:
        st.subheader("Fluxo de caixa: programado x realizado")
        with st.expander("Como ler esta aba", expanded=False):
            st.markdown("**Esta aba analisa o fluxo de caixa e a eficiência de cobrança.**\n\n- **Gráfico principal**: compara o valor programado com o realizado\n- **Projeção de recebíveis**: use os sliders para simular cenários\n- **Eficiência por mês**: mostra quanto foi coletado do que venceu\n- **Sazonalidade**: identifica dias da semana com maior/menor recebimento\n- **Velocidade de recuperação**: quando o cliente paga")
        
        st.markdown(f"**Eficiência histórica de cobrança:** {eficiencia:.1%} (12 meses) / **{eficiencia_recente:.1%}** (últimos 3 meses).")

        hist = cf[cf["mes"] <= pd.Period(hoje_ts, freq="M")]
        fut = cf[cf["mes"] > pd.Period(hoje_ts, freq="M")]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=hist["mes_ts"], y=hist["programado"], name="Programado (vencimentos)", marker_color="#BAB0AC"))
        fig.add_trace(go.Bar(x=hist["mes_ts"], y=hist["realizado"], name="Realizado (caixa)", marker_color=COLORS["azul"]))
        fig.add_trace(go.Bar(x=fut["mes_ts"], y=fut["recebivel_programado"], name="Recebiveis futuros", marker_color="#B279A2"))
        fig.add_trace(go.Bar(x=fut["mes_ts"], y=fut["projetado"], name="Projecao (x eficiencia)", marker_color=COLORS["verde"]))
        fig.update_layout(barmode="group", title="Programado x realizado por mês (e projeção)", xaxis_title="Mês", yaxis_title="Valor (R$)", height=420, margin=dict(l=20, r=20, t=50, b=20), legend=dict(orientation="h", y=1.18, x=0, xanchor="left", font=dict(size=11), bgcolor="rgba(255,255,255,0.8)"), xaxis_tickformat="%b/%y", font=dict(size=12))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        st.markdown("##### Projeção de recebíveis (cenário ajustável)")
        col1, col2 = st.columns(2)
        efic_cenario = col1.slider("Eficiencia de cobranca do cenario (%)", 30, 100, int(round(eficiencia_recente * 100)))
        rec_vencido = col2.slider("Recuperacao esperada do saldo vencido (%)", 0, 100, int(round(eficiencia_recente * 100)))

        proj = fut[["mes_label", "recebivel_programado"]].copy()
        proj["previsao"] = proj["recebivel_programado"] * (efic_cenario / 100)
        proj = proj.rename(columns={"mes_label": "Mês", "recebivel_programado": "Programado (R$)", "previsao": "Previsão (R$)"})
        proj["Previsão acumulada (R$)"] = proj["Previsão (R$)"].cumsum()
        proj["% eficiência"] = efic_cenario
        proj["Programado (R$)"] = proj["Programado (R$)"].round(2)
        proj["Previsão (R$)"] = proj["Previsão (R$)"].round(2)
        proj["Previsão acumulada (R$)"] = proj["Previsão acumulada (R$)"].round(2)
        if not proj.empty:
            show(proj)
        else:
            st.info("Sem projeção de recebíveis futuros no período selecionado.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Backlog vencido", fmt_brl(vencido), f"{vencido / aberto:.1%} do aberto | recuperável {fmt_brl(vencido * rec_vencido / 100)}" if aberto else "0")
        c2.metric("Recebíveis futuros (90d)", fmt_brl(fut["recebivel_programado"].head(3).sum()), f"{fut['recebivel_programado'].head(3).sum() / programado:.1%} do programado" if programado else "0")
        c3.metric("Previsão próximo mês", fmt_brl(proj.iloc[0]["Previsão (R$)"] if not proj.empty else 0), f"{efic_cenario}% eficiência" if not proj.empty else "")

        st.markdown("##### Eficiencia de cobranca por mes (recebido / programado de vencimento)")
        if not monthly_eff.empty:
            eff_plot = monthly_eff.copy()
            eff_plot["eficiencia"] = eff_plot["eficiencia"] * 100
            fige = go.Figure()
            fige.add_trace(go.Bar(x=eff_plot["mes_ts"], y=eff_plot["programado"], name="Programado (vencido)", marker_color="#BAB0AC"))
            fige.add_trace(go.Bar(x=eff_plot["mes_ts"], y=eff_plot["recebido"], name="Realizado (caixa)", marker_color=COLORS["azul"]))
            fige.add_trace(go.Scatter(x=eff_plot["mes_ts"], y=eff_plot["eficiencia"], name="Eficiencia (%)", yaxis="y2", mode="lines+markers+text", text=eff_plot["eficiencia"].round(0).astype(str) + "%", marker_color=COLORS["vermelho"], textposition="top center"))
            fige.update_layout(title="Programado vs realizado e eficiência por mês de vencimento", height=380, margin=dict(l=20, r=20, t=50, b=20), xaxis_title="Mês de vencimento", yaxis_title="Valor (R$)", yaxis2=dict(title="Eficiência (%)", overlaying="y", side="right", range=[0, 105]), legend=dict(orientation="h", y=1.18, x=0, xanchor="left", font=dict(size=11), bgcolor="rgba(255,255,255,0.8)"), xaxis_tickformat="%b/%y", font=dict(size=12))
            st.plotly_chart(fige, width="stretch", config={"displayModeBar": False})
        else:
            st.info("Sem dados de vencimento no periodo.")

        st.markdown("---")
        st.subheader("Comportamento de pagamento")
        colA, colB = st.columns(2)
        with colA:
            st.markdown("##### Recebimento por dia da semana")
            if not dow.empty:
                dow_plot = dow.copy()
                dow_plot["pct"] = dow_plot["pct"] * 100
                figd = px.bar(dow_plot, x="dia", y="valor", text="pct", color="dia", color_discrete_sequence=SEQUENCE, labels={"dia": "", "valor": "Recebido (R$)"}, title="Sazonalidade do caixa (seg a dom)")
                figd.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                figd.update_layout(height=340, margin=dict(l=20, r=20, t=50, b=20), showlegend=False, font=dict(size=12))
                st.plotly_chart(figd, width="stretch", config={"displayModeBar": False})
            else:
                st.info("Sem recebimentos no periodo.")

        with colB:
            st.markdown("##### Velocidade de recuperação (dias após vencimento)")
            if not recovery.empty:
                rec_plot = recovery.copy()
                rec_plot["pct_acumulado"] = rec_plot["pct_acumulado"] * 100
                rec_plot["pct_individual"] = (rec_plot["valor"] / rec_plot["valor"].sum() * 100).round(1)
                rec_plot["label"] = rec_plot.apply(lambda r: f"R$ {r['valor']:,.0f}\n({r['pct_individual']}%)" if r['valor'] > 0 else "", axis=1)
                figr = px.bar(rec_plot, x="janela", y="valor", color="janela", color_discrete_sequence=SEQUENCE, text="label", labels={"janela": "Janela após vencimento", "valor": "Valor recebido (R$)"}, title="Quando o cliente paga")
                figr.update_traces(texttemplate="%{text}", textposition="outside")
                figr.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20), showlegend=False, font=dict(size=12))
                st.plotly_chart(figr, width="stretch", config={"displayModeBar": False})
                st.caption("Acumulado: " + " → ".join([f"{r['janela']}: {r['pct_acumulado']:.1f}%" for _, r in rec_plot.iterrows()]))
            else:
                st.info("Sem dados de pagamento no periodo.")

    # =======================================================================
    # TAB 3 - AGENTES
    # =======================================================================
    with tab_agentes:
        st.subheader("Performance por agente")
        with st.expander("Como ler esta aba", expanded=False):
            st.markdown("**Esta aba avalia a performance de cada agente de cobrança.**\n\n- **Recebido por agente**: valor total coletado\n- **Saldo em aberto**: quanto cada agente tem em atraso\n- **Eficiência**: % do programado que foi efetivamente recebido")
        
        if agentes.empty:
            st.info("Sem dados de agentes no filtro atual.")
        else:
            view = agentes.copy()
            for c in view.columns:
                if "%" in c:
                    view[c] = (view[c] * 100).round(1)
                elif "R$" in c:
                    view[c] = view[c].round(2)
            show(view)

            c1, c2 = st.columns(2)
            with c1:
                top = agentes.sort_values("Recebido (R$)", ascending=False)
                fig = px.bar(top, x="Agente", y="Recebido (R$)", color="Eficiencia %", text="Eficiencia %", color_continuous_scale="Blues", labels={"Agente": "", "Recebido (R$)": "Recebido (R$)", "Eficiencia %": "Eficiencia"}, title="Recebido por agente (cor = eficiencia)")
                fig.update_traces(texttemplate="%{text:.0%}", textposition="outside")
                fig.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

            with c2:
                fig2 = px.bar(top, x="Agente", y=["Em aberto (R$)", "Vencido 90+ (R$)"], barmode="group", labels={"value": "Valor (R$)", "variable": "", "Agente": ""}, title="Saldo em aberto e inadimplencia por agente", color_discrete_sequence=[COLORS["laranja"], COLORS["vermelho"]])
                fig2.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20), legend=dict(orientation="h", y=1.12, x=0, xanchor="left", font=dict(size=11), bgcolor="rgba(255,255,255,0.8)"), font=dict(size=12))
                st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})

            custos_info = "; ".join([f"{u['usuario']}: comissao {u['comissao']}% | custo fixo {fmt_brl(u['custo_fixo'] or 0)}" for _, u in usuarios[usuarios["custo_fixo"].fillna(0) > 0].iterrows()])
            if custos_info:
                st.markdown(f"**Custos e comissoes (cadastro):** {custos_info}")

    # =======================================================================
    # TAB 4 - RISCOS E COBRANCA
    # =======================================================================
    with tab_risco:
        st.subheader("Backlog (saldo vencido) e provisão")
        with st.expander("Como ler esta aba", expanded=False):
            st.markdown("**Esta aba foca no risco de inadimplência e ações de cobrança.**\n\n- **Total vencido**: saldo em atraso total\n- **PDD**: provisão para devedores duvidosos\n- **Carteira líquida**: aberto menos PDD\n- **Saldo crítico 90+**: valores com mais de 90 dias de atraso")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total vencido", fmt_brl(vencido))
        c2.metric("PDD (provisao)", fmt_brl(pdd))
        c3.metric("Carteira liquida (aberto - PDD)", fmt_brl(aberto - pdd))
        c4.metric("Saldo critico 90+", fmt_brl(aberto_90))

        if not backlog_df.empty:
            fig = px.bar(backlog_df, x="faixa", y="valor", text="valor", color="faixa", color_discrete_sequence=SEQUENCE, title="Saldo vencido por faixa de atraso", labels={"faixa": "Faixa de atraso", "valor": "Valor vencido (R$)"})
            fig.update_traces(texttemplate="R$ %{text:,.0f}", textposition="outside")
            fig.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

            pdd_view = pdd_df.copy()
            pdd_view["PDD"] = pdd_view["PDD"].round(2)
            pdd_view["Valor em aberto"] = pdd_view["Valor em aberto"].round(2)
            pdd_view["% Provisao"] = (pdd_view["% Provisao"] * 100).astype(int).astype(str) + "%"
            st.markdown("##### Composicao da PDD")
            show(pdd_view)
        else:
            st.info("Nao ha saldo vencido no periodo filtrado.")

        st.markdown("---")
        st.subheader("Plano de acao sugerido")
        action = build_action_plan(open_next_30, vencido, aberto_80_89, aberto_90, pdd)
        show(action)

        st.markdown("---")
        st.subheader("Clientes prioritarios para cobranca")
        prio = build_priority_clients(movimentos)
        if not prio.empty:
            prio_view = prio.copy()
            prio_view["Valor em aberto"] = prio_view["Valor em aberto"].round(2)
            prio_view["Maior atraso"] = prio_view["Maior atraso"].astype(int)
            show(prio_view.head(20))

            top10 = prio_view.head(10)
            fig = px.bar(top10, x="Cliente", y="Valor em aberto", color="Prioridade", text="Valor em aberto", color_discrete_map={"Critica": COLORS["vermelho"], "Alta": COLORS["laranja"], "Media": COLORS["amarelo"], "Baixa": COLORS["verde"]}, title="Top 10 clientes por valor em aberto", labels={"Cliente": "", "Valor em aberto": "Valor em aberto (R$)", "Prioridade": "Prioridade"})
            fig.update_traces(texttemplate="R$ %{text:,.0f}", textposition="outside")
            fig.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20), xaxis_tickangle=-30)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        else:
            st.info("Nenhum cliente com saldo em aberto.")

    # =======================================================================
    # TAB 5 - CARTEIRA
    # =======================================================================
    with tab_carteira:
        st.subheader("Originação e maturação")
        with st.expander("Como ler esta aba", expanded=False):
            st.markdown("**Esta aba analisa a composição e evolução da carteira.**\n\n- **Novos contratos**: quantidade e valor dos contratos originados\n- **Maturação por coorte**: acompanha como grupos de contratos evoluem\n- **Segmentos**: distribuição por estabelecimento")

        new_ct = contratos[contratos["dtinicio"].notna()].copy()
        if apply_period and period_start_ts is not None and period_end_ts is not None:
            new_ct = new_ct[(new_ct["dtinicio"] >= period_start_ts) & (new_ct["dtinicio"] < period_end_ts)].copy()
        if not new_ct.empty:
            new_ct["mes"] = new_ct["dtinicio"].dt.to_period("M").dt.to_timestamp()
            by_month = new_ct.groupby("mes").size().rename("qtd").reset_index()
            by_month_val = new_ct.groupby("mes")["valor"].sum().rename("valor").reset_index()

            stats_start = period_start_ts if apply_period and period_start_ts is not None else new_ct["dtinicio"].min()
            stats_end = period_end_ts - pd.Timedelta(days=1) if apply_period and period_end_ts is not None else new_ct["dtinicio"].max()
            novos_stats = build_new_contract_stats(new_ct, stats_start, stats_end)
            novos_stats_view = pd.DataFrame(novos_stats).T.reset_index().rename(columns={"index": "Periodicidade"})

            c1, c2 = st.columns(2)
            c1.metric("Contratos no periodo", len(new_ct))
            c2.metric("Ticket medio", fmt_brl(new_ct["valor"].mean()))
            st.markdown("##### Novos contratos por periodicidade")
            show(novos_stats_view.round({"Media": 1, "Maximo": 0, "Minimo": 0}))

            fig = go.Figure()
            fig.add_trace(go.Bar(x=by_month["mes"], y=by_month["qtd"], name="Qtd contratos", marker_color=COLORS["azul"], yaxis="y"))
            fig.add_trace(go.Scatter(x=by_month_val["mes"], y=by_month_val["valor"], name="Valor originado (R$)", mode="lines+markers", marker_color=COLORS["laranja"], yaxis="y2"))
            fig.update_layout(title="Novos contratos por mês (quantidade e valor)", height=360, margin=dict(l=20, r=20, t=50, b=20), yaxis=dict(title="Qtd"), yaxis2=dict(title="Valor (R$)", overlaying="y", side="right"), legend=dict(orientation="h", y=1.18, x=0, xanchor="left", font=dict(size=11), bgcolor="rgba(255,255,255,0.8)"), font=dict(size=12))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        st.markdown("---")
        st.subheader("Maturacao por coorte (vintage)")
        vintage = build_vintage(movimentos, contratos)
        if apply_period and period_start_ts is not None:
            try:
                min_coorte = pd.Period(period_start_ts, freq="M")
                vintage = vintage[vintage["coorte"].apply(lambda c: pd.Period(c) >= min_coorte)]
            except Exception:
                pass
        if not vintage.empty:
            mat = vintage.pivot_table(index="coorte", columns="janela_dias", values="pct_programado") * 100
            mat = mat.sort_index()
            st.markdown("**Maturacao: % do cronograma contratual que ja venceu** (por dias desde o inicio):")
            show(mat.round(1))

            eff = vintage.pivot_table(index="coorte", columns="janela_dias", values="pct_realizado_do_programado") * 100
            eff = eff.sort_index()
            st.markdown("**Eficiencia de recebimento: % do vencido que ja foi recebido** (valores acima de 100% indicam pagamento adiantado):")
            show(eff.round(1))

            fig = px.line(vintage, x="janela_dias", y="pct_realizado_do_programado", color="coorte", markers=True, labels={"janela_dias": "Dias desde o início", "pct_realizado_do_programado": "% recebido do vencido", "coorte": "Mês de início"}, title="Eficiência de recebimento por coorte (maturação)")
            fig.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20), legend=dict(orientation="h", y=1.18, x=0, xanchor="left", font=dict(size=11), bgcolor="rgba(255,255,255,0.8)"), font=dict(size=12))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        else:
            st.info("Dados insuficientes para curvas de maturacao.")

        st.markdown("---")
        st.subheader("Segmentos (estabelecimento)")
        seg = movimentos.groupby("nome_estabelecimento_norm").agg(Recebido=("valorrecebido", "sum"), Em_aberto=("areceber", lambda s: s[movimentos.loc[s.index, "status_pago"] == False].sum()), Clientes=("idcliente", "nunique")).reset_index().rename(columns={"nome_estabelecimento_norm": "Segmento"})
        seg["Total"] = seg["Recebido"] + seg["Em_aberto"]
        seg["Eficiencia %"] = seg["Recebido"] / seg["Total"].replace(0, 1)
        seg = seg.sort_values("Total", ascending=False)
        view_seg = seg.copy()
        view_seg["Eficiencia %"] = (view_seg["Eficiencia %"] * 100).round(1)
        view_seg[["Recebido", "Em_aberto", "Total"]] = view_seg[["Recebido", "Em_aberto", "Total"]].round(2)
        show(view_seg)

        if not seg.empty:
            top_seg = seg.head(8)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=top_seg["Segmento"], y=top_seg["Recebido"], name="Recebido", marker_color=COLORS["azul"]))
            fig.add_trace(go.Bar(x=top_seg["Segmento"], y=top_seg["Em_aberto"], name="Em aberto", marker_color=COLORS["laranja"]))
            fig.update_layout(barmode="stack", title="Top segmentos: recebido x em aberto", height=360, margin=dict(l=20, r=20, t=50, b=20), legend=dict(orientation="h", y=1.12, x=0, xanchor="left", font=dict(size=11), bgcolor="rgba(255,255,255,0.8)"), xaxis_tickangle=-30, font=dict(size=12))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        st.markdown("---")
        st.subheader("Contratos com menor recebimento (foco de cobranca)")
        worst = contratos[contratos["valor_parcelado"] > 0].sort_values("percentual_recebido").head(15)
        worst_view = worst[["id", "status", "usuario_nome", "cliente", "valor", "valor_parcelado",
                            "total_recebido", "aberto_nao_pago", "percentual_recebido"]].copy()
        worst_view["percentual_recebido"] = (worst_view["percentual_recebido"] * 100).round(2)
        show(worst_view.rename(columns={
            "id": "Contrato", "status": "Status", "usuario_nome": "Agente", "cliente": "Cliente",
            "valor": "Principal", "valor_parcelado": "A receber", "total_recebido": "Recebido",
            "aberto_nao_pago": "Em aberto", "percentual_recebido": "% Recebido",
        }))

    # =======================================================================
    # TAB 6 - CONTROLE DE CARTEIRA
    # =======================================================================
    with tab_controle:
        st.subheader("Controle de carteira - contratos e exclusões")
        with st.expander("Como ler esta aba", expanded=False):
            st.markdown("**Esta aba mostra o controle administrativo dos contratos.**\n\n- **Resumo por situação**: total de contratos Vigentes vs Finalizados\n- **Contratos excluídos**: cancelados sem movimentação\n- **Contratos fora do período**: não ativos no período selecionado")

        st.markdown("##### Resumo por situacao (portfolio completo)")
        status_df = contratos_total.groupby("status").agg(Contratos=("id", "count"), Principal=("valor", "sum"), A_receber=("valor_parcelado", "sum")).reset_index()
        status_df["Principal"] = status_df["Principal"].round(2)
        status_df["A_receber"] = status_df["A_receber"].round(2)
        st.dataframe(status_df, hide_index=True, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total contratos (validos)", len(contratos_total))
        c2.metric("Principal total (portfolio)", fmt_brl(principal_total))
        c3.metric("Principal no periodo", fmt_brl(principal))

        st.markdown("---")
        st.markdown("##### Contratos excluidos (cancelados sem movimento)")
        if len(contratos_excluidos) > 0:
            excl_view = contratos_excluidos[["id", "dtinicio", "dtfim", "status", "valor", "valor_parcelado"]].copy()
            excl_view["valor"] = excl_view["valor"].round(2)
            excl_view["valor_parcelado"] = excl_view["valor_parcelado"].fillna(0).round(2)
            show(excl_view.rename(columns={"id": "Contrato", "dtinicio": "Dt inicio", "dtfim": "Dt fim", "status": "Status", "valor": "Principal (R$)", "valor_parcelado": "A receber (R$)"}))
            st.caption(f"Total excluido: {fmt_brl(contratos_excluidos['valor'].sum())} em {len(contratos_excluidos)} contratos.")
        else:
            st.info("Nenhum contrato excluido.")

        st.markdown("---")
        st.markdown("##### Contratos fora do periodo selecionado")
        if period_excluded_ids:
            period_excl = contratos_total[contratos_total["id"].isin(period_excluded_ids)].copy()
            period_view = period_excl[["id", "dtinicio", "dtfim", "status", "valor", "valor_parcelado"]].copy()
            period_view["valor"] = period_view["valor"].round(2)
            period_view["valor_parcelado"] = period_view["valor_parcelado"].fillna(0).round(2)
            show(period_view.rename(columns={"id": "Contrato", "dtinicio": "Dt inicio", "dtfim": "Dt fim", "status": "Status", "valor": "Principal (R$)", "valor_parcelado": "A receber (R$)"}))
            st.caption(f"Total fora do periodo: {fmt_brl(period_excl['valor'].sum())} em {len(period_excl)} contratos.")
        else:
            st.info("Nenhum contrato fora do periodo (ou filtro de periodo desativado).")

    # =======================================================================
    # TAB 7 - RENTABILIDADE
    # =======================================================================
    with tab_rent:
        st.subheader("Rentabilidade do produto")
        with st.expander("Como ler esta aba", expanded=False):
            st.markdown("**Esta aba analisa a rentabilidade e composição do retorno.**\n\n- **Principal**: valor total emprestado\n- **Juros previstos**: receita de juros esperada\n- **Juros realizados**: juros efetivamente recebidos em caixa\n- **Descontos**: abatimentos concedidos")

        frac_pond = (contratos["valor"].sum() / contratos["valor_parcelado"].sum()) if contratos["valor_parcelado"].sum() else 0
        juros_aberto_est = aberto * (1 - frac_pond)
        principal_aberto_est = aberto * frac_pond

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Principal", fmt_brl(principal))
        c2.metric("Juros previstos", fmt_brl(juros_previstos), f"{juros_previstos / principal:.1%} do principal" if principal else "")
        c3.metric("Juros realizados (caixa)", fmt_brl(juros_realizados))
        c4.metric("Descontos concedidos", fmt_brl(desconto_total), f"{desconto_total / recebido:.1%} do recebido" if recebido else "")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Principal em aberto (est.)", fmt_brl(principal_aberto_est))
        c6.metric("Juros em aberto (est.)", fmt_brl(juros_aberto_est))
        c7.metric("Retorno realizado", f"{recebido / principal:.1%}" if principal else "0")
        c8.metric("Ticket medio", fmt_brl(contratos["valor"].mean()) if len(contratos) else "0")

        st.markdown(f"**Leitura:** para cada {fmt_brl(1)} emprestado, o contrato previa receber {fmt_brl(1 + juros_previstos / principal) if principal else 0} ({juros_previstos / principal:.1%} de juros). Do total, {fmt_brl(recebido)} ja retornou em caixa ({recebido / principal:.1%} do principal).")

        st.markdown("##### Juros e desconto por agente")
        if not agentes.empty:
            rent = agentes[["Agente", "Principal (R$)", "Juros previstos (R$)", "Juros realizados (R$)", "Desconto (R$)", "Eficiencia %"]].copy()
            rent["Juros realizados % do previsto"] = rent["Juros realizados (R$)"] / rent["Juros previstos (R$)"].replace(0, 1)
            rent["Juros realizados % do previsto"] = (rent["Juros realizados % do previsto"] * 100).round(1)
            rent["Eficiencia %"] = (rent["Eficiencia %"] * 100).round(1)
            show(rent)

            fig = px.bar(agentes.sort_values("Juros realizados (R$)"), x="Agente", y=["Juros previstos (R$)", "Juros realizados (R$)"], barmode="group", labels={"value": "Valor (R$)", "variable": "", "Agente": ""}, title="Juros previstos x realizados por agente", color_discrete_sequence=[COLORS["azul"], COLORS["verde"]])
            fig.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20), legend=dict(orientation="h", y=1.12, x=0, xanchor="left", font=dict(size=11), bgcolor="rgba(255,255,255,0.8)"), font=dict(size=12))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        st.markdown("##### Concentracao da carteira (top clientes por valor)")
        conc = movimentos.groupby("idcliente").agg(Cliente=("cliente", "first"), Valor_contratado=("valor_parcelado", "first"), Recebido=("valorrecebido", "sum"), Em_aberto=("areceber", lambda s: s[movimentos.loc[s.index, "status_pago"] == False].sum())).reset_index()
        conc["Total"] = conc["Recebido"] + conc["Em_aberto"]
        conc = conc.sort_values("Total", ascending=False)
        conc["Participacao"] = conc["Total"] / conc["Total"].sum() if conc["Total"].sum() else 0
        conc_view = conc.head(15).copy()
        conc_view["Participacao"] = (conc_view["Participacao"] * 100).round(1)
        conc_view[["Valor_contratado", "Recebido", "Em_aberto", "Total"]] = conc_view[["Valor_contratado", "Recebido", "Em_aberto", "Total"]].round(2)
        show(conc_view.rename(columns={"Valor_contratado": "A receber (R$)", "Recebido": "Recebido (R$)", "Em_aberto": "Em aberto (R$)", "Total": "Total (R$)", "Participacao": "Participacao (%)"}))
        top10_share = conc.head(10)["Total"].sum() / conc["Total"].sum() if conc["Total"].sum() else 0
        st.markdown(f"**Concentracao:** os 10 maiores clientes representam **{top10_share:.1%}** do valor movimentado.")

        st.markdown("##### Retorno por mes de vencimento (recebido x inadimplente de juros)")
        if not monthly_return.empty:
            mr = monthly_return.copy()
            mr = mr.sort_values("mes_ts")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=mr["mes_ts"], y=mr["principal_recebido"], name="Principal recebido", marker_color=COLORS["azul"]))
            fig.add_trace(go.Bar(x=mr["mes_ts"], y=mr["juros_recebidos"], name="Juros recebidos (caixa)", marker_color=COLORS["verde"]))
            fig.add_trace(go.Bar(x=mr["mes_ts"], y=mr["juros_aberto"], name="Juros inadimplentes (est.)", marker_color=COLORS["vermelho"]))
            fig.add_trace(go.Bar(x=mr["mes_ts"], y=mr["desconto"], name="Descontos concedidos", marker_color=COLORS["laranja"]))
            fig.update_layout(barmode="group", title="Composição do retorno por mês", height=400, margin=dict(l=20, r=20, t=50, b=20), legend=dict(orientation="h", y=1.18, x=0, xanchor="left", font=dict(size=11), bgcolor="rgba(255,255,255,0.8)"), xaxis_title="Mês de vencimento", yaxis_title="Valor (R$)", xaxis_tickformat="%b/%y", font=dict(size=12))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        else:
            st.info("Sem dados de vencimento no periodo.")

        st.markdown("---")
        st.subheader("Perfil demografico da carteira")
        col1, col2 = st.columns(2)
        with col1:
            gen = movimentos.groupby("genero_cat").agg(Recebido=("valorrecebido", "sum"), Em_aberto=("areceber", lambda s: s[movimentos.loc[s.index, "status_pago"] == False].sum())).reset_index()
            gen["Total"] = gen["Recebido"] + gen["Em_aberto"]
            fig = px.bar(gen, x="genero_cat", y="Total", color="genero_cat", color_discrete_sequence=SEQUENCE, text="Total", labels={"genero_cat": "", "Total": "Valor (R$)"}, title="Valor movimentado por gênero")
            fig.update_traces(texttemplate="R$ %{text:,.0f}", textposition="outside")
            fig.update_layout(height=340, margin=dict(l=20, r=20, t=50, b=20), showlegend=False, font=dict(size=12))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        with col2:
            age = movimentos.groupby("faixa_idade").agg(Recebido=("valorrecebido", "sum"), Em_aberto=("areceber", lambda s: s[movimentos.loc[s.index, "status_pago"] == False].sum())).reset_index()
            age["Total"] = age["Recebido"] + age["Em_aberto"]
            age["Eficiencia %"] = age["Recebido"] / age["Total"].replace(0, 1)
            fig = px.bar(age, x="faixa_idade", y="Total", color="Eficiencia %", color_continuous_scale="Blues", text="Total", labels={"faixa_idade": "Faixa etária", "Total": "Valor (R$)", "Eficiencia %": "Eficiência"}, title="Valor movimentado por faixa etária")
            fig.update_traces(texttemplate="R$ %{text:,.0f}", textposition="outside")
            fig.update_layout(height=340, margin=dict(l=20, r=20, t=50, b=20), xaxis_tickangle=-20, font=dict(size=12))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # =======================================================================
    # TAB 8 - DADOS
    # =======================================================================
    with tab_dados:
        st.subheader("Dados brutos")
        if st.checkbox("Mostrar movimentacoes completas", value=False):
            show(movimentos)
        if st.checkbox("Mostrar contratos (periodo)", value=False):
            show(contratos)
        if st.checkbox("Mostrar todos os contratos (portfolio completo)", value=False):
            show(contratos_total)


if __name__ == "__main__":
    main()