import sqlite3
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

# Tentar importar Faker para gerar dados fictícios
try:
    from faker import Faker
    FAKER_AVAILABLE = True
except ImportError:
    FAKER_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# =============================================================================
# CONSTANTES
# =============================================================================
DB_PATH = Path(__file__).parent / "dbase.db"

COLORS = {
    "azul": "#4C78A8",
    "laranja": "#F58518",
    "verde": "#54A24B",
    "vermelho": "#E45756",
    "roxo": "#B279A2",
    "teal": "#72B7B2",
    "amarelo": "#EECA3B",
    "cinza": "#BAB0AC",
}
SEQUENCE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2", "#72B7B2", "#EECA3B", "#BAB0AC"]

PDD_FAIXAS_DEFAULT = [
    ("1-30 dias", 1, 30, 0.02),
    ("31-60 dias", 31, 60, 0.10),
    ("61-90 dias", 61, 90, 0.30),
    ("90+ dias", 91, 10_000, 0.50),
]

GLOSSARIO = {
    "PDD": "Provisão para Devedores Duvidosos - estimativa de perda esperada sobre saldo vencido.",
    "FPD": "First Payment Default - % de contratos que inadimpliram já na 1ª parcela.",
    "Roll Rate": "% do saldo que migra de uma faixa de atraso para a seguinte.",
    "LGD": "Loss Given Default - perda real observada dado que o cliente defaultou.",
    "EAD": "Exposure at Default - exposição esperada no momento do default.",
    "Recovery Rate": "% do saldo vencido efetivamente recuperado em um período.",
    "Vintage": "Análise de coortes por mês de origem, acompanhando maturação.",
    "HHI": "Índice Herfindahl-Hirschman - mede concentração da carteira.",
    "Curing Curve": "Curva de cura - mostra % do saldo vencido recuperado ao longo do tempo.",
    "PD": "Probability of Default - probabilidade de um contrato entrar em default.",
    "90+": "Contratos com atraso superior a 90 dias - considerados de alto risco.",
}

# =============================================================================
# GERADOR DE DADOS FAKE
# =============================================================================
def generate_fake_data(num_usuarios=8, num_clientes=300, num_contratos=500, num_movimentos=3000):
    """Gera dados fictícios para demonstração do painel."""
    
    if not FAKER_AVAILABLE:
        return None, None, None, None
    
    fake = Faker('pt_BR')
    Faker.seed(42)
    random.seed(42)
    np.random.seed(42)
    
    today = date.today()
    
    # 1. Gerar usuários (agentes)
    usuarios_data = []
    for i in range(num_usuarios):
        usuarios_data.append({
            "id": i + 1,
            "usuario": fake.name(),
            "email": fake.email(),
            "ativo": random.choice([0, 1])
        })
    usuarios = pd.DataFrame(usuarios_data)
    
    # 2. Gerar clientes
    clientes_data = []
    estabelecimentos = [
        "Restaurante do João", "Pizza Nova", "Bar do Zé", "Padaria Pão Quente",
        "Supermercado Bom Preço", "Farmácia Saúde", "Loja de Roupas Fashion",
        "Mecânica do Chico", "Salão de Beleza Vênus", "Mercearia da Dona Maria",
        "Construmaterial", "Lanchonete Sabor", "Hotel Parque", "Vendedor Ambulante",
        "Academia Fitness", "Pet Shop Amigo", "Papelaria Escolar", "Ótica Visão",
        "Joalheria Ouro", "Eletrônicos Tech"
    ]
    
    generos = ["Masculino", "Feminino"]
    avaliacoes = ["A", "B", "C", "D"]
    
    for i in range(num_clientes):
        idade = random.randint(18, 75)
        clientes_data.append({
            "id": i + 1,
            "cliente": fake.name(),
            "idade": idade,
            "genero": random.choice(generos),
            "cpf": fake.cpf(),
            "telefone": fake.phone_number(),
            "avaliacao": random.choice(avaliacoes),
            "nome_estabelecimento": random.choice(estabelecimentos),
            "endereco": fake.address(),
            "cidade": fake.city(),
            "estado": fake.state_abbr(),
        })
    clientes = pd.DataFrame(clientes_data)
    
    # 3. Gerar contratos
    contratos_data = []
    status_options = ["Ativo", "Finalizado", "Cancelado"]
    
    for i in range(num_contratos):
        idcliente = random.randint(1, num_clientes)
        idusuario = random.randint(1, num_usuarios)
        valor = random.uniform(500, 5000)
        qtd_parcela = random.choice([30, 60, 90, 120, 180])
        dtinicio = fake.date_between(start_date='-180d', end_date='-1d')
        dtfim = dtinicio + timedelta(days=qtd_parcela)
        
        taxa_juros = random.uniform(0.05, 0.15)
        valor_parcelado = valor * (1 + taxa_juros)
        
        contratos_data.append({
            "id": i + 1,
            "idcliente": idcliente,
            "idusuario": idusuario,
            "valor": round(valor, 2),
            "valor_parcelado": round(valor_parcelado, 2),
            "qtd_parcela": qtd_parcela,
            "dtinicio": dtinicio,
            "dtfim": dtfim,
            "dtatualizacao": fake.date_between(start_date='-30d', end_date='today'),
            "status": random.choices(status_options, weights=[0.7, 0.2, 0.1])[0],
            "observacao": fake.sentence()
        })
    contratos = pd.DataFrame(contratos_data)
    
    # 4. Gerar movimentações (parcelas)
    movimentos_data = []
    
    for _, contrato in contratos.iterrows():
        contrato_id = contrato["id"]
        idcliente = contrato["idcliente"]
        idusuario = contrato["idusuario"]
        valor_parcela = contrato["valor_parcelado"] / contrato["qtd_parcela"]
        dtinicio = contrato["dtinicio"]
        
        for parcela_num in range(1, contrato["qtd_parcela"] + 1):
            dtvenc = dtinicio + timedelta(days=parcela_num)
            
            # status_pago: 1 = pago, 0 = não pago
            status_pago = random.choices([0, 1], weights=[0.3, 0.7])[0]
            
            if status_pago == 0 and dtvenc < today:
                if random.random() < 0.3:
                    status_pago = 1
                    dtrecebimento = dtvenc + timedelta(days=random.randint(1, 90))
                    valorrecebido = valor_parcela * random.uniform(0.8, 1.0)
                    desconto = valor_parcela - valorrecebido if random.random() < 0.2 else 0
                else:
                    dtrecebimento = None
                    valorrecebido = 0
                    desconto = 0
            else:
                if status_pago == 1:
                    dtrecebimento = dtvenc + timedelta(days=random.randint(-5, 5))
                    valorrecebido = valor_parcela
                    desconto = 0
                else:
                    dtrecebimento = None
                    valorrecebido = 0
                    desconto = 0
            
            movimentos_data.append({
                "id": len(movimentos_data) + 1,
                "idcontrato": contrato_id,
                "idcliente": idcliente,
                "idusuario": idusuario,
                "dtinicio": dtinicio,
                "dtvenc": dtvenc,
                "dtrecebimento": dtrecebimento,
                "valorcontrato": contrato["valor"],
                "areceber": valor_parcela if status_pago == 0 else 0,
                "valorrecebido": round(valorrecebido, 2),
                "desconto": round(desconto, 2),
                "recebido": status_pago,
                "status_pago": status_pago == 1,
                "ok": "Sim" if status_pago == 1 else "Nao",
                "dtfim": contrato["dtfim"],
                "dtatualizacao": today,
            })
    
    movimentos = pd.DataFrame(movimentos_data)
    
    return usuarios, clientes, contratos, movimentos

# =============================================================================
# UTILITÁRIOS
# =============================================================================
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
    if pd.isna(v): return "R$ 0,00"
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def fmt_pct(v, digits=1):
    if pd.isna(v): return "0%"
    return f"{v*100:.{digits}f}%"

def normalize_estabelecimento(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return "OUTROS"
    x = name.upper()
    patterns = [
        ("RESTAURANT", "RESTAURANTE"), ("PIZZA", "PIZZARIA"), ("BAR", "BAR"),
        ("PUB", "BAR"), ("PADARI", "PADARIA/PANIFICADORA"), ("PANIFICAD", "PADARIA/PANIFICADORA"),
        ("BOULANGER", "PADARIA/PANIFICADORA"), ("LOJA", "LOJA"), ("SUPERMERC", "SUPERMERCADO"),
        ("MERCEARIA", "MERCEARIA"), ("CAF", "CAFETERIA"), ("HOTEL", "HOTEL"),
        ("ASSIST", "ASSISTENCIA"), ("BARBEARIA", "BARBEARIA"), ("ESPET", "ESPETINHO"),
        ("PEIXA", "PEIXARIA"), ("VEND", "VENDEDOR"), ("FRUTARIA", "FRUTARIA"),
        ("SALGADO", "SALGADOS"), ("SALAO", "SALAO/BELEZA"), ("BELEZA", "SALAO/BELEZA"),
        ("UNHA", "SALAO/BELEZA"), ("CELULA", "ELETRONICOS"), ("ELETRONIC", "ELETRONICOS"),
        ("SERRALHERIA", "SERRALHERIA"), ("MECANIC", "OFICINA/MECANICA"),
        ("BORRACHA", "OFICINA/MECANICA"), ("LAVA", "LAVA-JATO"), ("MERCADO", "MERCADO"),
        ("PENSAO", "PENSAO"), ("DEPOSITO", "DEPOSITO/ATACADO"), ("ATACAD", "DEPOSITO/ATACADO"),
        ("CONSTRU", "CONSTRUCAO"), ("MATERIAL", "CONSTRUCAO"), ("MOVEL", "MOVEIS"),
        ("ROUPA", "VESTUARIO"), ("VESTUARIO", "VESTUARIO"), ("SAPATO", "VESTUARIO"),
        ("ACOUGUE", "ACOUGUE"), ("LANCH", "LANCHONETE"), ("ACAI", "ACAI"),
        ("BOLO", "CONFEITARIA"), ("DOCE", "CONFEITARIA"), ("FARM", "FARMACIA"),
        ("CABEL", "SALAO/BELEZA"), ("MOTOR", "TRANSPORTE"), ("TRANSPORT", "TRANSPORTE"),
        ("MOTOTAXI", "TRANSPORTE"), ("PADARIA", "PADARIA/PANIFICADORA"),
    ]
    for token, label in patterns:
        if token in x:
            return label
    first = x.split()[0] if x.split() else ""
    valid_firsts = {"RESTAURANTE", "PIZZARIA", "PEIXARIA", "VENDEDOR", "CONVENIENCIA",
                    "BAR", "ESPETINHO", "BARBEARIA", "PADARIA", "LOJA", "SUPERMERCADO",
                    "CAFETERIA", "ASSISTENCIA", "HOTEL", "FRUTARIA", "SALGADOS", "LANCHONETE"}
    if first in valid_firsts:
        return first
    return "OUTROS"

def ensure_datetime(df, columns):
    """Garante que as colunas especificadas sejam datetime."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

# =============================================================================
# CARREGAMENTO DE DADOS (CORRIGIDO)
# =============================================================================
@st.cache_data(ttl=60)
def load_data(use_fake=False):
    """Carrega dados do banco ou gera dados fictícios, aplicando o mesmo processamento."""
    
    usuarios, clientes, contratos, movimentos = None, None, None, None
    
    # 1. Gera ou carrega os dados brutos
    if use_fake and FAKER_AVAILABLE:
        usuarios, clientes, contratos, movimentos = generate_fake_data(
            num_usuarios=8,
            num_clientes=300,
            num_contratos=500,
            num_movimentos=3000
        )
    else:
        # Carrega do banco
        if use_fake and not FAKER_AVAILABLE:
            st.warning("Faker não instalado. Usando dados reais do banco.")
        
        try:
            conn = sqlite3.connect(DB_PATH)
            usuarios = pd.read_sql_query("SELECT * FROM usuarios", conn)
            clientes = pd.read_sql_query("SELECT * FROM clientes", conn)
            contratos = pd.read_sql_query("SELECT * FROM contratos2", conn)
            movimentos = pd.read_sql_query("SELECT * FROM contratos_movimentacoes2", conn)
            conn.close()
        except Exception as e:
            st.error(f"Erro ao carregar dados do banco: {e}")
            if FAKER_AVAILABLE:
                st.info("Gerando dados fictícios como fallback...")
                usuarios, clientes, contratos, movimentos = generate_fake_data()
            else:
                return None, None, None, None
    
    # Verifica se os dados foram carregados
    if usuarios is None or clientes is None or contratos is None or movimentos is None:
        return None, None, None, None
    
    # =============================================================================
    # 2. PROCESSAMENTO COMUM (SEMPRE executado, independente da origem dos dados)
    # =============================================================================
    
    # --- Clientes ---
    clientes = ensure_datetime(clientes, ["dtinicio", "dtfim", "dtatualizacao"])
    clientes["idade"] = pd.to_numeric(clientes.get("idade"), errors="coerce")
    clientes["idade"] = clientes["idade"].where(clientes["idade"] > 0, np.nan)
    clientes["genero"] = clientes.get("genero").astype(str).str.strip()
    # Corrige o mapeamento para aceitar tanto "1"/"0" quanto "Masculino"/"Feminino"
    clientes["genero_cat"] = clientes["genero"].replace({"1": "Masculino", "0": "Feminino"}).fillna("Outro")
    
    idade_bins = [0, 18, 25, 35, 45, 55, 65, 200]
    idade_labels = ["<18", "18-25", "26-35", "36-45", "46-55", "56-65", ">65"]
    clientes["faixa_idade"] = pd.cut(clientes["idade"], bins=idade_bins, labels=idade_labels)
    clientes["faixa_idade"] = clientes["faixa_idade"].cat.add_categories(["Sem idade"]).fillna("Sem idade")
    clientes["avaliacao"] = clientes["avaliacao"].astype(str).fillna("Nao avaliado")
    clientes["nome_estabelecimento"] = clientes["nome_estabelecimento"].astype(str).fillna("Desconhecido")

    # --- Contratos ---
    contratos = ensure_datetime(contratos, ["dtinicio", "dtfim", "dtatualizacao"])
    contratos["valor"] = pd.to_numeric(contratos.get("valor"), errors="coerce").fillna(0)
    contratos["valor_parcelado"] = pd.to_numeric(contratos.get("valor_parcelado"), errors="coerce")
    contratos["valor_parcelado"] = contratos["valor_parcelado"].fillna(contratos["valor"])
    contratos["qtd_parcela"] = pd.to_numeric(contratos.get("qtd_parcela"), errors="coerce").fillna(0)
    
    contratos["parcela_esperada"] = np.where(
        contratos["qtd_parcela"] > 0, contratos["valor_parcelado"] / contratos["qtd_parcela"], 0
    )
    contratos["frac_principal"] = np.where(
        contratos["valor_parcelado"] > 0, contratos["valor"] / contratos["valor_parcelado"], 0
    )
    contratos["juros_previstos"] = contratos["valor_parcelado"] - contratos["valor"]
    
    contratos = contratos.merge(
        clientes[["id", "cliente", "genero_cat", "faixa_idade", "avaliacao", "nome_estabelecimento"]],
        left_on="idcliente", right_on="id", how="left", suffixes=("", "_cliente"),
    )
    if "usuario" in usuarios.columns:
        contratos["usuario_nome"] = contratos["idusuario"].map(usuarios.set_index("id")["usuario"])

    # --- Movimentos ---
    movimentos = ensure_datetime(movimentos, ["dtinicio", "dtfim", "dtvenc", "dtrecebimento", "dtatualizacao"])
    movimentos["valorrecebido"] = pd.to_numeric(movimentos.get("valorrecebido"), errors="coerce").fillna(0)
    movimentos["areceber"] = pd.to_numeric(movimentos.get("areceber"), errors="coerce").fillna(0)
    movimentos["valorcontrato"] = pd.to_numeric(movimentos.get("valorcontrato"), errors="coerce").fillna(0)
    movimentos["desconto"] = pd.to_numeric(movimentos.get("desconto"), errors="coerce").fillna(0)
    movimentos["recebido"] = pd.to_numeric(movimentos.get("recebido"), errors="coerce")
    movimentos["paid"] = movimentos["recebido"] == 1
    
    if "ok" in movimentos.columns:
        movimentos["status_pago"] = movimentos["paid"] | (movimentos["ok"].astype(str).str.lower() == "sim")
    else:
        movimentos["status_pago"] = movimentos["paid"]

    movimentos = movimentos.merge(
        contratos[["id", "parcela_esperada", "frac_principal", "valor", "valor_parcelado",
                   "juros_previstos", "dtinicio"]],
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

    # >>> COLUNAS CRÍTICAS QUE ESTAVAM FALTANDO <<<
    movimentos["vencido"] = (~movimentos["status_pago"]) & movimentos["dtvenc"].notna() & (movimentos["dtvenc"] < today)
    movimentos["a_vencer"] = (~movimentos["status_pago"]) & movimentos["dtvenc"].notna() & (movimentos["dtvenc"] >= today)
    movimentos["atraso_90"] = movimentos["vencido"] & (movimentos["dias_atraso"] >= 90)

    movimentos = movimentos.merge(
        clientes[["id", "cliente", "genero_cat", "faixa_idade", "avaliacao", "nome_estabelecimento"]],
        left_on="idcliente", right_on="id", how="left", suffixes=("", "_cliente"),
    )
    movimentos["nome_estabelecimento"] = movimentos["nome_estabelecimento"].fillna("Desconhecido")
    movimentos["nome_estabelecimento_norm"] = movimentos["nome_estabelecimento"].astype(str).apply(normalize_estabelecimento)
    if "usuario" in usuarios.columns:
        movimentos["usuario_nome"] = movimentos["idusuario"].map(usuarios.set_index("id")["usuario"])

    movimentos = movimentos.sort_values(["idcontrato", "dtvenc"])
    movimentos["num_parcela"] = movimentos.groupby("idcontrato").cumcount() + 1
    movimentos["is_primeira_parcela"] = movimentos["num_parcela"] == 1

    contrato_agg = movimentos.groupby("idcontrato").agg(
        total_recebido=("valorrecebido", "sum"),
        total_desconto=("desconto", "sum"),
        total_aberto=("areceber", "sum"),
        aberto_nao_pago=("areceber", lambda s: s[movimentos.loc[s.index, "status_pago"] == False].sum()),
        parcelas_pagas=("status_pago", "sum"),
        parcelas_total=("id", "count"),
        vencido_valor=("areceber", lambda s: s[movimentos.loc[s.index, "vencido"]].sum()),
        max_atraso=("dias_atraso", "max"),
        avg_atraso=("dias_atraso", "mean"),
    ).reset_index()
    
    contratos = contratos.merge(contrato_agg, left_on="id", right_on="idcontrato", how="left").fillna(
        {c: 0 for c in ["total_recebido", "total_desconto", "total_aberto", "aberto_nao_pago",
                        "parcelas_pagas", "parcelas_total", "vencido_valor", "max_atraso", "avg_atraso"]}
    )
    contratos["percentual_recebido"] = (
        contratos["total_recebido"] / contratos["valor_parcelado"].replace(0, np.nan)
    ).fillna(0).clip(0, 1)
    contratos["principal_realizado"] = contratos["total_recebido"] * contratos["frac_principal"]
    contratos["juros_realizados"] = contratos["total_recebido"] - contratos["principal_realizado"]
    contratos["default_90d"] = (contratos["max_atraso"] >= 90).astype(int)

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

# =============================================================================
# FUNÇÃO CORRIGIDA - apply_period_filter
# =============================================================================
def apply_period_filter(contratos, movimentos, start_date, end_date):
    # Converter para Timestamp para comparação correta
    st_ts = pd.Timestamp(start_date)
    en_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    
    # Garantir que a coluna dtvenc seja datetime e não tenha valores nulos
    movimentos = movimentos.copy()
    movimentos["dtvenc"] = pd.to_datetime(movimentos["dtvenc"], errors='coerce')
    
    # Filtrar apenas linhas com dtvenc não nulo
    mask = movimentos["dtvenc"].notna()
    
    # Filtrar pelo período usando comparação direta com Timestamp (mais seguro)
    mask = mask & (movimentos["dtvenc"] >= st_ts) & (movimentos["dtvenc"] < en_ts)
    
    movimentos_f = movimentos[mask].copy()
    
    contratos_ids = movimentos_f["idcontrato"].unique()
    contratos_f = contratos[contratos["id"].isin(contratos_ids)].copy()
    return contratos_f, movimentos_f

# =============================================================================
# FUNÇÕES DE ANÁLISE
# =============================================================================
def build_cashflow(movimentos, contratos, today, n_future=6):
    mo = movimentos.copy()
    # Garantir que dtvenc seja datetime
    mo["dtvenc"] = pd.to_datetime(mo["dtvenc"], errors='coerce')
    mo["dtrecebimento"] = pd.to_datetime(mo["dtrecebimento"], errors='coerce')
    
    sched = (mo.dropna(subset=["dtvenc"])
             .assign(mes=lambda d: d["dtvenc"].dt.to_period("M"))
             .groupby("mes")["parcela"].sum())
    real = (mo.dropna(subset=["dtrecebimento"])
            .assign(mes=lambda d: d["dtrecebimento"].dt.to_period("M"))
            .groupby("mes")["valorrecebido"].sum())
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

    future_open = (mo[mo["a_vencer"]].dropna(subset=["dtvenc"])
                   .assign(mes=lambda d: d["dtvenc"].dt.to_period("M"))
                   .groupby("mes")["areceber"].sum())
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
    bins = [0, 30, 60, 90, 10_000_000]
    labels = ["1-30 dias", "31-60 dias", "61-90 dias", "90+ dias"]
    venc["faixa"] = pd.cut(venc["dias"], bins=bins, labels=labels, include_lowest=True)
    return venc.groupby("faixa", observed=False).agg(
        valor=("areceber", "sum"), parcelas=("id", "count")
    ).reset_index()

def compute_lgd_observada(movimentos):
    venc = movimentos[movimentos["vencido"]].copy()
    if venc.empty:
        return {label: rate for label, _, _, rate in PDD_FAIXAS_DEFAULT}
    venc["dias"] = venc["dias_atraso"].clip(lower=1)
    bins = [0, 30, 60, 90, 10_000_000]
    labels = ["1-30 dias", "31-60 dias", "61-90 dias", "90+ dias"]
    venc["faixa"] = pd.cut(venc["dias"], bins=bins, labels=labels, include_lowest=True)

    cut = pd.Timestamp(date.today()) - pd.Timedelta(days=365)
    rec_hist = movimentos[(movimentos["dtrecebimento"] >= cut) & (movimentos["dias_atraso"] > 0)]
    if rec_hist.empty:
        return {label: rate for label, _, _, rate in PDD_FAIXAS_DEFAULT}

    rec_hist = rec_hist.copy()
    rec_hist["dias"] = rec_hist["dias_atraso"].clip(lower=1)
    rec_hist["faixa"] = pd.cut(rec_hist["dias"], bins=bins, labels=labels, include_lowest=True)

    lgd_obs = {}
    for label, lo, hi, default_rate in PDD_FAIXAS_DEFAULT:
        exp = venc.loc[venc["faixa"] == label, "areceber"].sum()
        rec = rec_hist.loc[rec_hist["faixa"] == label, "valorrecebido"].sum()
        if exp > 0:
            recovery = min(rec / exp, 1.0)
            lgd_obs[label] = max(1.0 - recovery, default_rate * 0.5)
        else:
            lgd_obs[label] = default_rate
    return lgd_obs

def build_pdd(backlog_df, lgd_rates=None):
    if backlog_df.empty:
        return pd.DataFrame(columns=["Faixa", "Valor em aberto", "% Provisao", "PDD"])
    if lgd_rates is None:
        lgd_rates = {label: rate for label, _, _, rate in PDD_FAIXAS_DEFAULT}
    rows = []
    for _, row in backlog_df.iterrows():
        label = row["faixa"]
        val = row["valor"]
        rate = lgd_rates.get(label, 0.0)
        if val > 0:
            rows.append({"Faixa": label, "Valor em aberto": val, "% Provisao": rate, "PDD": val * rate})
    return pd.DataFrame(rows)

def build_fpd(movimentos, contratos):
    first = movimentos[movimentos["is_primeira_parcela"]].copy()
    if first.empty:
        return {"fpd_30": 0.0, "fpd_90": 0.0, "total_contratos_1p": 0}
    first["dias"] = first["dias_atraso"].clip(lower=0)
    n = first["idcontrato"].nunique()
    fpd_30 = first.loc[first["dias"] >= 30, "idcontrato"].nunique()
    fpd_90 = first.loc[first["dias"] >= 90, "idcontrato"].nunique()
    return {
        "fpd_30": fpd_30 / n if n > 0 else 0.0,
        "fpd_90": fpd_90 / n if n > 0 else 0.0,
        "total_contratos_1p": n,
    }

def build_roll_rate(movimentos):
    mo = movimentos[movimentos["dtvenc"].notna()].copy()
    if mo.empty:
        return pd.DataFrame()
    mo["mes_venc"] = mo["dtvenc"].dt.to_period("M")
    mo["faixa"] = pd.cut(
        mo["dias_atraso"].clip(lower=0),
        bins=[-1, 0, 30, 60, 90, 10_000],
        labels=["Em dia", "1-30d", "31-60d", "61-90d", "90+d"]
    )
    pivot = mo.groupby(["idcontrato", "mes_venc"])["faixa"].first().reset_index()
    pivot = pivot.sort_values(["idcontrato", "mes_venc"])
    pivot["faixa_next"] = pivot.groupby("idcontrato")["faixa"].shift(-1)
    pivot = pivot.dropna(subset=["faixa_next"])
    if pivot.empty:
        return pd.DataFrame()
    trans = pivot.groupby(["faixa", "faixa_next"]).size().reset_index(name="count")
    total_by_from = trans.groupby("faixa")["count"].transform("sum")
    trans["pct"] = trans["count"] / total_by_from
    return trans

def build_recovery_curve(movimentos):
    paid = movimentos[movimentos["status_pago"] & movimentos["dtvenc"].notna() & movimentos["dtrecebimento"].notna()].copy()
    if paid.empty:
        return pd.DataFrame(columns=["janela", "valor", "pct_acumulado"])
    paid["atraso"] = (paid["dtrecebimento"] - paid["dtvenc"]).dt.days.clip(lower=0)
    total = paid["valorrecebido"].sum()
    janelas = [0, 3, 7, 15, 30, 60, 90, 10_000]
    nomes = ["em dia (0d)", "ate 3d", "ate 7d", "ate 15d", "ate 30d", "ate 60d", "ate 90d", "90+d"]
    rows, acc = [], 0.0
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
    g = mo.groupby("mes_venc").agg(
        programado=("parcela", "sum"),
        total_recebido=("valorrecebido", "sum"),
        parcelas=("id", "count"),
    ).reset_index()
    rec = mo[mo["dtrecebimento"].notna()].copy()
    rec["juros_rec"] = rec["valorrecebido"] * rec["juros_frac"]
    rec["princ_rec"] = rec["valorrecebido"] * rec["frac_principal"]
    g["juros_recebidos"] = g["mes_venc"].map(rec.groupby("mes_venc")["juros_rec"].sum()).fillna(0)
    g["principal_recebido"] = g["mes_venc"].map(rec.groupby("mes_venc")["princ_rec"].sum()).fillna(0)
    venc = mo[~mo["status_pago"]].copy()
    venc["juros_ab"] = venc["areceber"] * venc["juros_frac"]
    g["juros_aberto"] = g["mes_venc"].map(venc.groupby("mes_venc")["juros_ab"].sum()).fillna(0)
    g["desconto"] = g["mes_venc"].map(mo.groupby("mes_venc")["desconto"].sum()).fillna(0)
    g["mes_ts"] = g["mes_venc"].dt.to_timestamp()
    g["mes_label"] = g["mes_venc"].astype(str)
    keep = ["mes_venc", "mes_ts", "mes_label", "programado", "principal_recebido",
            "juros_recebidos", "juros_aberto", "desconto", "parcelas"]
    return g[keep]

def build_monthly_efficiency(movimentos):
    mo = movimentos.copy()
    mo["mes_venc"] = mo["dtvenc"].dt.to_period("M")
    g = (mo.dropna(subset=["mes_venc"])
         .groupby("mes_venc")
         .agg(programado=("parcela", "sum"), recebido=("valorrecebido", "sum"),
              parcelas=("id", "count")).reset_index())
    g["eficiencia"] = g["recebido"] / g["programado"].replace(0, np.nan)
    g["mes_ts"] = g["mes_venc"].dt.to_timestamp()
    g["mes_label"] = g["mes_venc"].astype(str)
    return g

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
        principal = g["valor"].sum()
        a_receber = g["valor_parcelado"].sum()
        programado_venc = mv.loc[mv["dtvenc"] <= pd.Timestamp(date.today()), "parcela"].sum()
        juros_prev = g["juros_previstos"].sum()
        juros_real = g["juros_realizados"].sum()
        eficiencia = recebido / programado_venc if programado_venc > 0 else 0.0
        rows.append({
            "Agente": nome, "Contratos": g["id"].nunique(), "Clientes": mv["idcliente"].nunique(),
            "Principal (R$)": principal, "A receber (R$)": a_receber,
            "Recebido (R$)": recebido, "Em aberto (R$)": aberto, "Desconto (R$)": desconto,
            "Eficiencia %": eficiencia, "Vencido 90+ (R$)": venc_90,
            "Juros previstos (R$)": juros_prev, "Juros realizados (R$)": juros_real,
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
    rec = (mo.dropna(subset=["dtrecebimento"])
           .assign(dias_rec=lambda d: (d["dtrecebimento"] - d["dtinicio"]).dt.days.clip(lower=0))
           .groupby(["coorte", "dias_rec"])["valorrecebido"].sum().reset_index()
           .rename(columns={"dias_rec": "dias_inicio"}))
    rows = []
    for c in coortes:
        p = prog[prog["coorte"] == c]
        r = rec[rec["coorte"] == c]
        base = p_tot.get(c, 0)
        for j in janelas:
            prog_j = p.loc[p["dias_inicio"] <= j, "parcela"].sum()
            rec_j = r.loc[r["dias_inicio"] <= j, "valorrecebido"].sum()
            rows.append({
                "coorte": str(c), "janela_dias": j,
                "programado_ate": prog_j, "recebido_ate": rec_j,
                "pct_programado": (prog_j / base) if base > 0 else 0,
                "pct_realizado_do_programado": (rec_j / prog_j) if prog_j > 0 else 0,
            })
    return pd.DataFrame(rows)

def build_concentration(movimentos):
    cli = movimentos.groupby("idcliente")["areceber"].sum().reset_index()
    cli.columns = ["idcliente", "valor"]
    total = cli["valor"].sum()
    if total <= 0:
        return {"hhi": 0, "top10_share": 0, "top20_share": 0, "lorenz": pd.DataFrame()}
    cli["share"] = cli["valor"] / total
    hhi = (cli["share"] ** 2).sum() * 10_000
    cli = cli.sort_values("share", ascending=True)
    cli["share_acum"] = cli["share"].cumsum()
    cli["clientes_acum_pct"] = (np.arange(1, len(cli) + 1) / len(cli)) * 100
    top10_share = cli.sort_values("share", ascending=False).head(10)["share"].sum()
    top20_share = cli.sort_values("share", ascending=False).head(20)["share"].sum()
    lorenz = cli[["clientes_acum_pct", "share_acum"]].copy()
    lorenz.columns = ["% Clientes (acum)", "% Saldo (acum)"]
    return {"hhi": hhi, "top10_share": top10_share, "top20_share": top20_share, "lorenz": lorenz}

def build_priority_clients(movimentos):
    pend = movimentos[movimentos["vencido"] | movimentos["a_vencer"]].copy()
    if pend.empty:
        return pd.DataFrame()
    pend["dias"] = pend["dias_atraso"].clip(lower=0)
    out = pend.groupby("idcliente", as_index=False).agg(
        Cliente=("cliente", "first"),
        Agente=("usuario_nome", "first"),
        Segmento=("nome_estabelecimento_norm", "first"),
        Avaliacao=("avaliacao", "first"),
        Valor_em_aberto=("areceber", "sum"),
        Parcelas_em_aberto=("id", "count"),
        Maior_atraso=("dias", "max"),
    ).sort_values(["Valor_em_aberto", "Maior_atraso"], ascending=[False, False])
    out["Maior_atraso_exibicao"] = out["Maior_atraso"].apply(lambda x: "90+" if x > 90 else f"{int(x)}d")
    out["Prioridade"] = np.select(
        [out["Maior_atraso"] >= 90, out["Maior_atraso"] >= 60, out["Maior_atraso"] >= 30],
        ["Critica", "Alta", "Media"], default="Baixa",
    )
    return out.rename(columns={
        "Valor_em_aberto": "Valor em aberto",
        "Parcelas_em_aberto": "Parcelas em aberto",
        "Maior_atraso_exibicao": "Maior atraso",
    })

def build_action_plan_prescritivo(aberto_90, open_80_89, vencido, open_next_30, pdd, eficiencia):
    rows = []
    if aberto_90 > 0:
        rows.append({
            "Acao": "Acionar cobranca externa / negociacao com desconto progressivo",
            "Prioridade": "CRITICA",
            "Detalhe": f"{fmt_brl(aberto_90)} em 90+ dias. Cada mes adicional reduz recuperacao em ~8pp.",
            "Impacto estimado": f"Recuperar 30-50% = {fmt_brl(aberto_90*0.4)} em caixa",
        })
    if open_80_89 > 0:
        rows.append({
            "Acao": "Reforcar cobranca preventiva (80-89 dias)",
            "Prioridade": "ALTA",
            "Detalhe": f"{fmt_brl(open_80_89)} prestes a entrar em 90+.",
            "Impacto estimado": f"Evitar perda de {fmt_brl(open_80_89*0.5)} em PDD",
        })
    if vencido > 0:
        rows.append({
            "Acao": "Campanha de recuperacao de backlog",
            "Prioridade": "ALTA",
            "Detalhe": f"{fmt_brl(vencido)} vencido em aberto.",
            "Impacto estimado": f"Meta de recuperacao: 25% = {fmt_brl(vencido*0.25)}",
        })
    if open_next_30 > 0:
        rows.append({
            "Acao": "Preparar recebiveis de curto prazo",
            "Prioridade": "MEDIA",
            "Detalhe": f"{fmt_brl(open_next_30)} a vencer nos proximos 30 dias.",
            "Impacto estimado": f"Se eficiencia subir 5pp = +{fmt_brl(open_next_30*0.05)}",
        })
    if eficiencia < 0.70:
        rows.append({
            "Acao": "Revisar processo de cobranca",
            "Prioridade": "ALTA",
            "Detalhe": f"Eficiencia atual ({eficiencia:.1%}) abaixo do benchmark (70-75%).",
            "Impacto estimado": "Ganho potencial de 5-10pp",
        })
    if not rows:
        rows.append({"Acao": "Operacao estavel", "Prioridade": "BAIXA",
                     "Detalhe": "Sem valores criticos.", "Impacto estimado": "-"})
    return pd.DataFrame(rows)

def build_new_contract_stats(contratos, start_date, end_date):
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    recent = contratos[
        contratos["dtinicio"].notna() &
        (contratos["dtinicio"] >= start) &
        (contratos["dtinicio"] <= end) &
        (contratos["dtinicio"].dt.dayofweek < 6)
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
    return {"Dia": summarize(daily), "Semana": summarize(weekly), "Mes": summarize(monthly)}

def build_insights_prescritivos(agentes, backlog, eficiencia, recebido, aberto, aberto_90, pdd,
                                best_agente, pior_agente, best_dow, worst_dow, contratos_novos_30d,
                                fpd, hhi):
    ins = []
    if recebido > 0:
        ins.append(f" **Conversao de caixa:** {eficiencia:.1%} do programado vencido foi recebido ({fmt_brl(recebido)}). "
                   f"**Acao:** se eficiencia < 70%, revisar roteiro de cobranca.")
    if aberto_90 > 0:
        ins.append(f" **Risco de perda iminente:** {fmt_brl(aberto_90)} ({aberto_90/aberto:.1%} do aberto) em 90+ dias. "
                   f"**Acao:** acionar cobranca externa com desconto progressivo (30-50%).")
    if pdd > 0:
        ins.append(f" **Provisao (PDD):** {fmt_brl(pdd)} ({pdd/aberto:.1%} do aberto). "
                   f"**Acao:** calibrar PDD com LGD observada trimestralmente.")
    if fpd["fpd_30"] > 0.10:
        ins.append(f"⚠️ **FPD30 alto:** {fpd['fpd_30']:.1%} inadimpliram na 1a parcela. "
                   f"**Acao:** revisar criterio de aprovacao.")
    if hhi > 2500:
        ins.append(f"🎯 **Concentracao elevada (HHI={hhi:.0f}).** "
                   f"**Acao:** diversificar carteira, limitar exposicao individual a 5%.")
    if best_agente:
        ins.append(f"🏆 **Referencia:** agente **{best_agente}** lidera eficiencia. "
                   f"**Acao:** documentar e replicar praticas.")
    if pior_agente:
        ins.append(f"⚠️ **Atencao:** agente **{pior_agente}** com menor eficiencia. "
                   f"**Acao:** acompanhamento semanal.")
    if best_dow and worst_dow:
        ins.append(f" **Sazonalidade:** {best_dow} = pico; {worst_dow} = dia mais fraco. "
                   f"**Acao:** concentrar cobranca ativa em {worst_dow}.")
    if contratos_novos_30d > 0:
        ins.append(f"🌱 **Origem:** {contratos_novos_30d} contratos novos nos ultimos 30 dias. "
                   f"**Acao:** monitorar FPD dessas coortes.")
    return ins

# =============================================================================
# FUNÇÕES DE VIABILIDADE E LUCRO
# =============================================================================
def build_monthly_profit(movimentos):
    rec = movimentos[movimentos["dtrecebimento"].notna()].copy()
    if rec.empty:
        return pd.DataFrame()
    
    rec["mes_recebimento"] = rec["dtrecebimento"].dt.to_period("M")
    
    if "frac_principal" not in rec.columns:
        rec["frac_principal"] = 0.0
    if "juros_frac" not in rec.columns:
        rec["juros_frac"] = 0.0
        
    rec["principal_recebido"] = rec["valorrecebido"] * rec["frac_principal"]
    rec["juros_recebidos"] = rec["valorrecebido"] * rec["juros_frac"]
    
    profit_df = rec.groupby("mes_recebimento").agg(
        total_recebido=("valorrecebido", "sum"),
        principal_recebido=("principal_recebido", "sum"),
        juros_recebidos=("juros_recebidos", "sum"),
        descontos=("desconto", "sum")
    ).reset_index()
    
    profit_df["lucro_bruto"] = profit_df["juros_recebidos"] - profit_df["descontos"]
    profit_df["margem_lucro_pct"] = (
        profit_df["lucro_bruto"] / profit_df["principal_recebido"].replace(0, np.nan)
    ).fillna(0) * 100
    profit_df["mes_label"] = profit_df["mes_recebimento"].astype(str)
    profit_df["mes_ts"] = profit_df["mes_recebimento"].dt.to_timestamp()
    
    return profit_df

def build_viability_analysis(contratos, movimentos, pdd_total):
    total_investido = contratos["valor"].sum()
    total_recebido = movimentos["valorrecebido"].sum()
    total_descontos = movimentos["desconto"].sum()
    
    total_a_receber_contratos = contratos["valor_parcelado"].sum()
    frac_pond = (total_investido / total_a_receber_contratos) if total_a_receber_contratos > 0 else 0
    
    principal_recuperado = total_recebido * frac_pond
    juros_recebidos = total_recebido - principal_recuperado
    
    lucro_bruto_real = juros_recebidos - total_descontos
    roi_bruto_pct = (lucro_bruto_real / total_investido * 100) if total_investido > 0 else 0
    
    lucro_liquido_ajustado = lucro_bruto_real - pdd_total
    cobertura_risco = lucro_bruto_real / pdd_total if pdd_total > 0 else 0
    margem_lucro_pct = (lucro_bruto_real / principal_recuperado * 100) if principal_recuperado > 0 else 0

    return {
        "total_investido": total_investido,
        "total_recebido": total_recebido,
        "principal_recuperado": principal_recuperado,
        "juros_recebidos": juros_recebidos,
        "descontos": total_descontos,
        "lucro_bruto_real": lucro_bruto_real,
        "roi_bruto_pct": roi_bruto_pct,
        "pdd_total": pdd_total,
        "lucro_liquido_ajustado": lucro_liquido_ajustado,
        "cobertura_risco": cobertura_risco,
        "margem_lucro_pct": margem_lucro_pct,
    }

# =============================================================================
# MODELOS PREDITIVOS
# =============================================================================
def prepare_features_pd(contratos, movimentos):
    try:
        co = contratos.copy()
        mo = movimentos.copy()
        
        contract_features = mo.groupby("idcontrato").agg(
            total_parcelas=("id", "count"),
            parcelas_pagas=("status_pago", "sum"),
            total_recebido=("valorrecebido", "sum"),
            total_aberto=("areceber", "sum"),
            max_atraso=("dias_atraso", "max"),
            avg_atraso=("dias_atraso", "mean"),
            qtd_vencida=("vencido", "sum"),
            qtd_90d=("atraso_90", "sum"),
        ).reset_index()
        
        co = co.merge(contract_features, left_on="id", right_on="idcontrato", how="left", suffixes=("", "_agg"))
        
        fill_cols = ["total_parcelas", "parcelas_pagas", "total_recebido", "total_aberto", 
                     "max_atraso", "avg_atraso", "qtd_vencida", "qtd_90d"]
        for col in fill_cols:
            co[col] = co[col].fillna(0) if col in co.columns else 0
        
        co["pct_pagas"] = co["parcelas_pagas"] / co["total_parcelas"].replace(0, 1)
        co["pct_vencido"] = co["qtd_vencida"] / co["total_parcelas"].replace(0, 1)
        co["pct_90d"] = co["qtd_90d"] / co["total_parcelas"].replace(0, 1)
        co["valor_parcela"] = co["valor_parcelado"] / co["qtd_parcela"].replace(0, 1)
        
        if "dtinicio" in co.columns:
            co["dias_desde_inicio"] = (pd.Timestamp(date.today()) - co["dtinicio"]).dt.days.fillna(0)
        else:
            co["dias_desde_inicio"] = 0
        
        if "genero_cat" in co.columns:
            co["genero_enc"] = co["genero_cat"].map({"Masculino": 0, "Feminino": 1, "Outro": 2}).fillna(2)
        else:
            co["genero_enc"] = 2
            
        if "faixa_idade" in co.columns:
            co["faixa_idade_enc"] = co["faixa_idade"].astype(str).astype('category').cat.codes
        else:
            co["faixa_idade_enc"] = 0
            
        if "nome_estabelecimento" in co.columns:
            co["segmento_enc"] = co["nome_estabelecimento"].astype(str).astype('category').cat.codes
        else:
            co["segmento_enc"] = 0
            
        if "usuario_nome" in co.columns:
            co["agente_enc"] = co["usuario_nome"].astype(str).astype('category').cat.codes
        else:
            co["agente_enc"] = 0
        
        co["default_90d"] = (co["max_atraso"] >= 90).astype(int)
        
        available_features = ["valor", "valor_parcelado", "qtd_parcela", "idade", "genero_enc", 
                             "faixa_idade_enc", "segmento_enc", "agente_enc", "total_parcelas",
                             "pct_pagas", "pct_vencido", "pct_90d", "max_atraso", "avg_atraso",
                             "valor_parcela", "dias_desde_inicio"]
        
        feature_cols = [f for f in available_features if f in co.columns]
        
        X = co[feature_cols].fillna(0)
        y = co["default_90d"]
        
        return X, y, co, feature_cols
        
    except Exception as e:
        st.error(f"Erro ao preparar features: {str(e)}")
        return pd.DataFrame(), pd.Series(), pd.DataFrame(), []

def train_pd_model(X, y, feature_cols):
    try:
        if len(X) < 20:
            raise ValueError(f"Dados insuficientes: {len(X)} contratos (minimo 20)")
        
        if y.sum() < 5:
            raise ValueError(f"Defaults insuficientes: {y.sum()} (minimo 5)")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'
        )
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = {
            "accuracy": float((y_pred == y_test).mean()),
            "auc_roc": float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else 0.0,
            "precision": float((y_pred[y_pred == 1] == y_test[y_pred == 1]).mean()) if (y_pred == 1).sum() > 0 else 0.0,
            "recall": float((y_pred[y_test == 1] == y_test[y_test == 1]).mean()) if (y_test == 1).sum() > 0 else 0.0,
        }
        
        feature_importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        
        return model, scaler, metrics, feature_importance, X_test, y_test, y_prob
        
    except Exception as e:
        st.error(f"Erro ao treinar modelo: {str(e)}")
        return None, None, None, None, None, None, None

def predict_pd_scores(model, scaler, X, feature_cols):
    try:
        if model is None or scaler is None:
            return np.zeros(len(X))
        
        X_filtered = X[feature_cols] if all(f in X.columns for f in feature_cols) else X
        X_scaled = scaler.transform(X_filtered)
        scores = model.predict_proba(X_scaled)[:, 1]
        return scores
    except Exception as e:
        st.error(f"Erro ao gerar scores: {str(e)}")
        return np.zeros(len(X))

def build_clustering_model(contratos, movimentos):
    try:
        co = contratos.copy()
        mo = movimentos.copy()
        
        contract_features = mo.groupby("idcontrato").agg(
            total_recebido=("valorrecebido", "sum"),
            total_aberto=("areceber", "sum"),
            max_atraso=("dias_atraso", "max"),
            avg_atraso=("dias_atraso", "mean"),
            parcelas_pagas=("status_pago", "sum"),
            total_parcelas=("id", "count"),
        ).reset_index()
        
        co = co.merge(contract_features, left_on="id", right_on="idcontrato", how="left", suffixes=("", "_cluster"))
        
        fill_cols = ["total_recebido", "total_aberto", "max_atraso", "avg_atraso", 
                     "parcelas_pagas", "total_parcelas"]
        for col in fill_cols:
            co[col] = co[col].fillna(0) if col in co.columns else 0
        
        co["pct_pagas"] = co["parcelas_pagas"] / co["total_parcelas"].replace(0, 1)
        co["pct_recebido"] = co["total_recebido"] / co["valor_parcelado"].replace(0, 1)
        
        feature_cols = ["valor", "max_atraso", "avg_atraso", "pct_pagas", "pct_recebido"]
        available_features = [f for f in feature_cols if f in co.columns]
        
        X = co[available_features].fillna(0)
        
        if len(X) < 10:
            raise ValueError(f"Dados insuficientes para clustering: {len(X)} contratos (minimo 10)")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        n_clusters = min(4, len(X) // 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        co["cluster"] = kmeans.fit_predict(X_scaled)
        
        cluster_summary = co.groupby("cluster").agg({
            "valor": "mean",
            "max_atraso": "mean",
            "pct_pagas": "mean",
            "pct_recebido": "mean"
        }).round(2)
        
        cluster_names = {i: f"Cluster {i+1}" for i in range(n_clusters)}
        co["segmento_comportamental"] = co["cluster"].map(cluster_names)
        
        return co, kmeans, scaler, cluster_summary
        
    except Exception as e:
        st.error(f"Erro no clustering: {str(e)}")
        return pd.DataFrame(), None, None, pd.DataFrame()

def build_propensity_model(contratos, movimentos):
    try:
        vencidas = movimentos[movimentos["vencido"] & ~movimentos["status_pago"]].copy()
        if vencidas.empty:
            return pd.DataFrame()
        
        vencidas["dias_atraso_atual"] = vencidas["dias_atraso"]
        vencidas["valor_devido"] = vencidas["areceber"]
        
        cliente_hist = movimentos.groupby("idcliente").agg(
            total_pagamentos=("status_pago", "sum"),
            total_parcelas=("id", "count"),
            atraso_medio=("dias_atraso", "mean"),
            atraso_max=("dias_atraso", "max"),
            valor_total=("valorrecebido", "sum")
        ).reset_index()
        
        vencidas = vencidas.merge(cliente_hist, on="idcliente", how="left")
        vencidas["taxa_pagamento"] = vencidas["total_pagamentos"] / vencidas["total_parcelas"].replace(0, 1)
        
        def calc_score(row):
            try:
                dias = float(row["dias_atraso_atual"]) if pd.notna(row["dias_atraso_atual"]) else 0
                taxa = float(row["taxa_pagamento"]) if pd.notna(row["taxa_pagamento"]) else 0
                valor = float(row["valor_devido"]) if pd.notna(row["valor_devido"]) else 0
                
                fator_valor = 1.0 if valor < 1000 else 0.7
                score = (1 / (1 + dias / 30)) * taxa * fator_valor
                return score
            except Exception:
                return 0.0
        
        vencidas["score_propensao"] = vencidas.apply(calc_score, axis=1)
        
        max_score = vencidas["score_propensao"].max()
        if max_score > 0:
            vencidas["score_propensao"] = (vencidas["score_propensao"] / max_score * 100).round(1)
        else:
            vencidas["score_propensao"] = 0.0
        
        vencidas["categoria"] = pd.cut(
            vencidas["score_propensao"],
            bins=[-0.1, 30, 60, 80, 100.1],
            labels=["Muito Baixa", "Baixa", "Media", "Alta"]
        )
        
        required_cols = ["idcliente", "cliente", "idcontrato", "dtvenc", "dias_atraso_atual",
                         "valor_devido", "score_propensao", "categoria", "taxa_pagamento"]
        available_cols = [c for c in required_cols if c in vencidas.columns]
        
        return vencidas[available_cols]
        
    except Exception as e:
        st.error(f"Erro no propensity model: {str(e)}")
        return pd.DataFrame()

def build_anomaly_detection(contratos, movimentos):
    try:
        co = contratos.copy()
        mo = movimentos.copy()
        
        contract_features = mo.groupby("idcontrato").agg(
            total_parcelas=("id", "count"),
            parcelas_pagas=("status_pago", "sum"),
            total_recebido=("valorrecebido", "sum"),
            max_atraso=("dias_atraso", "max"),
            avg_atraso=("dias_atraso", "mean"),
            qtd_vencida=("vencido", "sum"),
        ).reset_index()
        
        co = co.merge(contract_features, left_on="id", right_on="idcontrato", how="left")
        
        for col in ["total_parcelas", "parcelas_pagas", "total_recebido", "max_atraso", "avg_atraso", "qtd_vencida"]:
            co[col] = co[col].fillna(0) if col in co.columns else 0
        
        co["pct_pagas"] = co["parcelas_pagas"] / co["total_parcelas"].replace(0, 1)
        co["pct_vencido"] = co["qtd_vencida"] / co["total_parcelas"].replace(0, 1)
        co["valor_parcela"] = co["valor_parcelado"] / co["qtd_parcela"].replace(0, 1)
        
        features = ["valor", "valor_parcela", "pct_pagas", "pct_vencido", "max_atraso", "avg_atraso"]
        available = [f for f in features if f in co.columns]
        X = co[available].fillna(0)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        co["anomaly_score"] = iso_forest.fit_predict(X_scaled)
        co["anomaly_score"] = co["anomaly_score"].map({1: "Normal", -1: "Anomalo"})
        
        co["anomaly_value"] = -iso_forest.score_samples(X_scaled)
        min_val = co["anomaly_value"].min()
        max_val = co["anomaly_value"].max()
        if max_val > min_val:
            co["anomaly_value"] = ((co["anomaly_value"] - min_val) / (max_val - min_val) * 100).round(1)
        else:
            co["anomaly_value"] = 0.0
        
        required_cols = ["id", "cliente", "usuario_nome", "valor", "anomaly_score", "anomaly_value",
                         "pct_pagas", "max_atraso"]
        available_cols = [c for c in required_cols if c in co.columns]
        
        return co[available_cols]
        
    except Exception as e:
        st.error(f"Erro na deteccao de anomalias: {str(e)}")
        return pd.DataFrame()

def build_prophet_forecast(movimentos, days=30):
    try:
        if not PROPHET_AVAILABLE:
            return None, "Prophet nao instalado"
        
        mo = movimentos[movimentos["dtvenc"].notna()].copy()
        daily = mo.groupby("dtvenc").agg(
            recebido=("valorrecebido", "sum")
        ).reset_index()
        daily = daily.rename(columns={"dtvenc": "ds", "recebido": "y"})
        daily = daily.sort_values("ds")
        
        if len(daily) < 30:
            return None, "Dados insuficientes (minimo 30 dias)"
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        model.fit(daily)
        
        future = model.make_future_dataframe(periods=days)
        forecast = model.predict(future)
        
        forecast = forecast.merge(daily[["ds", "y"]], on="ds", how="left")
        
        return forecast, None
        
    except Exception as e:
        return None, str(e)

# =============================================================================
# VISUALIZACOES AVANCADAS
# =============================================================================
def build_sankey_data(movimentos):
    try:
        total = len(movimentos)
        pagas = len(movimentos[movimentos["status_pago"]])
        vencidas = len(movimentos[movimentos["vencido"]])
        a_vencer = len(movimentos[movimentos["a_vencer"]])
        
        pagas_em_dia = len(movimentos[(movimentos["status_pago"]) & (movimentos["dias_atraso"] == 0)])
        pagas_atrasadas = len(movimentos[(movimentos["status_pago"]) & (movimentos["dias_atraso"] > 0)])
        
        vencidas_1_30 = len(movimentos[(movimentos["vencido"]) & (movimentos["dias_atraso"] <= 30)])
        vencidas_31_60 = len(movimentos[(movimentos["vencido"]) & (movimentos["dias_atraso"].between(31, 60))])
        vencidas_61_90 = len(movimentos[(movimentos["vencido"]) & (movimentos["dias_atraso"].between(61, 90))])
        vencidas_90_plus = len(movimentos[(movimentos["vencido"]) & (movimentos["dias_atraso"] > 90)])
        
        return {
            "labels": ["Total Parcelas", "Pagas", "Vencidas", "A Vencer",
                      "Pagas Em Dia", "Pagas Atrasadas",
                      "Vencidas 1-30d", "Vencidas 31-60d", "Vencidas 61-90d", "Vencidas 90+d"],
            "values": [total, pagas, vencidas, a_vencer,
                      pagas_em_dia, pagas_atrasadas,
                      vencidas_1_30, vencidas_31_60, vencidas_61_90, vencidas_90_plus],
            "source": [0, 0, 0, 1, 1, 2, 2, 2, 2],
            "target": [1, 2, 3, 4, 5, 6, 7, 8, 9]
        }
    except Exception as e:
        st.error(f"Erro no Sankey: {str(e)}")
        return None

def build_radar_data(agentes):
    try:
        if agentes.empty:
            return pd.DataFrame()
        
        agentes_radar = agentes.copy()
        
        metrics = {
            "Eficiencia": agentes_radar["Eficiencia %"] * 100,
            "Volume": (agentes_radar["Recebido (R$)"] / agentes_radar["Recebido (R$)"].max() * 100) if agentes_radar["Recebido (R$)"].max() > 0 else 0,
            "Ticket Medio": (agentes_radar["Ticket medio (R$)"] / agentes_radar["Ticket medio (R$)"].max() * 100) if agentes_radar["Ticket medio (R$)"].max() > 0 else 0,
            "Recuperacao": (1 - agentes_radar["Vencido 90+ (R$)"] / agentes_radar["Em aberto (R$)"].replace(0, 1)) * 100,
            "Juros Realizados": (agentes_radar["Juros realizados (R$)"] / agentes_radar["Juros previstos (R$)"].replace(0, 1) * 100)
        }
        
        radar_df = pd.DataFrame(metrics)
        radar_df["Agente"] = agentes_radar["Agente"]
        
        return radar_df
    except Exception as e:
        st.error(f"Erro no Radar: {str(e)}")
        return pd.DataFrame()

def build_treemap_data(movimentos):
    try:
        seg = movimentos.groupby(["nome_estabelecimento_norm"]).agg(
            valor=("areceber", "sum"),
            count=("id", "count")
        ).reset_index()
        
        seg = seg[seg["valor"] > 0].sort_values("valor", ascending=False)
        
        return seg
    except Exception as e:
        st.error(f"Erro no Treemap: {str(e)}")
        return pd.DataFrame()

def build_funnel_data(movimentos):
    try:
        total_pendencias = len(movimentos[~movimentos["status_pago"]])
        contatos = int(total_pendencias * 0.8)
        promessas = int(total_pendencias * 0.4)
        pagamentos = len(movimentos[movimentos["status_pago"]])
        
        return pd.DataFrame({
            "Etapa": ["Pendencias Totais", "Contatos Realizados", "Promessas de Pagamento", "Pagamentos Efetivados"],
            "Valor": [total_pendencias, contatos, promessas, pagamentos]
        })
    except Exception as e:
        st.error(f"Erro no Funnel: {str(e)}")
        return pd.DataFrame()

def build_bubble_chart(contratos, movimentos):
    try:
        co = contratos.copy()
        mo = movimentos.copy()
        
        contract_stats = mo.groupby("idcontrato").agg(
            valor_aberto=("areceber", lambda s: s[mo.loc[s.index, "status_pago"] == False].sum()),
            max_atraso=("dias_atraso", "max"),
            qtd_parcelas=("id", "count")
        ).reset_index()
        
        co = co.merge(contract_stats, left_on="id", right_on="idcontrato", how="left")
        
        if "valor_aberto" not in co.columns:
            co["valor_aberto"] = 0
        if "max_atraso" not in co.columns:
            co["max_atraso"] = 0
        
        co["valor_aberto"] = co["valor_aberto"].fillna(0)
        co["max_atraso"] = co["max_atraso"].fillna(0)
        
        bubble_data = co[co["valor_aberto"] > 0].copy()
        
        if bubble_data.empty:
            return pd.DataFrame()
        
        max_val = bubble_data["valor_aberto"].max()
        bubble_data["tamanho"] = bubble_data["valor_aberto"] / max_val * 100 if max_val > 0 else 0
        
        required_cols = ["id", "cliente", "valor_aberto", "max_atraso", "tamanho", "usuario_nome"]
        available_cols = [c for c in required_cols if c in bubble_data.columns]
        
        return bubble_data[available_cols]
    except Exception as e:
        st.error(f"Erro no Bubble Chart: {str(e)}")
        return pd.DataFrame()

def build_box_plot(movimentos):
    try:
        venc = movimentos[movimentos["vencido"]].copy()
        if venc.empty:
            return pd.DataFrame()
        
        return venc[["nome_estabelecimento_norm", "dias_atraso"]]
    except Exception as e:
        st.error(f"Erro no Box Plot: {str(e)}")
        return pd.DataFrame()

# =============================================================================
# MAIN
# =============================================================================
def main():
    st.set_page_config(page_title="Painel Financeiro - Microcredito", layout="wide", initial_sidebar_state="expanded")

    st.sidebar.header("🎬 Modo de Demonstração")
    
    use_fake_data = st.sidebar.checkbox(
        "Usar dados fictícios (Faker)", 
        value=True,
        help="Ative para demonstração com dados gerados aleatoriamente. Desative para usar dados reais do banco."
    )
    
    st.sidebar.header("Filtros")
    today = date.today()
    start_date = st.sidebar.date_input("Data inicial", value=date(today.year, 1, 1), max_value=today, key="start_date")
    end_date = st.sidebar.date_input("Data final", value=today, max_value=today, key="end_date")
    apply_period = st.sidebar.checkbox("Aplicar filtro de periodo", value=True)
    st.sidebar.caption("Foca em contratos ativos com parcelas vencendo no periodo.")

    usuarios, clientes, contratos, movimentos = load_data(use_fake=use_fake_data)
    
    if usuarios is None:
        st.error("Não foi possível carregar os dados. Verifique o banco de dados ou instale o Faker.")
        st.stop()
    
    if use_fake_data and FAKER_AVAILABLE:
        st.sidebar.success("✅ Usando dados fictícios gerados com Faker")
    elif use_fake_data and not FAKER_AVAILABLE:
        st.sidebar.warning("⚠️ Faker não instalado. Usando dados reais.")
    else:
        st.sidebar.info("📊 Usando dados reais do banco")
    
    usuario_choices = sorted(usuarios["usuario"].dropna().unique())
    selected_users = st.sidebar.multiselect("Filtrar por agente", usuario_choices, default=usuario_choices)

    contratos, movimentos, contratos_excluidos = filter_cancelled_contracts(contratos, movimentos)
    if selected_users and len(selected_users) < len(usuario_choices):
        contratos = contratos[contratos["usuario_nome"].isin(selected_users)]
        movimentos = movimentos[movimentos["usuario_nome"].isin(selected_users)]

    contratos_total = contratos.copy()
    movimentos_total = movimentos.copy()

    if start_date and end_date and apply_period:
        contratos, movimentos = apply_period_filter(contratos, movimentos, start_date, end_date)

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

    lgd_rates = compute_lgd_observada(movimentos_total)
    backlog_df_total = build_backlog(movimentos_total, hoje_ts)
    pdd_df_total = build_pdd(backlog_df_total, lgd_rates)
    pdd_total = pdd_df_total["PDD"].sum() if not pdd_df_total.empty else 0.0

    backlog_df = build_backlog(movimentos, hoje_ts)
    pdd_df = build_pdd(backlog_df, lgd_rates)
    pdd = pdd_df["PDD"].sum() if not pdd_df.empty else 0.0

    cf, eficiencia, eficiencia_recente = build_cashflow(movimentos, contratos, hoje_ts)
    recovery = build_recovery_curve(movimentos)
    dow = build_dow_analysis(movimentos)
    agentes = build_agent_performance(contratos, movimentos, usuarios)
    monthly_return = build_monthly_return(movimentos)
    monthly_eff = build_monthly_efficiency(movimentos)
    fpd = build_fpd(movimentos, contratos)
    roll_rate = build_roll_rate(movimentos)
    concentracao = build_concentration(movimentos)

    aberto_80_89 = movimentos.loc[movimentos["vencido"] & movimentos["dias_atraso"].between(80, 89), "areceber"].sum()
    futuro_30d = hoje_ts + pd.Timedelta(days=30)
    open_next_30 = movimentos.loc[movimentos["a_vencer"] & (movimentos["dtvenc"] <= futuro_30d), "areceber"].sum()

    best_agente = pior_agente = None
    if not agentes.empty:
        best_agente = agentes.loc[agentes["Eficiencia %"].idxmax(), "Agente"]
        pior_agente = agentes.loc[agentes["Eficiencia %"].idxmin(), "Agente"]
    best_dow = worst_dow = None
    if not dow.empty:
        best_dow = dow.loc[dow["valor"].idxmax(), "dia"]
        worst_dow = dow.loc[dow["valor"].idxmin(), "dia"]
    contratos_novos_30d = contratos.loc[contratos["dtinicio"] >= hoje_ts - pd.Timedelta(days=30), "id"].nunique()

    insights = build_insights_prescritivos(
        agentes, backlog_df, eficiencia, recebido, aberto, aberto_90, pdd,
        best_agente, pior_agente, best_dow, worst_dow, contratos_novos_30d, fpd, concentracao["hhi"],
    )

    viab = build_viability_analysis(contratos_total, movimentos_total, pdd_total)
    monthly_profit = build_monthly_profit(movimentos_total)

    st.title("🎬 Painel Financeiro - Microcredito Diario")
    st.caption("Carteira de microcredito (90 parcelas diarias - Pix). Modo de demonstração com dados fictícios.")
    
    if use_fake_data and FAKER_AVAILABLE:
        st.info("📊 **Modo de Demonstração:** Exibindo dados gerados aleatoriamente com Faker. Os dados são para fins de demonstração apenas.")

    tab_geral, tab_caixa, tab_risco, tab_agentes, tab_carteira, tab_controle, tab_rent, tab_viabilidade, tab_modelos, tab_viz, tab_dados = st.tabs(
        ["Visao Geral", "Fluxo de Caixa", "Risco & Cobranca", "Agentes", "Carteira", "Controle", "Rentabilidade", "Viabilidade & Lucro", "Modelos Preditivos", "Visualizacoes", "Dados"]
    )

    # =========================================================================
    # TAB 1 - VISAO GERAL
    # =========================================================================
    with tab_geral:
        st.subheader("Resumo executivo")
        with st.expander("Como ler esta aba", expanded=False):
            st.markdown("""
            **Resumo geral da carteira no periodo selecionado.**
            - **Programado**: valor total das parcelas com vencimento no periodo
            - **Recebido**: valor efetivamente coletado (caixa)
            - **Em aberto**: parcelas ainda nao pagas (vencidas + a vencer)
            - **Inadimplencia 90+**: saldo vencido ha mais de 90 dias (risco de perda)
            - **PDD**: provisao para devedores duvidosos (calibrada com LGD observada)
            - **FPD30**: % de contratos que inadimpliram na 1a parcela (detector de fraude)
            - **HHI**: indice de concentracao (quanto maior, mais concentrada a carteira)
            """)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Programado no periodo", fmt_brl(programado), f"{parcelas_vencidas} parcelas")
        c2.metric("Recebido no periodo", fmt_brl(recebido),
                  f"{recebido/programado:.1%} do programado" if programado else "0")
        c3.metric("Em aberto no periodo", fmt_brl(aberto), f"{parcelas_abertas} parcelas")
        c4.metric("Inadimplencia 90+", fmt_brl(aberto_90),
                  f"{aberto_90/aberto:.1%} do aberto" if aberto else "0")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Vencido (backlog)", fmt_brl(vencido),
                  f"{vencido/aberto:.1%} do aberto" if aberto else "0")
        c6.metric("A vencer", fmt_brl(a_vencer))
        c7.metric("PDD (provisao)", fmt_brl(pdd),
                  f"{pdd/aberto:.1%} do aberto" if aberto else "0")
        c8.metric("FPD30", f"{fpd['fpd_30']:.1%}",
                  f"{fpd['total_contratos_1p']} contratos analisados")

        c9, c10 = st.columns(2)
        c9.metric("HHI (concentracao)", f"{concentracao['hhi']:.0f}",
                  "Alta" if concentracao["hhi"] > 2500 else ("Media" if concentracao["hhi"] > 1500 else "Baixa"))
        c10.metric("Top 10 clientes", f"{concentracao['top10_share']:.1%}",
                   f"Top 20 = {concentracao['top20_share']:.1%}")

        st.markdown("---")
        st.markdown("#### Portfólio completo (sem filtro de periodo)")
        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.metric("Total programado", fmt_brl(programado_total), f"{len(contratos_total)} contratos")
        pc2.metric("Total recebido", fmt_brl(recebido_total),
                   f"{recebido_total/programado_total:.1%} do total" if programado_total else "0")
        pc3.metric("Total em aberto", fmt_brl(aberto_total))
        pc4.metric("PDD total", fmt_brl(pdd_total))

        eficiencia_periodo = recebido / programado if programado else 0
        st.info(f"**Periodo:** {start_date} a {end_date} | "
                f"**Eficiencia:** {eficiencia_periodo:.1%} | "
                f"**Parcelas:** {parcelas_vencidas} vencidas, {parcelas_pagas} pagas, {parcelas_abertas} em aberto")

        st.markdown("##### Composicao do saldo em aberto")
        sit_df = pd.DataFrame({
            "situacao": ["A vencer 0-30d", "A vencer 31-60d", "A vencer 60+",
                         "Vencido 1-30d", "Vencido 31-60d", "Vencido 61-90d", "Vencido 90+d"],
            "valor": [0.0] * 7,
        })
        if aberto > 0:
            pend = movimentos[~movimentos["status_pago"] & movimentos["dtvenc"].notna()].copy()
            pend["dias_situacao"] = np.where(
                pend["a_vencer"], (pend["dtvenc"] - hoje_ts).dt.days, pend["dias_atraso"].clip(lower=1))
            pend["bucket"] = np.select([
                pend["a_vencer"] & (pend["dias_situacao"] <= 30),
                pend["a_vencer"] & (pend["dias_situacao"] <= 60), pend["a_vencer"],
                ~pend["a_vencer"] & (pend["dias_situacao"] <= 30),
                ~pend["a_vencer"] & (pend["dias_situacao"] <= 60),
                ~pend["a_vencer"] & (pend["dias_situacao"] < 90), ~pend["a_vencer"],
            ], ["A vencer 0-30d", "A vencer 31-60d", "A vencer 60+",
                "Vencido 1-30d", "Vencido 31-60d", "Vencido 61-90d", "Vencido 90+d"],
                default="Vencido 90+d")
            sit = pend.groupby("bucket", as_index=False)["areceber"].sum()
            sit_df = sit_df.merge(sit, left_on="situacao", right_on="bucket", how="left", suffixes=("", "_v"))
            sit_df["valor"] = sit_df["valor"].fillna(0) + sit_df["areceber"].fillna(0)
            sit_df = sit_df[["situacao", "valor"]]
        sit_df["pct"] = (sit_df["valor"] / sit_df["valor"].sum() * 100).round(1)
        sit_df["label"] = sit_df.apply(lambda r: f"R$ {r['valor']:,.0f}\n({r['pct']}%)" if r['valor'] > 0 else "", axis=1)
        cores = ["#72B7B2", "#4C78A8", "#BAB0AC", "#F58518", "#EECA3B", "#B279A2", "#E45756"]
        fig = px.bar(sit_df, x="situacao", y="valor", text="label", color="situacao",
                     color_discrete_map=dict(zip(sit_df["situacao"], cores)),
                     title="Composicao do saldo em aberto no periodo",
                     labels={"situacao": "", "valor": "Valor (R$)"})
        fig.update_traces(texttemplate="%{text}", textposition="outside")
        fig.update_layout(height=420, showlegend=False, xaxis_tickangle=-25)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown("##### Insights prescritivos (com acao recomendada)")
        for ins in insights:
            st.markdown(f"- {ins}")

        with st.expander(" Glossario de termos tecnicos", expanded=False):
            for k, v in GLOSSARIO.items():
                st.markdown(f"**{k}**: {v}")

    # =========================================================================
    # TAB 2 - FLUXO DE CAIXA
    # =========================================================================
    with tab_caixa:
        st.subheader("Fluxo de caixa: cronograma contratual vs caixa efetivo")
        st.markdown(f"**Eficiencia historica:** {eficiencia:.1%} | **Eficiencia recente (90d):** {eficiencia_recente:.1%}")

        hist = cf[cf["mes"] <= pd.Period(hoje_ts, freq="M")]
        fut = cf[cf["mes"] > pd.Period(hoje_ts, freq="M")]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=hist["mes_ts"], y=hist["programado"], name="Programado (vencimentos)",
                             marker_color=COLORS["cinza"]))
        fig.add_trace(go.Bar(x=hist["mes_ts"], y=hist["realizado"], name="Realizado (caixa)",
                             marker_color=COLORS["azul"]))
        fig.add_trace(go.Bar(x=fut["mes_ts"], y=fut["recebivel_programado"], name="Recebiveis futuros",
                             marker_color=COLORS["roxo"]))
        fig.add_trace(go.Bar(x=fut["mes_ts"], y=fut["projetado"], name="Projecao (x eficiencia)",
                             marker_color=COLORS["verde"]))
        fig.update_layout(barmode="group", title="Cronograma contratual vs caixa efetivo (e projecao)",
                          xaxis_title="Mes", yaxis_title="Valor (R$)", height=420,
                          legend=dict(orientation="h", y=1.15), xaxis_tickformat="%b/%y")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown("##### Cenario de projecao (ajustavel)")
        col1, col2 = st.columns(2)
        efic_cenario = col1.slider("Eficiencia do cenario (%)", 30, 100,
                                    int(round(eficiencia_recente * 100)))
        rec_vencido = col2.slider("Recuperacao esperada do backlog (%)", 0, 100,
                                   int(round(eficiencia_recente * 100)))
        proj = fut[["mes_label", "recebivel_programado"]].copy()
        proj["previsao"] = proj["recebivel_programado"] * (efic_cenario / 100)
        proj = proj.rename(columns={"mes_label": "Mes", "recebivel_programado": "Programado (R$)",
                                     "previsao": "Previsao (R$)"})
        proj["Previsao acumulada (R$)"] = proj["Previsao (R$)"].cumsum()
        proj["% eficiencia"] = efic_cenario
        if not proj.empty:
            show(proj.round(2))

        st.markdown("##### Eficiencia de cobranca por mes")
        if not monthly_eff.empty:
            eff_plot = monthly_eff.copy()
            eff_plot["eficiencia"] = eff_plot["eficiencia"] * 100
            fige = go.Figure()
            fige.add_trace(go.Bar(x=eff_plot["mes_ts"], y=eff_plot["programado"], name="Programado",
                                  marker_color=COLORS["cinza"]))
            fige.add_trace(go.Bar(x=eff_plot["mes_ts"], y=eff_plot["recebido"], name="Realizado",
                                  marker_color=COLORS["azul"]))
            fige.add_trace(go.Scatter(x=eff_plot["mes_ts"], y=eff_plot["eficiencia"], name="Eficiencia (%)",
                                      yaxis="y2", mode="lines+markers+text",
                                      text=eff_plot["eficiencia"].round(0).astype(str) + "%",
                                      marker_color=COLORS["vermelho"], textposition="top center"))
            fige.update_layout(title="Programado vs realizado e eficiencia por mes",
                               height=380, yaxis2=dict(title="Eficiencia (%)", overlaying="y", side="right", range=[0, 105]),
                               legend=dict(orientation="h", y=1.15), xaxis_tickformat="%b/%y")
            st.plotly_chart(fige, use_container_width=True, config={"displayModeBar": False})

        st.markdown("---")
        st.subheader("Comportamento de pagamento")
        colA, colB = st.columns(2)
        with colA:
            st.markdown("##### Recebimento por dia da semana")
            if not dow.empty:
                dow_plot = dow.copy()
                dow_plot["pct"] = dow_plot["pct"] * 100
                figd = px.bar(dow_plot, x="dia", y="valor", text="pct", color="dia",
                              color_discrete_sequence=SEQUENCE,
                              labels={"dia": "", "valor": "Recebido (R$)"},
                              title="Sazonalidade do caixa (seg a dom)")
                figd.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                figd.update_layout(height=340, showlegend=False)
                st.plotly_chart(figd, use_container_width=True, config={"displayModeBar": False})
        with colB:
            st.markdown("##### Curva de cura (quando o cliente paga)")
            if not recovery.empty:
                rec_plot = recovery.copy()
                rec_plot["pct_acumulado"] = rec_plot["pct_acumulado"] * 100
                rec_plot["pct_individual"] = (rec_plot["valor"] / rec_plot["valor"].sum() * 100).round(1)
                rec_plot["label"] = rec_plot.apply(
                    lambda r: f"R$ {r['valor']:,.0f}\n({r['pct_individual']}%)" if r['valor'] > 0 else "", axis=1)
                figr = px.bar(rec_plot, x="janela", y="valor", color="janela",
                              color_discrete_sequence=SEQUENCE, text="label",
                              labels={"janela": "Janela apos vencimento", "valor": "Valor recebido (R$)"},
                              title="Curva de cura - quando o cliente paga")
                figr.update_traces(texttemplate="%{text}", textposition="outside")
                figr.update_layout(height=380, showlegend=False)
                st.plotly_chart(figr, use_container_width=True, config={"displayModeBar": False})
                st.caption("Acumulado: " + " → ".join(
                    [f"{r['janela']}: {r['pct_acumulado']:.1f}%" for _, r in rec_plot.iterrows() 
                     if r['janela'] in ['em dia (0d)', 'ate 30d', 'ate 60d', 'ate 90d', '90+d']]
                ))

    # =========================================================================
    # TAB 3 - RISCO & COBRANCA
    # =========================================================================
    with tab_risco:
        st.subheader("Backlog, PDD e metricas de risco")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total vencido", fmt_brl(vencido))
        c2.metric("PDD (provisao)", fmt_brl(pdd))
        c3.metric("Carteira liquida (aberto - PDD)", fmt_brl(aberto - pdd))
        c4.metric("Saldo critico 90+", fmt_brl(aberto_90))

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("FPD30", f"{fpd['fpd_30']:.1%}")
        c6.metric("FPD90", f"{fpd['fpd_90']:.1%}")
        c7.metric("HHI", f"{concentracao['hhi']:.0f}")
        c8.metric("Recovery Rate 90d",
                  f"{recovery.loc[recovery['janela']=='ate 90d', 'pct_acumulado'].values[0]:.1%}"
                  if not recovery.empty and 'ate 90d' in recovery['janela'].values else "-")

        if not backlog_df.empty:
            fig = px.bar(backlog_df, x="faixa", y="valor", text="valor", color="faixa",
                         color_discrete_sequence=SEQUENCE,
                         title="Saldo vencido por faixa de atraso",
                         labels={"faixa": "Faixa de atraso", "valor": "Valor vencido (R$)"})
            fig.update_traces(texttemplate="R$ %{text:,.0f}", textposition="outside")
            fig.update_layout(height=360, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            st.markdown("##### PDD calibrada com LGD observada")
            pdd_view = pdd_df.copy()
            pdd_view["PDD"] = pdd_view["PDD"].round(2)
            pdd_view["Valor em aberto"] = pdd_view["Valor em aberto"].round(2)
            pdd_view["% Provisao (LGD obs)"] = (pdd_view["% Provisao"] * 100).round(1).astype(str) + "%"
            pdd_view = pdd_view.rename(columns={"% Provisao": "Taxa aplicada"})
            show(pdd_view[["Faixa", "Valor em aberto", "% Provisao (LGD obs)", "PDD"]])

        st.markdown("---")
        st.subheader("Roll Rate Matrix (transicao entre faixas)")
        if not roll_rate.empty:
            roll_pivot = roll_rate.pivot_table(index="faixa", columns="faixa_next", values="pct", fill_value=0) * 100
            st.markdown("**% do saldo que migrou de uma faixa para outra** (linhas = mes M, colunas = mes M+1):")
            show(roll_pivot.round(1))
            st.caption("Leitura: da linha '1-30d', X% permaneceu em dia, Y% foi para '31-60d', Z% para '61-90d'...")
        else:
            st.info("Dados insuficientes para calcular Roll Rate (precisa de >=2 meses de historico).")

        st.markdown("---")
        st.subheader("Concentracao da carteira (Lorenz + HHI)")
        lorenz = concentracao["lorenz"]
        if not lorenz.empty:
            fig_l = go.Figure()
            fig_l.add_trace(go.Scatter(x=lorenz["% Clientes (acum)"], y=lorenz["% Clientes (acum)"],
                                        mode="lines", name="Linha de igualdade",
                                        line=dict(dash="dash", color="gray")))
            fig_l.add_trace(go.Scatter(x=lorenz["% Clientes (acum)"], y=lorenz["% Saldo (acum)"] * 100,
                                        mode="lines", name="Curva de Lorenz",
                                        line=dict(color=COLORS["azul"], width=2)))
            fig_l.update_layout(title=f"Curva de Lorenz (HHI = {concentracao['hhi']:.0f})",
                                xaxis_title="% Clientes (acumulado)", yaxis_title="% Saldo (acumulado)",
                                height=380)
            st.plotly_chart(fig_l, use_container_width=True, config={"displayModeBar": False})
            st.markdown(f"**Interpretacao:** Top 10 clientes = {concentracao['top10_share']:.1%} | "
                        f"Top 20 = {concentracao['top20_share']:.1%} | "
                        f"HHI = {concentracao['hhi']:.0f} ({'ALTA concentracao' if concentracao['hhi']>2500 else 'Media/Baixa'})")

        st.markdown("---")
        st.subheader("Plano de acao prescritivo")
        action = build_action_plan_prescritivo(aberto_90, aberto_80_89, vencido, open_next_30, pdd, eficiencia)
        show(action)

        st.markdown("---")
        st.subheader("Clientes prioritarios para cobranca")
        prio = build_priority_clients(movimentos)
        if not prio.empty:
            prio_view = prio.copy()
            prio_view["Valor em aberto"] = prio_view["Valor em aberto"].round(2)
            show(prio_view.head(20))
            top10 = prio_view.head(10)
            fig = px.bar(top10, x="Cliente", y="Valor em aberto", color="Prioridade", text="Valor em aberto",
                         color_discrete_map={"Critica": COLORS["vermelho"], "Alta": COLORS["laranja"],
                                              "Media": COLORS["amarelo"], "Baixa": COLORS["verde"]},
                         title="Top 10 clientes por valor em aberto")
            fig.update_traces(texttemplate="R$ %{text:,.0f}", textposition="outside")
            fig.update_layout(height=380, xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # =========================================================================
    # TAB 4 - AGENTES
    # =========================================================================
    with tab_agentes:
        st.subheader("Performance por agente")
        if agentes.empty:
            st.info("Sem dados de agentes no filtro atual.")
        else:
            view = agentes.copy()
            for c in view.columns:
                if "%" in c: view[c] = (view[c] * 100).round(1)
                elif "R$" in c: view[c] = view[c].round(2)
            show(view)
            c1, c2 = st.columns(2)
            with c1:
                top = agentes.sort_values("Recebido (R$)", ascending=False)
                fig = px.bar(top, x="Agente", y="Recebido (R$)", color="Eficiencia %",
                             text="Eficiencia %", color_continuous_scale="Blues",
                             title="Recebido por agente (cor = eficiencia)")
                fig.update_traces(texttemplate="%{text:.0%}", textposition="outside")
                fig.update_layout(height=360)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            with c2:
                fig2 = px.bar(top, x="Agente", y=["Em aberto (R$)", "Vencido 90+ (R$)"], barmode="group",
                              title="Saldo em aberto e inadimplencia por agente",
                              color_discrete_sequence=[COLORS["laranja"], COLORS["vermelho"]])
                fig2.update_layout(height=360, legend=dict(orientation="h", y=1.12))
                st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # =========================================================================
    # TAB 5 - CARTEIRA
    # =========================================================================
    with tab_carteira:
        st.subheader("Originacao e maturacao")
        new_ct = contratos[contratos["dtinicio"].notna()].copy()
        period_start_ts = pd.Timestamp(start_date) if start_date else None
        period_end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) if end_date else None
        if apply_period and period_start_ts and period_end_ts:
            new_ct = new_ct[(new_ct["dtinicio"] >= period_start_ts) & (new_ct["dtinicio"] < period_end_ts)].copy()

        if not new_ct.empty:
            stats_start = period_start_ts if apply_period and period_start_ts else new_ct["dtinicio"].min()
            stats_end = period_end_ts - pd.Timedelta(days=1) if apply_period and period_end_ts else new_ct["dtinicio"].max()
            novos_stats = build_new_contract_stats(new_ct, stats_start, stats_end)
            novos_stats_view = pd.DataFrame(novos_stats).T.reset_index().rename(columns={"index": "Periodicidade"})
            new_ct["mes"] = new_ct["dtinicio"].dt.to_period("M").dt.to_timestamp()
            by_month = new_ct.groupby("mes").size().rename("qtd").reset_index()
            by_month_val = new_ct.groupby("mes")["valor"].sum().rename("valor").reset_index()

            c1, c2 = st.columns(2)
            c1.metric("Contratos no periodo", len(new_ct))
            c2.metric("Ticket medio", fmt_brl(new_ct["valor"].mean()))
            st.markdown("##### Novos contratos por periodicidade")
            show(novos_stats_view.round({"Media": 1, "Maximo": 0, "Minimo": 0}))

            fig = go.Figure()
            fig.add_trace(go.Bar(x=by_month["mes"], y=by_month["qtd"], name="Qtd contratos",
                                 marker_color=COLORS["azul"], yaxis="y"))
            fig.add_trace(go.Scatter(x=by_month_val["mes"], y=by_month_val["valor"], name="Valor originado (R$)",
                                     mode="lines+markers", marker_color=COLORS["laranja"], yaxis="y2"))
            fig.update_layout(title="Novos contratos por mes", height=360,
                              yaxis2=dict(title="Valor (R$)", overlaying="y", side="right"),
                              legend=dict(orientation="h", y=1.15))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown("---")
        st.subheader("Maturacao por coorte (vintage)")
        vintage = build_vintage(movimentos, contratos)
        if apply_period and period_start_ts:
            try:
                min_coorte = pd.Period(period_start_ts, freq="M")
                vintage = vintage[vintage["coorte"].apply(lambda c: pd.Period(c) >= min_coorte)]
            except Exception:
                pass
        if not vintage.empty:
            mat = vintage.pivot_table(index="coorte", columns="janela_dias", values="pct_programado") * 100
            st.markdown("**Maturacao: % do cronograma que ja venceu**")
            show(mat.round(1))
            eff = vintage.pivot_table(index="coorte", columns="janela_dias", values="pct_realizado_do_programado") * 100
            st.markdown("**Eficiencia de recebimento: % do vencido que ja foi recebido**")
            show(eff.round(1))
            fig = px.line(vintage, x="janela_dias", y="pct_realizado_do_programado", color="coorte", markers=True,
                          title="Eficiencia de recebimento por coorte (maturacao)")
            fig.update_layout(height=380, legend=dict(orientation="h", y=1.15))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown("---")
        st.subheader("Segmentos (estabelecimento)")
        seg = movimentos.groupby("nome_estabelecimento_norm").agg(
            Recebido=("valorrecebido", "sum"),
            Em_aberto=("areceber", lambda s: s[movimentos.loc[s.index, "status_pago"] == False].sum()),
            Clientes=("idcliente", "nunique"),
        ).reset_index().rename(columns={"nome_estabelecimento_norm": "Segmento"})
        seg["Total"] = seg["Recebido"] + seg["Em_aberto"]
        seg["Eficiencia %"] = seg["Recebido"] / seg["Total"].replace(0, 1)
        seg = seg.sort_values("Total", ascending=False)
        view_seg = seg.copy()
        view_seg["Eficiencia %"] = (view_seg["Eficiencia %"] * 100).round(1)
        view_seg[["Recebido", "Em_aberto", "Total"]] = view_seg[["Recebido", "Em_aberto", "Total"]].round(2)
        show(view_seg)

    # =========================================================================
    # TAB 6 - CONTROLE
    # =========================================================================
    with tab_controle:
        st.subheader("Controle de carteira - contratos e exclusoes")
        st.markdown("##### Resumo por situacao (portfolio completo)")
        status_df = contratos_total.groupby("status").agg(
            Contratos=("id", "count"),
            Principal=("valor", "sum"),
            A_receber=("valor_parcelado", "sum"),
        ).reset_index()
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
            show(excl_view.rename(columns={
                "id": "Contrato", "dtinicio": "Dt inicio", "dtfim": "Dt fim",
                "status": "Status", "valor": "Principal (R$)", "valor_parcelado": "A receber (R$)",
            }))
            st.caption(f"Total excluido: {fmt_brl(contratos_excluidos['valor'].sum())} em {len(contratos_excluidos)} contratos.")
        else:
            st.info("Nenhum contrato excluido.")

    # =========================================================================
    # TAB 7 - RENTABILIDADE
    # =========================================================================
    with tab_rent:
        st.subheader("Rentabilidade do produto")
        frac_pond = (contratos["valor"].sum() / contratos["valor_parcelado"].sum()) if contratos["valor_parcelado"].sum() else 0
        juros_aberto_est = aberto * (1 - frac_pond)
        principal_aberto_est = aberto * frac_pond

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Principal", fmt_brl(principal))
        c2.metric("Juros previstos", fmt_brl(juros_previstos),
                  f"{juros_previstos/principal:.1%} do principal" if principal else "")
        c3.metric("Juros realizados (caixa)", fmt_brl(juros_realizados))
        c4.metric("Descontos concedidos", fmt_brl(desconto_total))

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Principal em aberto (est.)", fmt_brl(principal_aberto_est))
        c6.metric("Juros em aberto (est.)", fmt_brl(juros_aberto_est))
        c7.metric("Retorno realizado", f"{recebido/principal:.1%}" if principal else "0")
        c8.metric("Ticket medio", fmt_brl(contratos["valor"].mean()) if len(contratos) else "0")

        st.markdown("---")
        st.subheader("Perfil demografico")
        col1, col2 = st.columns(2)
        with col1:
            gen = movimentos.groupby("genero_cat").agg(
                Recebido=("valorrecebido", "sum"),
                Em_aberto=("areceber", lambda s: s[movimentos.loc[s.index, "status_pago"] == False].sum())
            ).reset_index()
            gen["Total"] = gen["Recebido"] + gen["Em_aberto"]
            gen["Eficiencia %"] = gen["Recebido"] / gen["Total"].replace(0, 1)
            fig = px.bar(gen, x="genero_cat", y="Total", color="genero_cat",
                         color_discrete_sequence=SEQUENCE, text="Total",
                         title="Valor movimentado por genero")
            fig.update_traces(texttemplate="R$ %{text:,.0f}", textposition="outside")
            fig.update_layout(height=340, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with col2:
            age = movimentos.groupby("faixa_idade").agg(
                Recebido=("valorrecebido", "sum"),
                Em_aberto=("areceber", lambda s: s[movimentos.loc[s.index, "status_pago"] == False].sum())
            ).reset_index()
            age["Total"] = age["Recebido"] + age["Em_aberto"]
            age["Eficiencia %"] = age["Recebido"] / age["Total"].replace(0, 1)
            fig = px.bar(age, x="faixa_idade", y="Total", color="Eficiencia %",
                         color_continuous_scale="Blues", text="Total",
                         title="Valor movimentado por faixa etaria")
            fig.update_traces(texttemplate="R$ %{text:,.0f}", textposition="outside")
            fig.update_layout(height=340, xaxis_tickangle=-20)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # =========================================================================
    # TAB 8 - VIABILIDADE & LUCRO
    # =========================================================================
    with tab_viabilidade:
        st.subheader("💰 Viabilidade Financeira: Lucro vs Risco")
        st.caption("Responde: quanto tive de lucro? quanto por mes? vale o investimento e risco pela margem?")

        with st.expander("Como ler esta aba", expanded=False):
            st.markdown("""
            **Esta aba responde as 3 perguntas cruciais do negocio:**
            1. **Quanto tive de lucro?** → Lucro Bruto Real = Juros recebidos em caixa - Descontos concedidos.
               *Atencao:* o Total Recebido NAO e lucro - a maior parte dele e devolucao do seu proprio dinheiro (Principal Recuperado).
            2. **Quanto de lucro por mes?** → Grafico mensal mostra o lucro liquido entrando no caixa mês a mês, com a margem de lucro (%).
            3. **Vale o investimento e risco?** → O sistema compara o Lucro Bruto com a PDD (Provisao de Perdas).
               Se o lucro cobre pelo menos **2x** a PDD, a operacao e saudavel. Caso contrario, a margem esta apertada.
            """)

        st.markdown("### 1. Resumo do Investimento vs Retorno (Acumulado)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💵 Total Investido (Principal)", fmt_brl(viab["total_investido"]))
        c2.metric("💸 Total Recebido (Caixa)", fmt_brl(viab["total_recebido"]))
        c3.metric("🔄 Principal Recuperado", fmt_brl(viab["principal_recuperado"]),
                  help="Parte do dinheiro recebido que e devolucao do seu investimento, NAO lucro.")
        c4.metric(" ROI Bruto Realizado", f"{viab['roi_bruto_pct']:.1f}%",
                  help="Lucro bruto sobre o total investido.")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("💹 Juros Recebidos", fmt_brl(viab["juros_recebidos"]))
        c6.metric("🎁 Descontos Concedidos", fmt_brl(viab["descontos"]))
        c7.metric("✅ Lucro Bruto Real", fmt_brl(viab["lucro_bruto_real"]),
                  help="Juros recebidos - Descontos = seu lucro efetivo ate agora.")
        c8.metric("📊 Margem de Lucro", f"{viab['margem_lucro_pct']:.1f}%",
                  help="Para cada R$ 1 de principal recuperado, quanto virou lucro.")

        st.markdown("---")
        st.markdown("### 2. O Veredito: Vale a pena o risco?")

        if viab["lucro_bruto_real"] > 0 and viab["cobertura_risco"] >= 2.0:
            st.success(f"""
            ✅ **SIM, a operacao e VIÁVEL e SAUDÁVEL.**
            - Seu **Lucro Bruto Real** e de **{fmt_brl(viab['lucro_bruto_real'])}**.
            - Esse lucro cobre **{viab['cobertura_risco']:.1f}x** a Provisao de Perdas (PDD) estimada de **{fmt_brl(viab['pdd_total'])}**.
            - Mesmo descontando o risco de inadimplencia, seu **Lucro Líquido Ajustado ao Risco** e positivo: **{fmt_brl(viab['lucro_liquido_ajustado'])}**.
            - **Margem de lucro:** {viab['margem_lucro_pct']:.1f}% sobre o principal recuperado.
            **Recomendacao:** A operacao esta gerando valor. Considere escalar com cautela.
            """)
        elif viab["lucro_bruto_real"] > 0 and viab["cobertura_risco"] >= 1.0:
            st.warning(f"""
            ⚠️ **VIÁVEL, mas com MARGEM APERTADA.**
            - Voce teve lucro bruto de **{fmt_brl(viab['lucro_bruto_real'])}**, mas ele cobre apenas **{viab['cobertura_risco']:.1f}x** o risco de perda (PDD de {fmt_brl(viab['pdd_total'])}).
            - Lucro Líquido Ajustado ao Risco: **{fmt_brl(viab['lucro_liquido_ajustado'])}**.
            **Recomendacao:** Qualquer aumento na inadimplencia pode corroer todo o lucro.
            """)
        elif viab["lucro_bruto_real"] > 0:
            st.warning(f"""
            ⚠️ **LUCRO CONTÁBIL, mas INSUFICIENTE para cobrir o risco.**
            - Lucro bruto: **{fmt_brl(viab['lucro_bruto_real'])}** (positivo).
            - PDD estimada: **{fmt_brl(viab['pdd_total'])}** (maior que o lucro).
            - Lucro Líquido Ajustado ao Risco: **{fmt_brl(viab['lucro_liquido_ajustado'])}** (negativo).
            """)
        else:
            st.error(f"""
            ❌ **OPERACAO EM PREJUÍZO CONTÁBIL.**
            - O lucro bruto ({fmt_brl(viab['lucro_bruto_real'])}) ja e negativo.
            - PDD estimada: {fmt_brl(viab['pdd_total'])}.
            **Acao Imediata:** Suspenda novas originacoes e acione cobranca externa intensiva.
            """)

        st.markdown("##### Indicadores de Saude da Operacao")
        hc1, hc2, hc3, hc4 = st.columns(4)
        hc1.metric("Cobertura do Risco", f"{viab['cobertura_risco']:.1f}x",
                   help="Quantas vezes o lucro cobre a PDD. Ideal: >= 2x")
        hc2.metric("Lucro Liquido Ajustado", fmt_brl(viab['lucro_liquido_ajustado']))
        hc3.metric("PDD / Total Investido", f"{(viab['pdd_total']/viab['total_investido']*100) if viab['total_investido']>0 else 0:.1f}%")
        hc4.metric("ROI Ajustado ao Risco", f"{(viab['lucro_liquido_ajustado']/viab['total_investido']*100) if viab['total_investido']>0 else 0:.1f}%")

        st.markdown("---")
        st.markdown("### 3. Lucro Líquido Mes a Mes")
        if not monthly_profit.empty:
            lucro_medio_mes = monthly_profit["lucro_bruto"].mean()
            lucro_total_acum = monthly_profit["lucro_bruto"].sum()
            margem_media = monthly_profit["margem_lucro_pct"].mean()
            meses_positivos = (monthly_profit["lucro_bruto"] > 0).sum()

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Lucro Medio Mensal", fmt_brl(lucro_medio_mes))
            k2.metric("Lucro Total Acumulado", fmt_brl(lucro_total_acum))
            k3.metric("Margem Media Mensal", f"{margem_media:.1f}%")
            k4.metric("Meses com Lucro Positivo", f"{meses_positivos}/{len(monthly_profit)}")

            fig_profit = go.Figure()
            fig_profit.add_trace(go.Bar(
                x=monthly_profit["mes_ts"], y=monthly_profit["lucro_bruto"],
                name="Lucro Bruto (Juros - Descontos)",
                marker_color=[COLORS["verde"] if v >= 0 else COLORS["vermelho"] for v in monthly_profit["lucro_bruto"]],
                text=monthly_profit["lucro_bruto"].apply(lambda x: fmt_brl(x)),
                textposition="outside"
            ))
            fig_profit.add_trace(go.Scatter(
                x=monthly_profit["mes_ts"], y=monthly_profit["margem_lucro_pct"],
                name="Margem de Lucro (%)", yaxis="y2", mode="lines+markers+text",
                marker_color=COLORS["azul"],
                text=monthly_profit["margem_lucro_pct"].round(1).astype(str) + "%",
                textposition="top center"
            ))
            fig_profit.update_layout(
                title="Evolucao do Lucro Bruto e Margem Mensal",
                xaxis_title="Mes de Recebimento", yaxis_title="Valor (R$)",
                yaxis2=dict(title="Margem (%)", overlaying="y", side="right"),
                height=420, legend=dict(orientation="h", y=1.15), xaxis_tickformat="%b/%y"
            )
            st.plotly_chart(fig_profit, use_container_width=True, config={"displayModeBar": False})

            st.markdown("##### Detalhamento Mensal")
            view_profit = monthly_profit[["mes_label", "total_recebido", "principal_recebido",
                                          "juros_recebidos", "descontos", "lucro_bruto", "margem_lucro_pct"]].copy()
            view_profit = view_profit.rename(columns={
                "mes_label": "Mes", "total_recebido": "Recebido Total",
                "principal_recebido": "(-) Principal Devolvido", "juros_recebidos": "(=) Juros Brutos",
                "descontos": "(-) Descontos", "lucro_bruto": "(=) Lucro Liquido", "margem_lucro_pct": "Margem %"
            })
            for col in ["Recebido Total", "(-) Principal Devolvido", "(=) Juros Brutos", "(-) Descontos", "(=) Lucro Liquido"]:
                view_profit[col] = view_profit[col].round(2)
            view_profit["Margem %"] = view_profit["Margem %"].round(1)
            show(view_profit)

        st.markdown("---")
        st.markdown("### 4. Simulador de Cenarios (E se...?)")
        sc1, sc2, sc3 = st.columns(3)
        cenario_inadimplencia = sc1.slider("Aumento de inadimplencia (%)", 0, 50, 0)
        cenario_descontos = sc2.slider("Aumento de descontos (%)", 0, 100, 0)
        cenario_juros = sc3.slider("Reducao de juros recebidos (%)", 0, 50, 0)

        juros_cenario = viab["juros_recebidos"] * (1 - cenario_juros/100)
        descontos_cenario = viab["descontos"] * (1 + cenario_descontos/100)
        pdd_cenario = viab["pdd_total"] * (1 + cenario_inadimplencia/100)
        lucro_cenario = juros_cenario - descontos_cenario
        lucro_liquido_cenario = lucro_cenario - pdd_cenario
        cobertura_cenario = lucro_cenario / pdd_cenario if pdd_cenario > 0 else 0

        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("Lucro Bruto no Cenario", fmt_brl(lucro_cenario), f"{(lucro_cenario - viab['lucro_bruto_real']):+.0f} vs atual")
        cc2.metric("Lucro Liquido Ajustado", fmt_brl(lucro_liquido_cenario), f"{(lucro_liquido_cenario - viab['lucro_liquido_ajustado']):+.0f} vs atual")
        cc3.metric("Cobertura do Risco", f"{cobertura_cenario:.1f}x", f"{(cobertura_cenario - viab['cobertura_risco']):+.1f}x vs atual")
        cc4.metric("Status do Cenario", "✅ Saudavel" if cobertura_cenario >= 2 else ("⚠️ Apertado" if cobertura_cenario >= 1 else "❌ Em risco"))

    # =========================================================================
    # TAB 9 - MODELOS PREDITIVOS
    # =========================================================================
    with tab_modelos:
        st.subheader("🤖 Modelos Preditivos")
        
        if not SKLEARN_AVAILABLE:
            st.error("scikit-learn nao instalado! Instale com: pip install scikit-learn")
        else:
            st.markdown("### 1️ PD - Probability of Default")
            if st.button("🚀 Treinar Modelo PD", key="train_pd"):
                with st.spinner("Treinando modelo..."):
                    X, y, co_features, feature_cols = prepare_features_pd(contratos_total, movimentos_total)
                    
                    if len(X) == 0:
                        st.error("Erro ao preparar features.")
                    else:
                        model, scaler, metrics, feature_imp, X_test, y_test, y_prob = train_pd_model(X, y, feature_cols)
                        
                        if model is not None:
                            st.success("✅ Modelo treinado com sucesso!")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Acuracia", f"{metrics['accuracy']:.1%}")
                            col2.metric("AUC-ROC", f"{metrics['auc_roc']:.3f}")
                            col3.metric("Precisao", f"{metrics['precision']:.1%}")
                            col4.metric("Recall", f"{metrics['recall']:.1%}")
                            
                            if not feature_imp.empty:
                                st.markdown("#### Importancia das Features")
                                fig_imp = px.bar(feature_imp.head(10), x="importance", y="feature", 
                                               orientation="h", title="Top 10 features mais importantes")
                                fig_imp.update_layout(height=400)
                                st.plotly_chart(fig_imp, use_container_width=True)
                            
                            st.markdown("#### Matriz de Confusao")
                            cm = confusion_matrix(y_test, model.predict(scaler.transform(X_test)))
                            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                             labels={"x": "Previsto", "y": "Real"},
                                             x=["Nao Default", "Default"], y=["Nao Default", "Default"])
                            fig_cm.update_layout(height=300, width=400)
                            st.plotly_chart(fig_cm, use_container_width=False)
                            
                            st.markdown("#### Scores PD por Contrato")
                            scores = predict_pd_scores(model, scaler, X, feature_cols)
                            co_features["score_pd"] = scores
                            co_features["risco_pd"] = pd.cut(scores, bins=[0, 0.2, 0.5, 0.8, 1.0],
                                                              labels=["Baixo", "Medio", "Alto", "Critico"])
                            
                            scores_view = co_features[["id", "cliente", "usuario_nome", "valor", 
                                                      "score_pd", "risco_pd", "default_90d"]].copy()
                            scores_view["score_pd"] = (scores_view["score_pd"] * 100).round(1)
                            scores_view = scores_view.sort_values("score_pd", ascending=False)
                            show(scores_view.head(20).rename(columns={
                                "id": "Contrato", "cliente": "Cliente", "usuario_nome": "Agente",
                                "valor": "Principal (R$)", "score_pd": "Score PD (%)", 
                                "risco_pd": "Classificacao", "default_90d": "Default Real"
                            }))
            
            st.markdown("---")
            
            st.markdown("### 2️⃣ Clustering de Clientes")
            if st.button("🚀 Executar Clustering", key="run_cluster"):
                with st.spinner("Executando clustering..."):
                    co_cluster, kmeans, scaler_cluster, cluster_summary = build_clustering_model(
                        contratos_total, movimentos_total
                    )
                    
                    if not co_cluster.empty:
                        st.success("✅ Clustering executado com sucesso!")
                        
                        st.markdown("#### Resumo dos Clusters")
                        show(cluster_summary)
                        
                        st.markdown("#### Distribuicao de Clientes")
                        cluster_dist = co_cluster["segmento_comportamental"].value_counts().reset_index()
                        cluster_dist.columns = ["Cluster", "Qtd Clientes"]
                        fig = px.pie(cluster_dist, values="Qtd Clientes", names="Cluster",
                                    title="Distribuicao de clientes por cluster",
                                    color_discrete_sequence=SEQUENCE)
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("#### Clientes por Cluster")
                        cluster_view = co_cluster[["id", "cliente", "usuario_nome", "valor", 
                                                  "segmento_comportamental", "max_atraso", "pct_pagas"]].copy()
                        cluster_view["pct_pagas"] = (cluster_view["pct_pagas"] * 100).round(1)
                        cluster_view = cluster_view.sort_values("segmento_comportamental")
                        show(cluster_view.head(30).rename(columns={
                            "id": "Contrato", "cliente": "Cliente", "usuario_nome": "Agente",
                            "valor": "Principal (R$)", "segmento_comportamental": "Cluster",
                            "max_atraso": "Maior Atraso", "pct_pagas": "% Parcelas Pagas"
                        }))
            
            st.markdown("---")
            
            st.markdown("### 3️⃣ Propensity to Pay")
            if st.button("🚀 Executar Propensity to Pay", key="propensity"):
                with st.spinner("Calculando propensao..."):
                    propensity = build_propensity_model(contratos_total, movimentos_total)
                    
                    if not propensity.empty:
                        st.success("✅ Propensao calculada!")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Alta propensao", len(propensity[propensity["categoria"] == "Alta"]))
                        col2.metric("Media propensao", len(propensity[propensity["categoria"] == "Media"]))
                        col3.metric("Baixa propensao", len(propensity[propensity["categoria"] == "Baixa"]))
                        col4.metric("Muito baixa", len(propensity[propensity["categoria"] == "Muito Baixa"]))
                        
                        cat_dist = propensity["categoria"].value_counts().reset_index()
                        cat_dist.columns = ["Categoria", "Quantidade"]
                        fig = px.pie(cat_dist, values="Quantidade", names="Categoria",
                                    title="Distribuicao de propensao a pagar",
                                    color_discrete_sequence=[COLORS["verde"], COLORS["azul"], 
                                                            COLORS["laranja"], COLORS["vermelho"]])
                        st.plotly_chart(fig, use_container_width=True)
                        
                        prop_view = propensity.sort_values("score_propensao", ascending=False)
                        prop_view = prop_view[["cliente", "dtvenc", "dias_atraso_atual", 
                                              "valor_devido", "score_propensao", "categoria"]].copy()
                        prop_view["dtvenc"] = prop_view["dtvenc"].dt.strftime("%d/%m/%Y")
                        prop_view.columns = ["Cliente", "Vencimento", "Dias atraso", 
                                           "Valor devido (R$)", "Score", "Categoria"]
                        show(prop_view.head(20))
            
            st.markdown("---")
            
            st.markdown("### 4️⃣ Detecao de Anomalias")
            if st.button("🚀 Executar Detecao de Anomalias", key="anomaly"):
                with st.spinner("Detectando anomalias..."):
                    anomalies = build_anomaly_detection(contratos_total, movimentos_total)
                    
                    if not anomalies.empty:
                        st.success("✅ Anomalias detectadas!")
                        
                        n_anomalous = len(anomalies[anomalies["anomaly_score"] == "Anomalo"])
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Contratos analisados", len(anomalies))
                        col2.metric("Anomalos detectados", n_anomalous)
                        col3.metric("% Anomalos", f"{n_anomalous/len(anomalies)*100:.1f}%")
                        
                        fig = px.histogram(anomalies, x="anomaly_value", color="anomaly_score",
                                          title="Distribuicao de scores de anomalia",
                                          labels={"anomaly_value": "Score de Anomalia", "count": "Quantidade"},
                                          color_discrete_map={"Normal": COLORS["azul"], "Anomalo": COLORS["vermelho"]})
                        st.plotly_chart(fig, use_container_width=True)
                        
                        anom_view = anomalies[anomalies["anomaly_score"] == "Anomalo"].copy()
                        anom_view = anom_view.sort_values("anomaly_value", ascending=False)
                        anom_view["pct_pagas"] = (anom_view["pct_pagas"] * 100).round(1)
                        anom_view = anom_view[["cliente", "usuario_nome", "valor", "anomaly_value",
                                              "pct_pagas", "max_atraso"]]
                        anom_view.columns = ["Cliente", "Agente", "Principal (R$)", "Score Anomalia",
                                           "% Pagas", "Maior Atraso"]
                        show(anom_view.head(20))
            
            st.markdown("---")
            
            st.markdown("### 5️ Prophet Forecast")
            if st.button(" Executar Prophet Forecast", key="prophet"):
                if not PROPHET_AVAILABLE:
                    st.error("Prophet nao instalado! Instale com: pip install prophet")
                else:
                    with st.spinner("Treinando Prophet..."):
                        forecast, error = build_prophet_forecast(movimentos_total, days=30)
                        
                        if error:
                            st.error(f"Erro: {error}")
                        elif forecast is not None:
                            st.success("✅ Forecast Prophet calculado!")
                            
                            fig = go.Figure()
                            
                            hist_data = forecast[forecast["ds"] <= pd.Timestamp(date.today())].copy()
                            if not hist_data.empty and 'y' in hist_data.columns:
                                fig.add_trace(go.Scatter(
                                    x=hist_data["ds"], y=hist_data["y"],
                                    mode="markers", name="Historico",
                                    marker=dict(color=COLORS["azul"], size=6)
                                ))
                            
                            fig.add_trace(go.Scatter(
                                x=forecast["ds"], y=forecast["yhat"],
                                mode="lines", name="Previsao",
                                line=dict(color=COLORS["verde"], width=2)
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=forecast["ds"], y=forecast["yhat_upper"],
                                mode="lines", name="Limite superior",
                                line=dict(color=COLORS["verde"], width=0),
                                showlegend=False
                            ))
                            fig.add_trace(go.Scatter(
                                x=forecast["ds"], y=forecast["yhat_lower"],
                                mode="lines", name="Limite inferior",
                                line=dict(color=COLORS["verde"], width=0),
                                fill="tonexty", fillcolor="rgba(84, 162, 75, 0.2)",
                                showlegend=False
                            ))
                            
                            fig.update_layout(
                                title="Forecast de recebimentos - Prophet (proximos 30 dias)",
                                xaxis_title="Data", yaxis_title="Valor (R$)",
                                height=400, hovermode="x unified"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            future_forecast = forecast[forecast["ds"] > pd.Timestamp(date.today())]
                            total_previsto = future_forecast["yhat"].sum()
                            media_diaria = future_forecast["yhat"].mean()
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Total previsto (30 dias)", fmt_brl(total_previsto))
                            col2.metric("Media diaria", fmt_brl(media_diaria))
                            col3.metric("Dia de maior recebimento", 
                                       fmt_brl(future_forecast["yhat"].max()))

    # =========================================================================
    # TAB 10 - VISUALIZACOES AVANCADAS
    # =========================================================================
    with tab_viz:
        st.subheader("📊 Visualizacoes Avancadas")
        
        st.markdown("### 1️⃣ Sankey Diagram - Fluxo de Status")
        if st.button("🚀 Gerar Sankey", key="sankey"):
            sankey_data = build_sankey_data(movimentos_total)
            
            if sankey_data:
                fig = go.Figure(data=[go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=sankey_data["labels"],
                        color=SEQUENCE[:len(sankey_data["labels"])]
                    ),
                    link=dict(
                        source=sankey_data["source"],
                        target=sankey_data["target"],
                        value=sankey_data["values"]
                    )
                )])
                
                fig.update_layout(title="Fluxo de Status de Parcelas", font_size=12, height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### 2️⃣ Radar Chart - Perfil de Agentes")
        if st.button(" Gerar Radar", key="radar"):
            radar_data = build_radar_data(agentes)
            
            if not radar_data.empty:
                fig = go.Figure()
                
                for _, row in radar_data.iterrows():
                    fig.add_trace(go.Scatterpolar(
                        r=[row["Eficiencia"], row["Volume"], row["Ticket Medio"], 
                           row["Recuperacao"], row["Juros Realizados"], row["Eficiencia"]],
                        theta=["Eficiencia", "Volume", "Ticket Medio", "Recuperacao", 
                              "Juros Realizados", "Eficiencia"],
                        fill='toself',
                        name=row["Agente"]
                    ))
                
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    title="Perfil Multidimensional de Agentes",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### 3️⃣ Treemap - Hierarquia de Segmentos")
        if st.button("🚀 Gerar Treemap", key="treemap"):
            treemap_data = build_treemap_data(movimentos_total)
            
            if not treemap_data.empty:
                fig = px.treemap(
                    treemap_data,
                    path=[px.Constant("Carteira"), "nome_estabelecimento_norm"],
                    values="valor",
                    color="valor",
                    color_continuous_scale="Blues",
                    title="Distribuicao de Valor por Segmento"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### 4️⃣ Funil de Cobranca")
        if st.button("🚀 Gerar Funil", key="funnel"):
            funnel_data = build_funnel_data(movimentos_total)
            
            if not funnel_data.empty:
                fig = go.Figure(go.Funnel(
                    y=funnel_data["Etapa"],
                    x=funnel_data["Valor"],
                    textinfo="value+percent initial",
                    marker=dict(color=SEQUENCE[:len(funnel_data)])
                ))
                
                fig.update_layout(title="Funil de Cobranca", height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### 5️⃣ Bubble Chart - Valor x Atraso x Risco")
        if st.button("🚀 Gerar Bubble Chart", key="bubble"):
            bubble_data = build_bubble_chart(contratos_total, movimentos_total)
            
            if not bubble_data.empty and "max_atraso" in bubble_data.columns:
                fig = px.scatter(
                    bubble_data,
                    x="max_atraso",
                    y="valor_aberto",
                    size="tamanho",
                    color="usuario_nome",
                    hover_name="cliente",
                    labels={
                        "max_atraso": "Maior Atraso (dias)",
                        "valor_aberto": "Valor em Aberto (R$)",
                        "usuario_nome": "Agente"
                    },
                    title="Contratos em Aberto (tamanho = valor em aberto)",
                    size_max=60
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Sem dados para gerar Bubble Chart.")
        
        st.markdown("---")
        
        st.markdown("### 6️⃣ Box Plot - Distribuicao de Atrasos por Segmento")
        if st.button("🚀 Gerar Box Plot", key="boxplot"):
            box_data = build_box_plot(movimentos_total)
            
            if not box_data.empty:
                top_segments = box_data["nome_estabelecimento_norm"].value_counts().head(10).index
                box_data = box_data[box_data["nome_estabelecimento_norm"].isin(top_segments)]
                
                fig = px.box(
                    box_data,
                    x="nome_estabelecimento_norm",
                    y="dias_atraso",
                    color="nome_estabelecimento_norm",
                    title="Distribuicao de Atrasos por Segmento (Top 10)",
                    labels={
                        "nome_estabelecimento_norm": "Segmento",
                        "dias_atraso": "Dias de Atraso"
                    }
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    # =========================================================================
    # TAB 11 - DADOS
    # =========================================================================
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