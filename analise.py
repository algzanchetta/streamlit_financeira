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

import re
import unicodedata


def _normalizar_texto(texto: str) -> str:
    """Remove acentos, coloca em maiúsculas e normaliza espaços."""
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = texto.upper()
    texto = re.sub(r"[^A-Z0-9]+", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def classificar_estabelecimento(name: str):
    """Retorna (categoria, subcategoria) em formato claro para o estabelecimento."""
    if not isinstance(name, str) or not name.strip():
        return ("OUTROS", "OUTROS")

    x = _normalizar_texto(name)

    regras = [
        ("TRANSPORTE", ["UBER", "MOTO TAXI", "MOTOTAXI", "MOTORISTA", "MOTORISTA APP", "ENTREGADOR", "DELIVERY", "VIAGEM", "ONIBUS", "LOZAMO", "TAXI", "APP"]),
        ("ALIMENTAÇÃO", ["PEIXARIA", "PEIXE", "ACOUGUE", "ACOUQUE", "FRUTARIA", "VERDURAS", "QUITANDA", "SORVETERIA", "SORVETE", "ACAI", "PIZZARIA", "PIZZA", "ESPETINHO", "ESPETARIA", "PADARIA", "PANIFICADORA", "SALGADO", "SALGADOS", "CONFEITARIA", "BOLO", "DOCES", "CAFETERIA", "CAFE", "RESTAURANTE", "COMIDA", "LANCHONETE", "LANCHE", "BAR", "BUTECO", "PUB", "FAST FOOD", "KIBAB"]),
        ("BELEZA", ["SALAO", "BELEZA", "BARBEARIA", "BARBEIRO", "UNHA", "MANICURE", "DEPIL", "BRONZE", "TATUAGEM", "PENTEADO", "MAQUIAGEM", "ESTETICA"]),
        ("TECNOLOGIA", ["CELULAR", "CELULARES", "INFORMATICA", "ELETRONICA", "ELETROTECNICO", "ELETRONICO", "COMPUTADOR", "NOTEBOOK", "TABLET", "SMARTPHONE", "ASSISTENCIA"]),
        ("AUTOMOTIVO", ["BORRACHARIA", "FUNILARIA", "LAVA JATO", "LAVAJATO", "MECANICA", "OFICINA", "MOTO", "MOTOS", "AUTO PECAS", "PNEUS", "PNEU", "PINTURA", "ELETRO AUTO"]),
        ("CONSTRUÇÃO", ["SERRALHERIA", "VIDRACARIA", "VIDROS", "REFRIGERACAO", "CLIMATIZACAO", "PEDREIRO", "OBRAS", "CONSTRUCAO", "CHAVEIRO", "MATERIAIS", "MADEIRAS", "FERREIRA", "PISO", "REVESTIMENTO"]),
        ("COMÉRCIO", ["GAS", "OTICA", "OPTICA", "MERCADO", "MERCADINHO", "SUPERMERCADO", "CONVENIENCIA", "ROUPA", "MODA", "SAPATO", "CALCADOS", "SEMIJOIA", "PERFUME", "VARIEDADES", "MOVEIS", "FILTROS", "LIMPEZA", "FARMACIA", "PET SHOP", "PETSHOP", "PAPELARIA", "BOUTIQUE", "JOIAS", "COSMETICOS", "ARTESANATO", "AQUARIO", "FLORICULTURA", "FLORES"]),
        ("HOSPEDAGEM", ["HOTEL", "PENSAO", "HOSTEL", "PENSÃO"]),
    ]

    for categoria, tokens in regras:
        for token in tokens:
            if token in x:
                if categoria == "ALIMENTAÇÃO":
                    if "PEIXARIA" in x or "PEIXE" in x:
                        return (categoria, "PEIXARIA")
                    if "ACOUGUE" in x or "ACOUQUE" in x:
                        return (categoria, "ACOUGUE/CARNES")
                    if "FRUTARIA" in x or "VERDURAS" in x:
                        return (categoria, "FRUTARIA/VERDURAS")
                    if "QUITANDA" in x:
                        return (categoria, "QUITANDA")
                    if "SORVETERIA" in x or "SORVETE" in x:
                        return (categoria, "SORVETERIA")
                    if "ACAI" in x:
                        return (categoria, "ACAI")
                    if "PIZZARIA" in x or "PIZZA" in x:
                        return (categoria, "PIZZARIA")
                    if "ESPETINHO" in x or "ESPETARIA" in x or "CHURRASQUINHO" in x:
                        return (categoria, "ESPETINHO/CHURRASQUINHO")
                    if "PADARIA" in x or "PANIFICADORA" in x or "PANIFICACAO" in x:
                        return (categoria, "PADARIA/PANIFICADORA")
                    if "SALGADO" in x:
                        return (categoria, "SALGADOS")
                    if "CONFEITARIA" in x or "BOLO" in x or "DOCES" in x:
                        return (categoria, "DOCES/BOLOS")
                    if "CAFETERIA" in x or "CAFE" in x:
                        return (categoria, "CAFETERIA/CAFE")
                    if "RESTAURANTE" in x or "COMIDA" in x:
                        return (categoria, "RESTAURANTE")
                    if "LANCHONETE" in x or "LANCHE" in x:
                        return (categoria, "LANCHONETE")
                    if re.search(r"\bBAR\b", x) or "BUTECO" in x or "PUB" in x:
                        return (categoria, "BAR")
                    return (categoria, token)
                if categoria == "BELEZA":
                    if "SALAO" in x or "BELEZA" in x:
                        return (categoria, "SALAO DE BELEZA")
                    if "BARBEARIA" in x or "BARBEIRO" in x:
                        return (categoria, "BARBEARIA")
                    if "UNHA" in x or "MANICURE" in x:
                        return (categoria, "MANICURE/UNHAS")
                    if "DEPIL" in x:
                        return (categoria, "ESTETICA/DEPILACAO")
                    if "BRONZE" in x:
                        return (categoria, "BRONZEAMENTO")
                    if "TATTO" in x or "TATUAGEM" in x:
                        return (categoria, "TATUAGEM")
                    return (categoria, token)
                if categoria == "TECNOLOGIA":
                    if "CELULAR" in x or "CELULARES" in x:
                        return (categoria, "CELULARES/ACESSORIOS")
                    if "INFORMATICA" in x or "ELETRONICA" in x:
                        return (categoria, "TECNOLOGIA/INFORMATICA")
                    return (categoria, token)
                if categoria == "AUTOMOTIVO":
                    if "BORRACHARIA" in x:
                        return (categoria, "BORRACHARIA")
                    if "FUNILARIA" in x:
                        return (categoria, "FUNILARIA/PINTURA")
                    if "LAVA JATO" in x or "LAVAJATO" in x:
                        return (categoria, "LAVA-JATO")
                    if "MECANICA" in x or "OFICINA" in x:
                        return (categoria, "OFICINA MECANICA")
                    if "MOTO" in x or "MOTOS" in x:
                        return (categoria, "OFICINA/MOTOS")
                    return (categoria, token)
                if categoria == "CONSTRUÇÃO":
                    if "SERRALHERIA" in x:
                        return (categoria, "SERRALHERIA")
                    if "VIDRO" in x or "VIDRACARIA" in x:
                        return (categoria, "VIDRACARIA")
                    if "REFRIGERACAO" in x or "CLIMATIZACAO" in x:
                        return (categoria, "REFRIGERACAO/CLIMATIZACAO")
                    if "PEDREIRO" in x or "OBRAS" in x or "CONSTRUCAO" in x:
                        return (categoria, "CONSTRUCAO/OBRAS")
                    if "CHAVEIRO" in x:
                        return (categoria, "CHAVEIRO")
                    return (categoria, token)
                if categoria == "COMÉRCIO":
                    if "GAS" in x:
                        return (categoria, "GAS/AGUA")
                    if "OTICA" in x or "OPTICA" in x:
                        return (categoria, "OTICA")
                    if "MERCADO" in x or "MERCADINHO" in x or "SUPERMERCADO" in x:
                        return (categoria, "MERCADO/MERCADINHO")
                    if "CONVENIENCIA" in x:
                        return (categoria, "CONVENIENCIA")
                    if "ROUPA" in x or "MODA" in x:
                        return (categoria, "VESTUARIO")
                    if "SAPATO" in x or "CALCADOS" in x:
                        return (categoria, "CALCADOS")
                    if "SEMIJOIA" in x:
                        return (categoria, "SEMIJOIAS/ACESSORIOS")
                    if "PERFUME" in x or "COSMETICO" in x:
                        return (categoria, "COSMETICOS/PERFUMARIA")
                    if "VARIEDADES" in x or "NOVIDADES" in x:
                        return (categoria, "VARIEDADES")
                    if "MOVEIS" in x:
                        return (categoria, "MOVEIS")
                    if "FILTROS" in x:
                        return (categoria, "FILTROS")
                    if "LIMPEZA" in x:
                        return (categoria, "MATERIAL DE LIMPEZA")
                    return (categoria, token)
                if categoria == "HOSPEDAGEM":
                    if "HOTEL" in x:
                        return (categoria, "HOTEL")
                    if "PENSAO" in x:
                        return (categoria, "PENSAO")
                    return (categoria, token)
                if categoria == "TRANSPORTE":
                    if "MOTO TAXI" in x or "MOTOTAXI" in x or "MOTORISTA APP" in x:
                        return (categoria, "MOTORISTA APP")
                    if "UBER" in x or "MOTORISTA" in x:
                        return (categoria, "TRANSPORTE")
                    return (categoria, token)
                return (categoria, token)

    return ("OUTROS", "OUTROS")


def normalize_estabelecimento(name: str) -> str:
    """Compatibilidade: devolve apenas a subcategoria para o restante do sistema."""
    _, subcategoria = classificar_estabelecimento(name)
    return subcategoria


def ensure_datetime(df, columns):
    """Garante que as colunas especificadas sejam datetime."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

# =============================================================================
# CARREGAMENTO DE DADOS
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
    # 2. PROCESSAMENTO COMUM
    # =============================================================================
    
    # --- Clientes ---
    clientes = ensure_datetime(clientes, ["dtinicio", "dtfim", "dtatualizacao"])
    clientes["idade"] = pd.to_numeric(clientes.get("idade"), errors="coerce")
    clientes["idade"] = clientes["idade"].where(clientes["idade"] > 0, np.nan)
    clientes["genero"] = clientes.get("genero").astype(str).str.strip()
    clientes["genero_cat"] = clientes["genero"].replace({"1": "Masculino", "0": "Feminino"}).fillna("Outro")
    
    idade_bins = [0, 18, 25, 35, 45, 55, 65, 200]
    idade_labels = ["<18", "18-25", "26-35", "36-45", "46-55", "56-65", ">65"]
    clientes["faixa_idade"] = pd.cut(clientes["idade"], bins=idade_bins, labels=idade_labels)
    clientes["faixa_idade"] = clientes["faixa_idade"].cat.add_categories(["Sem idade"]).fillna("Sem idade")
    clientes["avaliacao"] = clientes["avaliacao"].astype(str).fillna("Nao avaliado")
    clientes["nome_estabelecimento_original"] = clientes["nome_estabelecimento"].astype(str).fillna("Desconhecido")
    clientes["nome_estabelecimento"] = clientes["nome_estabelecimento_original"]
    clientes[["categoria_estabelecimento", "subcategoria_estabelecimento"]] = clientes["nome_estabelecimento_original"].apply(
        lambda v: pd.Series(classificar_estabelecimento(v))
    )
    clientes["nome_estabelecimento_norm"] = clientes["subcategoria_estabelecimento"]

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

    movimentos["vencido"] = (~movimentos["status_pago"]) & movimentos["dtvenc"].notna() & (movimentos["dtvenc"] < today)
    movimentos["a_vencer"] = (~movimentos["status_pago"]) & movimentos["dtvenc"].notna() & (movimentos["dtvenc"] >= today)
    movimentos["atraso_90"] = movimentos["vencido"] & (movimentos["dias_atraso"] >= 90)

    movimentos = movimentos.merge(
        clientes[["id", "cliente", "genero_cat", "faixa_idade", "avaliacao", "nome_estabelecimento"]],
        left_on="idcliente", right_on="id", how="left", suffixes=("", "_cliente"),
    )
    movimentos["nome_estabelecimento_original"] = movimentos["nome_estabelecimento"].fillna("Desconhecido")
    movimentos["nome_estabelecimento"] = movimentos["nome_estabelecimento_original"].astype(str)
    movimentos[["categoria_estabelecimento", "subcategoria_estabelecimento"]] = movimentos["nome_estabelecimento_original"].apply(
        lambda v: pd.Series(classificar_estabelecimento(v))
    )
    movimentos["nome_estabelecimento_norm"] = movimentos["subcategoria_estabelecimento"]
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
    st_ts = pd.Timestamp(start_date)
    en_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    
    movimentos = movimentos.copy()
    movimentos["dtvenc"] = pd.to_datetime(movimentos["dtvenc"], errors='coerce')
    
    mask = movimentos["dtvenc"].notna()
    mask = mask & (movimentos["dtvenc"] >= st_ts) & (movimentos["dtvenc"] < en_ts)
    
    movimentos_f = movimentos[mask].copy()
    
    contratos_ids = movimentos_f["idcontrato"].unique()
    contratos_f = contratos[contratos["id"].isin(contratos_ids)].copy()
    return contratos_f, movimentos_f

# =============================================================================
# FUNÇÕES DE ANÁLISE
# =============================================================================
def build_cashflow(movimentos, contratos, today, n_future=6, period_start=None, period_end=None):
    mo = movimentos.copy()
    mo["dtvenc"] = pd.to_datetime(mo["dtvenc"], errors='coerce')
    mo["dtrecebimento"] = pd.to_datetime(mo["dtrecebimento"], errors='coerce')

    if period_start is not None and period_end is not None:
        start_period = pd.Timestamp(period_start).to_period("M")
        end_month_boundary = pd.Timestamp(period_end) - pd.Timedelta(days=1)
        end_period = end_month_boundary.to_period("M")
        sched = mo.dropna(subset=["dtvenc"]).copy()
        sched = sched[(sched["dtvenc"] >= period_start) & (sched["dtvenc"] < period_end)]
        sched = sched.assign(mes=lambda d: d["dtvenc"].dt.to_period("M")).groupby("mes")["parcela"].sum()
        real = mo.dropna(subset=["dtrecebimento"]).copy()
        real = real[(real["dtrecebimento"] >= period_start) & (real["dtrecebimento"] < period_end)]
        real = real.assign(mes=lambda d: d["dtrecebimento"].dt.to_period("M")).groupby("mes")["valorrecebido"].sum()
        idx = pd.period_range(start_period, end_period, freq="M")
    else:
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

def build_monthly_return(movimentos, period_start=None, period_end=None):
    mo = movimentos.copy()
    if "juros_frac" not in mo.columns or "frac_principal" not in mo.columns:
        return pd.DataFrame()
    mo = mo[mo["dtvenc"].notna()].copy()
    if period_start is not None and period_end is not None:
        mo = mo[(mo["dtvenc"] >= period_start) & (mo["dtvenc"] < period_end)].copy()
    mo["mes_venc"] = mo["dtvenc"].dt.to_period("M")
    if period_start is not None and period_end is not None:
        start_period = pd.Timestamp(period_start).to_period("M")
        end_month_boundary = pd.Timestamp(period_end) - pd.Timedelta(days=1)
        end_period = end_month_boundary.to_period("M")
        all_months = pd.period_range(start_period, end_period, freq="M")
    else:
        all_months = mo["mes_venc"].dropna().unique()
    g = mo.groupby("mes_venc").agg(
        programado=("parcela", "sum"),
        total_recebido=("valorrecebido", "sum"),
        parcelas=("id", "count"),
    ).reindex(all_months, fill_value=0).reset_index().rename(columns={"index": "mes_venc"})
    rec = mo[mo["dtrecebimento"].notna()].copy()
    if period_start is not None and period_end is not None:
        rec = rec[(rec["dtrecebimento"] >= period_start) & (rec["dtrecebimento"] < period_end)].copy()
    rec["juros_rec"] = rec["valorrecebido"] * rec["juros_frac"]
    rec["princ_rec"] = rec["valorrecebido"] * rec["frac_principal"]
    rec_by_mes = rec.groupby("mes_venc")["juros_rec"].sum() if not rec.empty else pd.Series(dtype=float)
    g["juros_recebidos"] = g["mes_venc"].map(rec_by_mes).fillna(0)
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

def build_monthly_efficiency(movimentos, period_start=None, period_end=None):
    mo = movimentos.copy()
    if period_start is not None and period_end is not None:
        mo = mo[(mo["dtvenc"] >= period_start) & (mo["dtvenc"] < period_end)].copy()
    mo["mes_venc"] = mo["dtvenc"].dt.to_period("M")
    if period_start is not None and period_end is not None:
        start_period = pd.Timestamp(period_start).to_period("M")
        end_month_boundary = pd.Timestamp(period_end) - pd.Timedelta(days=1)
        end_period = end_month_boundary.to_period("M")
        all_months = pd.period_range(start_period, end_period, freq="M")
    else:
        all_months = mo["mes_venc"].dropna().unique()
    g = (mo.dropna(subset=["mes_venc"]).groupby("mes_venc").agg(programado=("parcela", "sum"), recebido=("valorrecebido", "sum"), parcelas=("id", "count")).reindex(all_months, fill_value=0).reset_index().rename(columns={"index": "mes_venc"}))
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

def build_segment_summary(movimentos, dimension="subcategoria_estabelecimento"):
    """Resumo agregando por categoria/subcategoria/segmento para uso em gráficos e análise."""
    if dimension not in movimentos.columns:
        dimension = "nome_estabelecimento_norm" if "nome_estabelecimento_norm" in movimentos.columns else None
    if dimension is None:
        return pd.DataFrame(columns=["Segmento", "Recebido", "Em_aberto", "Total", "Clientes", "Eficiencia %"])

    base = movimentos[[dimension, "valorrecebido", "areceber", "status_pago", "idcliente"]].copy()
    base[dimension] = base[dimension].fillna("OUTROS")
    recebido = base.groupby(dimension, dropna=False)["valorrecebido"].sum().rename("Recebido")
    em_aberto = base.loc[~base["status_pago"]].groupby(dimension, dropna=False)["areceber"].sum().rename("Em_aberto")
    clientes = base.groupby(dimension, dropna=False)["idcliente"].nunique().rename("Clientes")
    seg = pd.concat([recebido, em_aberto, clientes], axis=1).fillna(0).reset_index().rename(columns={dimension: "Segmento"})
    seg["Total"] = seg["Recebido"] + seg["Em_aberto"]
    seg["Eficiencia %"] = seg["Recebido"] / seg["Total"].replace(0, np.nan)
    seg = seg.sort_values("Total", ascending=False).reset_index(drop=True)
    return seg


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

def build_new_contract_stats(contratos, start_date, end_date, dimension=None):
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    recent = contratos[
        contratos["dtinicio"].notna() &
        (contratos["dtinicio"] >= start) &
        (contratos["dtinicio"] <= end) &
        (contratos["dtinicio"].dt.dayofweek < 6)
    ].copy()

    if dimension is not None and dimension in recent.columns:
        recent["segmento_dim"] = recent[dimension].fillna("OUTROS")
        by_dim = recent.groupby("segmento_dim").agg(
            Contratos=("id", "count"),
            Valor_originado=("valor", "sum"),
            Ticket_medio=("valor", "mean"),
        ).reset_index().rename(columns={"segmento_dim": "Segmento"})
        by_dim = by_dim.sort_values(["Contratos", "Valor_originado"], ascending=[False, False]).reset_index(drop=True)
        return by_dim

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
# FUNÇÕES DE VIABILIDADE E LUCRO (CORRIGIDAS)
# =============================================================================
def build_monthly_profit(movimentos, period_start=None, period_end=None):
    """Calcula o lucro mensal com base na DATA DE RECEBIMENTO."""
    rec = movimentos[movimentos["dtrecebimento"].notna()].copy()
    
    # IMPORTANTE: Filtrar pela data de RECEBIMENTO, não vencimento
    if period_start is not None and period_end is not None:
        rec = rec[(rec["dtrecebimento"] >= period_start) & (rec["dtrecebimento"] < period_end)].copy()
    
    if rec.empty:
        return pd.DataFrame()
    
    rec["mes_recebimento"] = rec["dtrecebimento"].dt.to_period("M")
    
    if "frac_principal" not in rec.columns:
        rec["frac_principal"] = 0.0
    if "juros_frac" not in rec.columns:
        rec["juros_frac"] = 0.0
        
    rec["principal_recebido"] = rec["valorrecebido"] * rec["frac_principal"]
    rec["juros_recebidos"] = rec["valorrecebido"] * rec["juros_frac"]
    
    # Definir todos os meses do período filtrado
    if period_start is not None and period_end is not None:
        start_period = pd.Timestamp(period_start).to_period("M")
        end_month_boundary = pd.Timestamp(period_end) - pd.Timedelta(days=1)
        end_period = end_month_boundary.to_period("M")
        all_months = pd.period_range(start_period, end_period, freq="M")
    else:
        all_months = rec["mes_recebimento"].unique()
    
    profit_df = rec.groupby("mes_recebimento").agg(
        total_recebido=("valorrecebido", "sum"),
        principal_recebido=("principal_recebido", "sum"),
        juros_recebidos=("juros_recebidos", "sum"),
        descontos=("desconto", "sum")
    ).reindex(all_months, fill_value=0).reset_index().rename(columns={"index": "mes_recebimento"})
    
    profit_df["lucro_bruto"] = profit_df["juros_recebidos"] - profit_df["descontos"]
    profit_df["margem_lucro_pct"] = (
        profit_df["lucro_bruto"] / profit_df["principal_recebido"].replace(0, np.nan)
    ).fillna(0) * 100
    profit_df["mes_label"] = profit_df["mes_recebimento"].astype(str)
    profit_df["mes_ts"] = profit_df["mes_recebimento"].dt.to_timestamp()
    
    return profit_df

def build_viability_analysis(contratos, movimentos, pdd_total, period_start=None, period_end=None):
    """Calcula a viabilidade financeira com base na DATA DE RECEBIMENTO."""
    mov = movimentos[movimentos["dtrecebimento"].notna()].copy()
    
    # Filtrar pela data de RECEBIMENTO
    if period_start is not None and period_end is not None:
        mov = mov[(mov["dtrecebimento"] >= period_start) & (mov["dtrecebimento"] < period_end)].copy()
    
    total_investido = contratos["valor"].sum()
    total_recebido = mov["valorrecebido"].sum()
    total_descontos = mov["desconto"].sum()
    
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

def build_payment_profile(movimentos, by):
    """Perfil de comportamento de pagamento por uma ou mais dimensões."""
    mo = movimentos[~movimentos["dtvenc"].isna()].copy()
    g = mo.groupby(by)
    df = g.agg(
        Clientes=("idcliente", "nunique"),
        Parcelas=("id", "count"),
        Recebido=("valorrecebido", "sum"),
        Em_aberto=("areceber", lambda s: s[mo.loc[s.index, "status_pago"] == False].sum()),
        Aberto_90=("areceber", lambda s: s[mo.loc[s.index, "atraso_90"]].sum()),
        Parcelas_pagas=("status_pago", "sum"),
        Atraso_medio=("dias_atraso", "mean"),
    ).reset_index()
    df["Total"] = df["Recebido"] + df["Em_aberto"]
    df["Eficiencia %"] = df["Recebido"] / df["Total"].replace(0, np.nan)
    df["% Parcelas pagas"] = df["Parcelas_pagas"] / df["Parcelas"].replace(0, np.nan)
    df["% Aberto 90+"] = df["Aberto_90"] / df["Em_aberto"].replace(0, np.nan)
    df["Atraso medio (dias)"] = df["Atraso_medio"].round(1)
    df = df.sort_values("Eficiencia %", ascending=False)
    return df

def _grupo_label(row, label_cols):
    """Constrói rótulo legível do grupo."""
    if not label_cols:
        return "-"
    partes = []
    for col in label_cols:
        if col in row.index:
            v = row[col]
            if isinstance(v, (pd.Period, pd.Interval)):
                v = str(v)
            partes.append(str(v))
    return " + ".join(partes) if partes else "-"

def comparar_pagadores(profile_df, label_cols):
    """Identifica o melhor e o pior pagador."""
    if profile_df is None or profile_df.empty:
        return None, None, None, None, None
    df = profile_df.copy()
    df = df[df["Clientes"] > 0]
    df = df[df["Eficiencia %"].notna()]
    if df.empty:
        return None, None, None, None, None
    melhor = df.sort_values("Eficiencia %", ascending=False).iloc[0]
    pior = df.sort_values("Eficiencia %", ascending=True).iloc[0]
    dif = melhor["Eficiencia %"] - pior["Eficiencia %"]
    igual = dif < 0.05 and len(df) > 1
    m_label = _grupo_label(melhor, label_cols)
    p_label = _grupo_label(pior, label_cols)
    return melhor, pior, igual, m_label, p_label

def build_coorte_recebimento(movimentos, by):
    """Por grupo, mede recebimento e inadimplência."""
    mo = movimentos[~movimentos["dtvenc"].isna()].copy()
    g = mo.groupby(by)
    df = g.agg(
        Contratos=("idcontrato", "nunique"),
        Clientes=("idcliente", "nunique"),
        Parcelas=("id", "count"),
        Recebido_geral=("valorrecebido", "sum"),
        Vencido=("areceber", lambda s: s[mo.loc[s.index, "vencido"]].sum()),
        Atraso_90=("areceber", lambda s: s[mo.loc[s.index, "atraso_90"]].sum()),
        Total_mov=("areceber", "sum"),
    ).reset_index()
    df["Inadimplencia"] = df["Vencido"]
    df["% Inadimplencia"] = df["Inadimplencia"] / df["Total_mov"].replace(0, np.nan)
    df["Recebido_geral"] = df["Recebido_geral"].round(2)
    df["Vencido"] = df["Vencido"].round(2)
    df["Atraso_90"] = df["Atraso_90"].round(2)
    df["Inadimplencia"] = df["Inadimplencia"].round(2)
    df["Contratos"] = df["Contratos"].astype(int)
    df["Clientes"] = df["Clientes"].astype(int)
    df["Parcelas"] = df["Parcelas"].astype(int)
    return df

def top_coorte(df, col, label_cols, n=3, descendente=True):
    """Retorna os n grupos de maior/menor valor."""
    if df is None or df.empty or col not in df.columns:
        return []
    d = df[df[col].notna()].sort_values(col, ascending=not descendente).head(n)
    out = []
    for _, r in d.iterrows():
        lbl = _grupo_label(r, label_cols)
        out.append((lbl, r[col]))
    return out

FAIXAS_VALOR_CONTRATO = [0, 1000, 2000, 3000, 4000, 5000, 1_000_000]

def build_perfil_valor_contrato(contratos):
    """Por faixa de valor de contrato, mede recebimento e inadimplência."""
    cc = contratos.dropna(subset=["valor"]).copy()
    if cc.empty:
        return pd.DataFrame()
    cc["Faixa Valor"] = pd.cut(cc["valor"], bins=FAIXAS_VALOR_CONTRATO,
                               right=False, include_lowest=True)
    g = cc.groupby("Faixa Valor", observed=True)
    df = g.agg(
        Contratos=("id", "count"),
        Clientes=("idcliente", "nunique"),
        Valor_total=("valor", "sum"),
        Recebido=("total_recebido", "sum"),
        Aberto=("total_aberto", "sum"),
        Vencido=("vencido_valor", "sum"),
        Default_90d=("default_90d", "sum"),
        Receb_pct_med=("percentual_recebido", "mean"),
    ).reset_index()
    df["Faixa Valor"] = df["Faixa Valor"].astype(str)
    df["Default_%"] = df["Default_90d"] / df["Contratos"].replace(0, np.nan)
    df["% Inadimplencia"] = df["Vencido"] / (df["Aberto"] + df["Recebido"]).replace(0, np.nan)
    for col in ["Valor_total", "Recebido", "Aberto", "Vencido"]:
        df[col] = df[col].round(2)
    df = df.sort_values("Valor_total", ascending=False)
    return df

def build_perfil_por_faixas(contratos, passo=500, faixas_selecionadas=None):
    """Por faixa de valor em intervalos de 'passo' reais (ex.: R$500), mede risco e inadimplência.

    Permite filtrar apenas as faixas escolhidas (ex.: 500 a 5000). Retorna DataFrame com
    colunas de faixa, abertura (lo/hi), recebimento, inadimplência e default."""
    cc = contratos.dropna(subset=["valor"]).copy()
    if cc.empty:
        return pd.DataFrame()
    lo_val = int(cc["valor"].min() // passo) * passo
    hi_val = int(cc["valor"].max() // passo + 1) * passo
    bins = list(range(lo_val, hi_val + 1, passo))
    if len(bins) < 2:
        return pd.DataFrame()
    cc["Faixa Valor"] = pd.cut(cc["valor"], bins=bins, right=False, include_lowest=True)
    g = cc.groupby("Faixa Valor", observed=True)
    df = g.agg(
        Contratos=("id", "count"),
        Clientes=("idcliente", "nunique"),
        Valor_total=("valor", "sum"),
        Recebido=("total_recebido", "sum"),
        Aberto=("total_aberto", "sum"),
        Vencido=("vencido_valor", "sum"),
        Default_90d=("default_90d", "sum"),
    ).reset_index()
    df["Faixa Valor"] = df["Faixa Valor"].astype(str)
    inter = cc["Faixa Valor"].cat.categories
    lo_map = {str(iv): iv.left for iv in inter}
    hi_map = {str(iv): iv.right for iv in inter}
    df["lo"] = df["Faixa Valor"].map(lo_map)
    df["hi"] = df["Faixa Valor"].map(hi_map)
    df["Default_%"] = df["Default_90d"] / df["Contratos"].replace(0, np.nan)
    df["% Inadimplencia"] = df["Vencido"] / (df["Aberto"] + df["Recebido"]).replace(0, np.nan)
    if faixas_selecionadas:
        df = df[df["Faixa Valor"].isin(faixas_selecionadas)]
    for col in ["Valor_total", "Recebido", "Aberto", "Vencido"]:
        df[col] = df[col].round(2)
    df = df.sort_values("lo")
    return df

def build_coorte_valor_contrato(contratos):
    """Coorte por faixa de valor de contrato."""
    cc = contratos.dropna(subset=["valor"]).copy()
    if cc.empty:
        return pd.DataFrame()
    cc["Faixa Valor"] = pd.cut(cc["valor"], bins=FAIXAS_VALOR_CONTRATO,
                               right=False, include_lowest=True)
    g = cc.groupby("Faixa Valor", observed=True)
    df = g.agg(
        Contratos=("id", "count"),
        Clientes=("idcliente", "nunique"),
        Parcelas=("id", "count"),
        Recebido_geral=("total_recebido", "sum"),
        Vencido=("vencido_valor", "sum"),
        Atraso_90=("default_90d", lambda s: int(s.sum())),
    ).reset_index()
    df["Faixa Valor"] = df["Faixa Valor"].astype(str)
    df["Inadimplencia"] = df["Vencido"]
    denom = (df["Vencido"] + df["Recebido_geral"]).replace(0, np.nan)
    df["% Inadimplencia"] = df["Vencido"] / denom
    for col in ["Recebido_geral", "Vencido", "Atraso_90", "Inadimplencia"]:
        df[col] = df[col].round(2)
    df = df.sort_values("Recebido_geral", ascending=False)
    return df

def perfil_cliente(contratos, idcliente):
    """Histórico de relacionamento de um cliente."""
    h = contratos[contratos["idcliente"] == idcliente]
    if h.empty:
        return {"n_contratos": 0, "default": 0, "receb_pct": None,
                "valor_medio": None, "valor_solicitado": None}
    return {
        "n_contratos": int(h["id"].nunique()),
        "default": int(h["default_90d"].sum()),
        "receb_pct": float(h["percentual_recebido"].mean()),
        "valor_medio": float(h["valor"].mean()),
        "valor_solicitado": float(h["valor"].max()),
    }

def classificar_risco_cliente(perfil):
    """Classifica o risco do cliente."""
    n = perfil["n_contratos"]
    if n == 0:
        return "Novo"
    if perfil["default"] > 0:
        return "Risco"
    rp = perfil["receb_pct"]
    if rp is None:
        return "Recorrente"
    if n >= 2 and rp >= 0.75:
        return "Confiavel"
    return "Recorrente"

def recomendar_valor_contrato(perfil, valor_solicitado=None, valor_max_teto=None):
    """Estratégia de valor inicial do contrato."""
    risco = classificar_risco_cliente(perfil)
    solicitado = valor_solicitado if valor_solicitado is not None else perfil.get("valor_solicitado")
    base = solicitado if solicitado and solicitado > 0 else (perfil.get("valor_medio") or 0)
    teto = valor_max_teto if valor_max_teto is not None else base
    if base <= 0:
        base = 1000.0

    regras = {
        "Novo": dict(ini=0.30, passo=0.25, desc="cliente novo, sem histórico"),
        "Risco": dict(ini=0.25, passo=0.20, desc="histórico com inadimplência 90+"),
        "Recorrente": dict(ini=0.55, passo=0.25, desc="relação recorrente, sem default"),
        "Confiavel": dict(ini=0.75, passo=0.25, desc="relação sólida, ≥2 contratos pagos"),
    }
    regra = regras.get(risco, regras["Novo"])

    inicial = round(teto * regra["ini"], 2)
    limite_fixo = round(teto * 0.75, 2) if teto else inicial

    plano = []
    cap = inicial
    etapa = 1
    while cap < limite_fixo - 1:
        cap = min(round(cap * (1 + regra["passo"]), 2), limite_fixo)
        plano.append({"etapa": etapa,
                      "capacidade": cap,
                      "pct_teto": cap / teto if teto else 0,
                      "condicao": f"{etapa} contrato(s) pago(s) em dia"})
        etapa += 1
    if teto and round(limite_fixo, 2) < round(teto, 2):
        plano.append({"etapa": etapa,
                      "capacidade": round(teto, 2),
                      "pct_teto": 1.0,
                      "condicao": "consistência de longo prazo"})

    return {
        "classe": risco,
        "descricao": regra["desc"],
        "teto": teto,
        "valor_inicial": inicial,
        "pct_inicial": regra["ini"],
        "limite_fixo": limite_fixo,
        "plano": plano,
    }

_IDADE_BINS = [0, 18, 25, 35, 45, 55, 65, 200]
_IDADE_LABELS = ["<18", "18-25", "26-35", "36-45", "46-55", "56-65", ">65"]

def _idade_para_faixa(idade):
    if idade is None:
        return None
    try:
        idd = int(idade)
    except Exception:
        return None
    if idd <= 0:
        return None
    label = pd.cut([idd], bins=_IDADE_BINS, labels=_IDADE_LABELS)[0]
    return str(label)

def analisar_viabilidade_perfil(movimentos, contratos, genero=None, idade=None,
                                segmento=None, valor=None, categoria=None, subcategoria=None):
    """Simula a viabilidade de conceder um contrato a um perfil."""
    faixa_idade = _idade_para_faixa(idade)
    mo = movimentos[~movimentos["dtvenc"].isna()].copy()
    amostra = mo.copy()

    if genero and genero != "Todos":
        amostra = amostra[amostra["genero_cat"] == genero]
    if faixa_idade:
        amostra = amostra[amostra["faixa_idade"] == faixa_idade]
    if segmento and segmento != "Sem segmento":
        amostra = amostra[amostra["nome_estabelecimento_norm"] == segmento]
    if categoria and categoria != "Todas":
        amostra = amostra[amostra["categoria_estabelecimento"] == categoria]
    if subcategoria and subcategoria != "Todas":
        amostra = amostra[amostra["subcategoria_estabelecimento"] == subcategoria]

    n_clientes = amostra["idcliente"].nunique()
    n_contratos = amostra["idcontrato"].nunique()
    recebido = float(amostra["valorrecebido"].sum())
    aberto_np = float(amostra.loc[~amostra["status_pago"], "areceber"].sum())
    vencido = float(amostra.loc[amostra["vencido"], "areceber"].sum())
    aberto_90 = float(amostra.loc[amostra["atraso_90"], "areceber"].sum())

    eficiencia = recebido / (recebido + aberto_np) if (recebido + aberto_np) > 0 else None
    inad_pct = vencido / (vencido + recebido) if (vencido + recebido) > 0 else 0.0
    grave_90_pct = aberto_90 / vencido if vencido > 0 else 0.0

    base_default = None
    previa = contratos[contratos["idcliente"].isin(amostra["idcliente"].unique())]
    if len(previa) > 0:
        base_default = float(previa["default_90d"].mean())

    if eficiencia is None:
        tendencia = "Sem dados"
        veredito = "Não avaliável (sem histórico para o perfil)"
        valor_inicial = (valor or 1000) * 0.30
    else:
        if inad_pct > 0.72 or eficiencia < 0.62 or (base_default is not None and base_default > 0.55):
            tendencia = "Alta"
        elif inad_pct > 0.58 or eficiencia < 0.70:
            tendencia = "Média"
        else:
            tendencia = "Baixa"

        risco_faixa = 0.5
        if valor:
            vp = build_perfil_valor_contrato(contratos)
            if vp is not None and not vp.empty:
                alvo = valor
                for _, r in vp.iterrows():
                    if "Faixa Valor" in r.index and str(r["Faixa Valor"]).startswith("["):
                        try:
                            lo, hi = str(r["Faixa Valor"]).strip("[])").split(",")
                            lo = float(lo); hi = float(hi)
                            if lo <= alvo < hi:
                                risco_faixa = float(r["Default_%"]) if pd.notna(r["Default_%"]) else 0.5
                                break
                        except Exception:
                            continue
            if tendencia == "Alta" or risco_faixa > 0.55:
                veredito = "Alto risco — conceder valor menor ou recusar."
                valor_inicial = (valor or 1000) * 0.25
            elif tendencia == "Média" or risco_faixa > 0.45:
                veredito = "Viável com cautela — conceder valor reduzido."
                valor_inicial = (valor or 1000) * 0.40
            else:
                veredito = "Viável — perfil tende a não inadimplir."
                valor_inicial = (valor or 1000) * 0.60
        else:
            veredito = ("Perfil tende a inadimplência" if tendencia in ("Alta", "Média")
                        else "Perfil tende a não inadimplir")
            valor_inicial = None

    return {
        "genero": genero or "Todos",
        "idade": idade,
        "faixa_idade": faixa_idade or "Todas",
        "segmento": segmento or "Sem segmento",
        "categoria": categoria,
        "subcategoria": subcategoria,
        "valor_solicitado": valor,
        "amostra_contratos": n_contratos,
        "amostra_clientes": n_clientes,
        "recebido": recebido,
        "aberto": aberto_np,
        "vencido": vencido,
        "aberto_90": aberto_90,
        "eficiencia": eficiencia,
        "inadimplencia_pct": inad_pct,
        "default_base_pct": base_default,
        "grave_90_pct": grave_90_pct,
        "tendencia": tendencia,
        "veredito": veredito,
        "valor_inicial_sugerido": valor_inicial,
    }

# =============================================================================
# RELATORIOS
# =============================================================================

def fmt_brl_rep(v):
    try:
        return "R$ {:,.0f}".format(float(v))
    except Exception:
        return "R$ 0"

def _rep_period(ctx):
    """Período exibido no relatório."""
    if ctx["apply_period"] and ctx["start_date"] and ctx["end_date"]:
        return f"{ctx['start_date']} a {ctx['end_date']}"
    datas = ctx["movimentos"]["dtvenc"].dropna()
    if len(datas) > 0:
        return f"{datas.min().date()} a {datas.max().date()}"
    return f"{ctx['start_date']} a {ctx['end_date']}"

def gerar_relatorio_visao_geral(ctx):
    c = ctx
    hhi = c["concentracao"]["hhi"]
    hhi_label = "Alta" if hhi > 2500 else ("Media" if hhi > 1500 else "Baixa")
    pct_receb = (c['recebido'] / c['programado'] * 100) if c['programado'] else 0
    pct_90 = (c['aberto_90'] / c['aberto'] * 100) if c['aberto'] else 0
    return f"""### 1. Visão Geral
**Período:** {_rep_period(c)}

- **Programado no periodo:** {fmt_brl_rep(c['programado'])} ({c['parcelas_vencidas']} parcelas vencíveis)
- **Recebido no periodo:** {fmt_brl_rep(c['recebido'])} ({pct_receb:.1f}% do programado)
- **Em aberto no periodo:** {fmt_brl_rep(c['aberto'])} ({c['parcelas_abertas']} parcelas)
- **Vencido (backlog):** {fmt_brl_rep(c['vencido'])} | **A vencer:** {fmt_brl_rep(c['a_vencer'])}
- **Inadimplencia 90+:** {fmt_brl_rep(c['aberto_90'])} ({pct_90:.1f}% do aberto)
- **PDD (provisao):** {fmt_brl_rep(c['pdd'])} | **FPD30:** {c['fpd']['fpd_30']:.1%}
- **HHI (concentracao):** {hhi:.0f} ({hhi_label})
- **Top 10 clientes:** {c['concentracao']['top10_share']:.1%} | **Top 20:** {c['concentracao']['top20_share']:.1%}
- **Parcelas:** {c['parcelas_vencidas']} vencidas, {c['parcelas_pagas']} pagas, {c['parcelas_abertas']} em aberto

**Portfólio completo (sem filtro de periodo):**
- **Total programado:** {fmt_brl_rep(c['programado_total'])} ({len(c['contratos_total'])} contratos)
- **Total recebido:** {fmt_brl_rep(c['recebido_total'])} | **Total em aberto:** {fmt_brl_rep(c['aberto_total'])} | **PDD total:** {fmt_brl_rep(c['pdd_total'])}"""

def gerar_relatorio_fluxo_caixa(ctx):
    c = ctx
    best_dow = c["best_dow"] if c["best_dow"] else "n/d"
    worst_dow = c["worst_dow"] if c["worst_dow"] else "n/d"
    rec_90 = "-"
    if not c["recovery"].empty and "ate 90d" in c["recovery"]["janela"].values:
        rec_90 = "{:.1%}".format(c["recovery"].loc[c["recovery"]["janela"] == "ate 90d", "pct_acumulado"].values[0])
    return f"""### 2. Fluxo de Caixa
**Período:** {_rep_period(c)}

- **Eficiencia historica:** {c['eficiencia']:.1%}
- **Eficiencia recente (90d):** {c['eficiencia_recente']:.1%}
- **Recebimento por dia da semana:** melhor dia {fmt_brl_rep(c['dow']['valor'].max()) if not c['dow'].empty else '-'} ({best_dow}) | pior dia ({worst_dow})
- **Curva de cura - recuperado ate 90 dias:** {rec_90}
- **Projecao futura:** {len(c['cf'][c['cf']['mes'] > pd.Period(pd.Timestamp(c['today']), freq='M')])} meses projetados no cronograma"""

def gerar_relatorio_risco(ctx):
    c = ctx
    rec_90 = "-"
    if not c["recovery"].empty and "ate 90d" in c["recovery"]["janela"].values:
        rec_90 = "{:.1%}".format(c["recovery"].loc[c["recovery"]["janela"] == "ate 90d", "pct_acumulado"].values[0])
    n_prio = len(c["prio"])
    return f"""### 3. Risco & Cobrança
**Período:** {_rep_period(c)}

- **Total vencido:** {fmt_brl_rep(c['vencido'])}
- **PDD (provisao):** {fmt_brl_rep(c['pdd'])}
- **Carteira liquida (aberto - PDD):** {fmt_brl_rep(c['aberto'] - c['pdd'])}
- **Saldo critico 90+:** {fmt_brl_rep(c['aberto_90'])}
- **FPD30:** {c['fpd']['fpd_30']:.1%} | **FPD90:** {c['fpd']['fpd_90']:.1%}
- **HHI:** {c['concentracao']['hhi']:.0f}
- **Recovery Rate 90d:** {rec_90}
- **Clientes prioritarios para cobranca:** {n_prio}"""

def gerar_relatorio_agentes(ctx):
    c = ctx
    if c["agentes"].empty:
        return "### 4. Agentes\nSem dados de agentes no filtro atual."
    total_agentes = len(c["agentes"])
    total_recebido = c["agentes"]["Recebido (R$)"].sum()
    total_aberto = c["agentes"]["Em aberto (R$)"].sum()
    return f"""### 4. Agentes
**Período:** {_rep_period(c)}

- **Numero de agentes:** {total_agentes}
- **Total recebido:** {fmt_brl_rep(total_recebido)} | **Total em aberto:** {fmt_brl_rep(total_aberto)}
- **Melhor agente (eficiencia):** {c['best_agente'] if c['best_agente'] else '-'}
- **Pior agente (eficiencia):** {c['pior_agente'] if c['pior_agente'] else '-'}"""

def gerar_relatorio_carteira(ctx):
    c = ctx
    if c["new_ct"] is None or c["new_ct"].empty:
        novos = 0
        ticket = 0
    else:
        novos = len(c["new_ct"])
        ticket = c["new_ct"]["valor"].mean()
    n_seg = len(c["seg"]) if c["seg"] is not None else 0
    return f"""### 5. Carteira
**Período:** {_rep_period(c)}

- **Contratos no periodo:** {novos}
- **Ticket medio:** {fmt_brl_rep(ticket)}
- **Segmentos (estabelecimento) analisados:** {n_seg}"""

def gerar_relatorio_controle(ctx):
    c = ctx
    n_excl = len(c["contratos_excluidos"])
    val_excl = c["contratos_excluidos"]["valor"].sum() if n_excl > 0 else 0
    return f"""### 6. Controle
**Período:** {_rep_period(c)}

- **Total contratos (validos):** {len(c['contratos_total'])}
- **Principal total (portfolio):** {fmt_brl_rep(c['principal_total'])}
- **Principal no periodo:** {fmt_brl_rep(c['principal'])}
- **Contratos excluidos:** {n_excl} (valor {fmt_brl_rep(val_excl)})"""

def gerar_relatorio_rentabilidade(ctx):
    c = ctx
    retorno = (c['recebido'] / c['principal'] * 100) if c['principal'] else 0
    ticket = c["contratos_total"]["valor"].mean() if len(c["contratos_total"]) else 0
    return f"""### 7. Rentabilidade
**Período:** {_rep_period(c)}

- **Principal:** {fmt_brl_rep(c['principal'])}
- **Juros previstos:** {fmt_brl_rep(c['juros_previstos'])}
- **Juros realizados (caixa):** {fmt_brl_rep(c['juros_realizados'])}
- **Descontos concedidos:** {fmt_brl_rep(c['desconto_total'])}
- **Retorno realizado:** {retorno:.1f}%
- **Ticket medio:** {fmt_brl_rep(ticket)}"""

def gerar_relatorio_viabilidade(ctx):
    c = ctx
    v = c["viab"]
    if v.get("lucro_liquido_ajustado", 0) > 0 or v.get("lucro_bruto_real", 0) > 0:
        status = "VIÁVEL e SAUDÁVEL" if v.get("cobertura_risco", 0) >= 2 else ("VIÁVEL, porém apertado" if v.get("cobertura_risco", 0) >= 1 else "LUCRO CONTÁBIL, insuficiente para o risco")
    else:
        status = "EM PREJUÍZO CONTÁBIL"
    return f"""### 8. Viabilidade & Lucro
**Período:** {_rep_period(c)}

- **Total investido (principal):** {fmt_brl_rep(v.get('total_investido', 0))}
- **Total recebido (caixa):** {fmt_brl_rep(v.get('total_recebido', 0))}
- **Principal recuperado:** {fmt_brl_rep(v.get('principal_recuperado', 0))}
- **Juros recebidos:** {fmt_brl_rep(v.get('juros_recebidos', 0))}
- **Descontos concedidos:** {fmt_brl_rep(v.get('descontos', 0))}
- **Lucro bruto real:** {fmt_brl_rep(v.get('lucro_bruto_real', 0))}
- **Margem de lucro:** {v.get('margem_lucro_pct', 0):.1f}%
- **ROI bruto realizado:** {v.get('roi_bruto_pct', 0):.1f}%
- **Cobertura do risco:** {v.get('cobertura_risco', 0):.1f}x
- **Lucro liquido ajustado ao risco:** {fmt_brl_rep(v.get('lucro_liquido_ajustado', 0))}
- **PDD / Total investido:** {v.get('pdd_total', 0)/v.get('total_investido', 0)*100 if v.get('total_investido', 0) else 0:.1f}%
- **Status da operacao:** {status}"""

def gerar_conclusao_geral(ctx):
    """Resumo executivo consolidado."""
    c = ctx
    v = c["viab"]
    hhi_int = int(round(float(c["concentracao"]["hhi"])))
    hhi_label = "ALTA" if hhi_int > 2500 else ("MEDIA" if hhi_int > 1500 else "BAIXA")
    pct_90 = (c["aberto_90"] / c["aberto"] * 100) if c["aberto"] else 0
    pct_vencido = (c["vencido"] / c["aberto"] * 100) if c["aberto"] else 0
    pdd_pct_aberto = (c["pdd"] / c["aberto"] * 100) if c["aberto"] else 0
    cobertura = v.get("cobertura_risco", 0)
    lucro = v.get("lucro_bruto_real", 0)
    lucro_liq = v.get("lucro_liquido_ajustado", 0)
    roi_aj = (lucro_liq / v.get("total_investido", 0) * 100) if v.get("total_investido", 0) else 0

    if lucro > 0 and cobertura >= 2.0:
        saude_titulo = "A operação é VIÁVEL e SAUDÁVEL."
        saude_txt = (f"A rentabilidade cobre com folga o risco: o lucro bruto de {fmt_brl_rep(lucro)} "
                     f"representa {cobertura:.1f}x a PDD ({fmt_brl_rep(v.get('pdd_total', 0))}), gerando lucro líquido "
                     f"ajustado ao risco positivo de {fmt_brl_rep(lucro_liq)}.")
    elif lucro > 0 and cobertura >= 1.0:
        saude_titulo = "A operação é VIÁVEL, MAS COM MARGEM APERTADA."
        saude_txt = (f"A rentabilidade nominal é alta, mas o saldo em atraso consome boa parte do lucro: o bruto de "
                     f"{fmt_brl_rep(lucro)} cobre apenas {cobertura:.1f}x a PDD ({fmt_brl_rep(v.get('pdd_total', 0))}), "
                     f"reduzindo o lucro líquido ajustado ao risco para {fmt_brl_rep(lucro_liq)}.")
    elif lucro > 0:
        saude_titulo = "A operação tem LUCRO CONTÁBIL, mas INSUFICIENTE para o risco."
        saude_txt = (f"O lucro bruto ({fmt_brl_rep(lucro)}) é positivo, porém menor que a PDD "
                     f"({fmt_brl_rep(v.get('pdd_total', 0))}), deixando o lucro líquido ajustado negativo "
                     f"({fmt_brl_rep(lucro_liq)}).")
    else:
        saude_titulo = "A operação está EM PREJUÍZO CONTÁBIL."
        saude_txt = (f"O lucro bruto já é negativo ({fmt_brl_rep(lucro)}). Ação imediata de contenção de risco "
                     f"e cobrança é necessária.")

    backlog_txt = "-"
    if not c["backlog_df"].empty:
        bf = c["backlog_df"]
        partes_backlog = [f"{r['faixa']} {fmt_brl_rep(r['valor'])}" for _, r in bf.iterrows()]
        backlog_txt = " | ".join(partes_backlog)

    maior_pdd_faixa = "-"
    if not c["pdd_df"].empty:
        pddf = c["pdd_df"].sort_values("PDD", ascending=False)
        top = pddf.iloc[0]
        maior_pdd_faixa = f"{top['Faixa']} ({fmt_brl_rep(top['PDD'])})"

    rec_90 = "-"
    if not c["recovery"].empty and "ate 90d" in c["recovery"]["janela"].values:
        rec_90 = "{:.1%}".format(c["recovery"].loc[c["recovery"]["janela"] == "ate 90d", "pct_acumulado"].values[0])

    roll_90_txt = "-"
    if not c["roll_rate"].empty:
        rr = c["roll_rate"]
        prox = rr[rr["faixa"] == "90+d"]
        if not prox.empty:
            stay = prox[prox["faixa_next"] == "90+d"]
            if not stay.empty:
                roll_90_txt = "{:.1%}".format(stay["pct"].values[0])

    proj_30_txt = "-"
    try:
        fut30 = pd.Timestamp(c["today"]) + pd.Timedelta(days=30)
        sel = c["movimentos"][c["movimentos"]["a_vencer"] & (c["movimentos"]["dtvenc"] <= fut30)]
        proj_30_txt = fmt_brl_rep(sel["areceber"].sum())
    except Exception:
        proj_30_txt = "-"

    pct_receb = (c["recebido"] / c["programado"] * 100) if c["programado"] else 0
    if c["eficiencia_recente"] >= 0.80:
        cob_qual = "Boa"
    elif c["eficiencia_recente"] >= 0.60:
        cob_qual = "Moderada"
    else:
        cob_qual = "Fraca"

    if hhi_int > 2500 or c["concentracao"]["top10_share"] > 0.50:
        conc_qual = "ELEVADA"
    elif hhi_int > 1500 or c["concentracao"]["top10_share"] > 0.35:
        conc_qual = "MÉDIA"
    else:
        conc_qual = "BAIXA"

    agt_txt = "-"
    if not c["agentes"].empty:
        ag = c["agentes"].sort_values("Eficiencia %", ascending=False)
        pior_ag = ag.iloc[-1]
        maior_aberto = ag.sort_values("Em aberto (R$)", ascending=False).iloc[0]
        ag_top_txt = ", ".join(f"{r['Agente']} ({r['Eficiencia %']:.1%})" for _, r in ag.head(2).iterrows())
        agt_txt = (f"Maiores eficiências: {ag_top_txt}. Menor: {pior_ag['Agente']} ({pior_ag['Eficiencia %']:.1%}). "
                   f"Maior saldo aberto: {maior_aberto['Agente']} ({fmt_brl_rep(maior_aberto['Em aberto (R$)'])} "
                   f"com eficiência de {maior_aberto['Eficiencia %']:.1%}).")

    seg_top_txt = "-"
    seg_crit_txt = "-"
    if c["seg"] is not None and not c["seg"].empty and c["seg"]["Eficiencia %"].notna().any():
        seg_ok = c["seg"][c["seg"]["Eficiencia %"].notna()].copy()
        seg_ok = seg_ok[seg_ok["Recebido"] + seg_ok["Em_aberto"] > 0]
        if not seg_ok.empty:
            top_seg = seg_ok.sort_values("Eficiencia %", ascending=False).head(2)
            crit_seg = seg_ok.sort_values("Eficiencia %", ascending=True).head(2)
            seg_top_txt = ", ".join(f"{r['Segmento']} ({r['Eficiencia %']:.1%})" for _, r in top_seg.iterrows())
            seg_crit_txt = ", ".join(f"{r['Segmento']} ({r['Eficiencia %']:.1%})" for _, r in crit_seg.iterrows())

    ticket = c["contratos_total"]["valor"].mean() if len(c["contratos_total"]) else 0
    kpi_rows = [
        ("Lucro Bruto Real", fmt_brl_rep(lucro), "Lucro acumulado sem deduzir provisão"),
        ("Lucro Líquido Ajustado ao Risco", fmt_brl_rep(lucro_liq),
         f"Lucro após deduzir a PDD ({fmt_brl_rep(v.get('pdd_total', 0))})"),
        ("Cobertura do Risco", f"{cobertura:.1f}x",
         "Margem de cobertura para perdas (ideal ≥ 2x)"),
        ("Inadimplência Crítica (90+ dias)", fmt_brl_rep(c["aberto_90"]),
         f"{pct_90:.1f}% do saldo em aberto"),
        ("ROI Ajustado ao Risco", f"{roi_aj:.1f}%",
         "Retorno líquido real em relação ao investimento"),
        ("Eficiência de Cobrança (90d)", f"{c['eficiencia_recente']:.1%}",
         f"Qualidade {cob_qual}; histórica {c['eficiencia']:.1%}"),
    ]
    kpi_tabela = "| Indicador (KPI) | Valor Atual | Status / Impacto |\n|---|---|---|\n"
    kpi_tabela += "\n".join(f"| {a} | {b} | {d} |" for a, b, d in kpi_rows)

    pontos_fortes = [
        f"Modelo gera lucro bruto significativo ({fmt_brl_rep(lucro)}) e cobre {cobertura:.1f}x o risco de perda estimado.",
        f"Eficiência de cobrança recente é {cob_qual.lower()} ({c['eficiencia_recente']:.1%}) e a inadimplência inicial "
        f"(FPD30) está em {c['fpd']['fpd_30']:.1%} — controlada na origem.",
        f"Concentração de risco {conc_qual.lower()} (HHI = {hhi_int}; Top 10 = {c['concentracao']['top10_share']:.1%}).",
    ]
    riscos_imediatos = [
        f"Estoque vencido 90+ de {fmt_brl_rep(c['aberto_90'])} ({pct_90:.1f}% do aberto) é o maior fator de risco, "
        f"consumindo a maior parte da provisão.",
        f"PDD de {fmt_brl_rep(c['pdd'])} equivale a {pdd_pct_aberto:.1f}% do saldo em aberto; maior provisão em {maior_pdd_faixa}.",
    ]
    if c["roll_rate"].empty:
        riscos_imediatos.append("Sem histórico suficiente de Roll Rate para avaliar a persistência das faixas de atraso.")
    else:
        riscos_imediatos.append(f"Roll Rate indica que {roll_90_txt} do saldo 90+d permanece na faixa crítica, "
                                f"confirmando dificuldade de recuperação.")
    if not c["agentes"].empty and (c["agentes"]["Eficiencia %"] < 0.20).any():
        piores = c["agentes"][c["agentes"]["Eficiencia %"] < 0.20]["Agente"].tolist()
        riscos_imediatos.append(f"Forte desigualdade de performance entre agentes ({', '.join(piores)} com eficiência < 20%), "
                                f"comprometendo a saúde da carteira.")
    if lucro_liq < 0:
        riscos_imediatos.append(f"A margem líquida ajustada ao risco é negativa ({fmt_brl_rep(lucro_liq)}), tornando a operação "
                                f"sensível a qualquer aumento de inadimplência ou desconto.")

    rec_lista = []
    rec_lista.append(f"**Prioridade Crítica:** acionar cobrança externa e negociar com desconto progressivo para recuperar "
                     f"parte dos {fmt_brl_rep(c['aberto_90'])} em atraso 90+.")
    rec_lista.append("**Reforçar cobrança preventiva** nos atrasos 61-90 dias para evitar migração para a faixa crítica.")
    if not c["agentes"].empty and (c["agentes"]["Eficiencia %"] < 0.20).any():
        baixos = c["agentes"][c["agentes"]["Eficiencia %"] < 0.20]["Agente"].tolist()
        rec_lista.append(f"**Revisar performance dos agentes** ({', '.join(baixos)}): investigar causas, treinar ou realocar carteira.")
    elif c["best_agente"]:
        rec_lista.append(f"**Espalhar boas práticas** do melhor agente ({c['best_agente']}) para elevar a eficiência dos demais.")
    if seg_crit_txt != "-":
        rec_lista.append(f"**Focar em segmentos rentáveis** (ex.: {seg_top_txt}) e revisar política de crédito para segmentos críticos ({seg_crit_txt}).")
    if cobertura < 2.0:
        rec_lista.append("**Ampliar o colchão de margem:** revisar pricing/taxas e política de descontos até a cobertura do risco chegar a ≥ 2x.")

    pag_por_idade = "Sem dados suficientes."
    try:
        prof_idade = build_payment_profile(c["movimentos"], ["faixa_idade"])
        melhor, pior, igual, ml, pl = comparar_pagadores(prof_idade, ["faixa_idade"])
        if melhor is not None:
            if igual:
                pag_por_idade = f"pagadores **semelhantes** (~{melhor['Eficiencia %']:.1%} de eficiência)."
            else:
                pag_por_idade = (f"melhor pagador: **{ml}** ({melhor['Eficiencia %']:.1%} efic., "
                                 f"{melhor['% Parcelas pagas']:.1%} parcelas pagas); mais inadimplente: **{pl}** "
                                 f"({pior['Eficiencia %']:.1%} efic., {pior['% Aberto 90+']:.1%} do aberto em 90+).")
    except Exception:
        pass

    pag_por_sexo_idade = "Sem dados suficientes."
    try:
        prof_gen = build_payment_profile(c["movimentos"], ["genero_cat", "faixa_idade"])
        melhor, pior, igual, ml, pl = comparar_pagadores(prof_gen, ["genero_cat", "faixa_idade"])
        if melhor is not None:
            if igual:
                pag_por_sexo_idade = f"pagadores **semelhantes** (~{melhor['Eficiencia %']:.1%} de eficiência)."
            else:
                pag_por_sexo_idade = (f"melhor pagador: **{ml}** ({melhor['Eficiencia %']:.1%} efic., "
                                      f"{melhor['% Parcelas pagas']:.1%} parcelas pagas); mais inadimplente: **{pl}** "
                                      f"({pior['Eficiencia %']:.1%} efic., {pior['% Aberto 90+']:.1%} do aberto em 90+).")
    except Exception:
        pass

    pag_por_segmento = "Sem dados suficientes."
    try:
        prof_seg = build_payment_profile(c["movimentos"], ["nome_estabelecimento_norm"])
        if prof_seg is not None and not prof_seg.empty:
            prof_seg = prof_seg.rename(columns={"nome_estabelecimento_norm": "Segmento"})
            melhor, pior, igual, ml, pl = comparar_pagadores(prof_seg, ["Segmento"])
            if melhor is not None:
                if igual:
                    pag_por_segmento = f"pagadores **semelhantes** (~{melhor['Eficiencia %']:.1%} de eficiência)."
                else:
                    pag_por_segmento = (f"melhor pagador: **{ml}** ({melhor['Eficiencia %']:.1%} efic., "
                                        f"{melhor['% Parcelas pagas']:.1%} parcelas pagas); mais inadimplente: **{pl}** "
                                        f"({pior['Eficiencia %']:.1%} efic., {pior['% Aberto 90+']:.1%} do aberto em 90+).")
    except Exception:
        pass

    def _resumo_coorte(by, label_cols):
        try:
            co = build_coorte_recebimento(c["movimentos"], by)
            mais_rec = top_coorte(co, "Recebido_geral", label_cols)
            mais_inad = top_coorte(co, "Inadimplencia", label_cols)
            maior_risco = top_coorte(co, "% Inadimplencia", label_cols)
            if co is None or co.empty:
                return "Sem dados suficientes."
            parte_rec = " • ".join(f"{l} (R${v:,.0f})" for l, v in mais_rec) if mais_rec else "-"
            parte_inad = " • ".join(f"{l} (R${v:,.0f})" for l, v in mais_inad) if mais_inad else "-"
            parte_risco = " • ".join(f"{l} ({v:.1%})" for l, v in maior_risco) if maior_risco else "-"
            return (f"+ recebimento: **{parte_rec}**; mais inadimplência (valor): **{parte_inad}**; "
                    f"maior risco (%): **{parte_risco}**.")
        except Exception:
            return "Erro ao calcular."

    coorte_idade = _resumo_coorte(["faixa_idade"], ["faixa_idade"])
    coorte_sexo_idade = _resumo_coorte(["genero_cat", "faixa_idade"], ["genero_cat", "faixa_idade"])

    vf = build_perfil_valor_contrato(c["contratos"])
    vf_txt = "Sem dados suficientes."
    if vf is not None and not vf.empty:
        faixa_mais_rec = vf.loc[vf["Recebido"].idxmax()]
        faixa_mais_inad = vf.loc[vf["Vencido"].idxmax()]
        faixa_mais_risco = vf.loc[vf["Default_%"].idxmax()]
        vf_txt = (f"+ recebimento: **{faixa_mais_rec['Faixa Valor']}** (R${faixa_mais_rec['Recebido']:,.0f}); "
                  f"mais inadimplência: **{faixa_mais_inad['Faixa Valor']}** (R${faixa_mais_inad['Vencido']:,.0f}); "
                  f"maior default 90d: **{faixa_mais_risco['Faixa Valor']}** ({faixa_mais_risco['Default_%']:.1%}).")

    estrategia_txt = (
        "Valor inicial por perfil (% do teto/capacidade): **Novo 30%**, **Risco 25%**, "
        "**Recorrente 55%**, **Confiável 75%**. Escalonamento: +25% do limite a cada contrato "
        "pago em dia (Risco: +20%), até um teto fixo de 75% do valor solicitado."
    )

    viab_perfil_txt = "Sem dados suficientes."
    try:
        perfil_alto = analisar_viabilidade_perfil(
            c["movimentos"], c["contratos"], genero=None, idade=None, segmento=None, valor=None)
        if perfil_alto["amostra_clientes"] > 0:
            viab_perfil_txt = (
                f"A carteira registra eficiência de pagamento de "
                f"{perfil_alto['eficiencia']:.1%} e inadimplência de {perfil_alto['inadimplencia_pct']:.1%} "
                f"do movimentado. A avaliação por perfil combina essa eficiência com a faixa de valor "
                f"solicitada: perfis com tendência **Alta/Média** (inadimplência elevada ou baixa eficiência) "
                f"devem receber valor inicial reduzido (~25–40%) e escalonar conforme pagamentos; perfis de "
                f"tendência **Baixa** podem receber próximo do valor solicitado (~60%)."
            )
    except Exception:
        viab_perfil_txt = "Não foi possível calcular no momento."

    return f"""## Resumo Executivo

**{saude_titulo}** {saude_txt}

**Volume de Operações:** {fmt_brl_rep(c['principal_total'])} em {len(c['contratos_total'])} contratos, com ticket médio de {fmt_brl_rep(ticket)}.

**Arrecadação e Retorno:** Total de {fmt_brl_rep(c['recebido_total'])} recebidos em caixa ({pct_receb:.1f}% do programado), gerando um Lucro Bruto Real de {fmt_brl_rep(lucro)} e ROI Bruto de {v.get('roi_bruto_pct', 0):.1f}%.

**Perfil de Inadimplência:** Saldo em aberto de {fmt_brl_rep(c['aberto'])}, com {fmt_brl_rep(c['vencido'])} vencido (backlog, {pct_vencido:.1f}% do aberto) e {fmt_brl_rep(c['aberto_90'])} concentrados no prazo crítico de 90+ dias ({pct_90:.1f}% do saldo aberto).

**Provisão de Risco (PDD):** Provisão acumulada em {fmt_brl_rep(c['pdd'])} ({pdd_pct_aberto:.1f}% do saldo aberto).

**Concentração e Performance:** O índice HHI indica concentração **{hhi_label}** ({hhi_int}), com Top 10 de clientes representando {c['concentracao']['top10_share']:.1%} do total em aberto (Top 20 = {c['concentracao']['top20_share']:.1%}). {agt_txt}

{kpi_tabela}

## Análise Detalhada

### 1. Desempenho Geral da Carteira
- **Portfólio:** {len(c['contratos_total'])} contratos, principal total de {fmt_brl_rep(c['principal_total'])}.
- **Recebimento Total:** {fmt_brl_rep(c['recebido_total'])}, representando {pct_receb:.1f}% do programado ({fmt_brl_rep(c['programado_total'])}).
- **Saldo em Aberto:** {fmt_brl_rep(c['aberto'])}, com destaque para **vencido (backlog)** de {fmt_brl_rep(c['vencido'])} ({pct_vencido:.1f}% do aberto).
- **Inadimplência Crítica:** {fmt_brl_rep(c['aberto_90'])} em atraso superior a 90 dias ({pct_90:.1f}% do aberto).

### 2. Eficiência de Cobrança e Fluxo de Caixa
- **Eficiência histórica:** {c['eficiencia']:.1%}; nos últimos 90 dias: {c['eficiencia_recente']:.1%}.
- **Projeção 30 dias:** recebimentos previstos de {proj_30_txt} nas próximas 30 dias.
- **Backlog por faixa de atraso:** {backlog_txt}.
- **Curva de cura:** recuperação acumulada em até 90 dias de {rec_90}.

### 3. Risco e Provisão (PDD)
- **PDD (Provisão para Devedores Duvidosos):** {fmt_brl_rep(c['pdd'])}, representando {pdd_pct_aberto:.1f}% do saldo em aberto.
- **Carteira líquida (aberto - PDD):** {fmt_brl_rep(c['aberto'] - c['pdd'])}.
- **FPD30:** {c['fpd']['fpd_30']:.1%} — inadimplência inicial {('controlada' if c['fpd']['fpd_30'] < 0.10 else 'elevada')}, mas o estoque antigo é alto.
- **Roll Rate:** a persistência do saldo 90+d na faixa crítica é de {roll_90_txt}.

### 4. Concentração e Performance por Agente
- **HHI (Índice de Herfindahl-Hirschman):** {hhi_int} — concentração **{conc_qual}** de risco.
- **Top 10 clientes:** {c['concentracao']['top10_share']:.1%} do saldo em aberto; **Top 20:** {c['concentracao']['top20_share']:.1%}.
- **Performance por agente:** {agt_txt}.

### 5. Segmentação e Rentabilidade
- **Segmentos com melhor eficiência:** {seg_top_txt}.
- **Segmentos críticos:** {seg_crit_txt}.
- **Lucro Bruto Realizado:** {fmt_brl_rep(lucro)}, com margem de {v.get('margem_lucro_pct', 0):.1f}% sobre o principal recuperado.
- **Ajustado pelo risco (PDD):** lucro líquido de {fmt_brl_rep(lucro_liq)}, com ROI ajustado ao risco de {roi_aj:.1f}%.

### 6. Perfil de Comportamento de Pagamento
- **Por faixa etária:** {pag_por_idade}
- **Por sexo + faixa etária:** {pag_por_sexo_idade}
- **Por segmento:** {pag_por_segmento}

### 7. Inadimplência vs Recebimento por Faixa Etária
- **Por faixa etária:** {coorte_idade}
- **Por sexo + faixa etária:** {coorte_sexo_idade}

### 8. Perfil por Faixa de Valor do Contrato
{vf_txt}

### 9. Estratégia de Valor Inicial por Perfil
{estrategia_txt}

### 10. Viabilidade por Perfil (Novo Contrato)
{viab_perfil_txt}

## Conclusão

{saude_titulo} {saude_txt}

## Pontos Fortes
{chr(10).join('• ' + p for p in pontos_fortes)}

## Riscos Imediatos
{chr(10).join('• ' + p for p in riscos_imediatos)}

## Recomendações Estratégicas
{chr(10).join('• ' + p for p in rec_lista)}"""

def gerar_relatorio_geral(ctx, paginas_selecionadas):
    geradores = {
        "Visao Geral": gerar_relatorio_visao_geral,
        "Fluxo de Caixa": gerar_relatorio_fluxo_caixa,
        "Risco & Cobranca": gerar_relatorio_risco,
        "Agentes": gerar_relatorio_agentes,
        "Carteira": gerar_relatorio_carteira,
        "Rentabilidade": gerar_relatorio_rentabilidade,
        "Viabilidade & Lucro": gerar_relatorio_viabilidade,
    }
    try:
        data_gerado = ctx["hoje"].strftime("%d/%m/%Y")
    except Exception:
        data_gerado = str(ctx["today"])
    partes = [f"# RELATÓRIO DA ANÁLISE FINANCEIRA\nGerado em {data_gerado} | Período: {_rep_period(ctx)}"]
    if paginas_selecionadas:
        fatias = [g for k, g in geradores.items() if k in paginas_selecionadas]
        incluir_conclusao = False
    else:
        fatias = list(geradores.values())
        incluir_conclusao = True
    for g in fatias:
        try:
            partes.append(g(ctx))
        except Exception as e:
            partes.append(f"(erro ao gerar seção {getattr(g, '__name__', '')}: {e})")
    if incluir_conclusao:
        try:
            partes.append(gerar_conclusao_geral(ctx))
        except Exception as e:
            partes.append(f"(erro ao gerar conclusão: {e})")
    return "\n\n".join(partes)

# =============================================================================
# MAIN
# =============================================================================
def main():
    st.set_page_config(page_title="Painel Financeiro - Microcredito", layout="wide", initial_sidebar_state="expanded")

    st.sidebar.header("🎬 Modo de Demonstração")
    
    use_fake_data = st.sidebar.checkbox(
        "Usar dados fictícios (Faker)", 
        value=False,
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
    period_active = bool(apply_period and start_date and end_date)

    if period_active:
        contratos, movimentos = apply_period_filter(contratos, movimentos, start_date, end_date)

    ativos_contratos = contratos.copy()
    ativos_movimentos = movimentos.copy()

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

    dataset_base_contratos = contratos if period_active else contratos_total
    dataset_base_movimentos = movimentos if period_active else movimentos_total

    lgd_rates = compute_lgd_observada(dataset_base_movimentos)
    backlog_df_total = build_backlog(dataset_base_movimentos, hoje_ts)
    pdd_df_total = build_pdd(backlog_df_total, lgd_rates)
    pdd_total = pdd_df_total["PDD"].sum() if not pdd_df_total.empty else 0.0

    backlog_df = build_backlog(dataset_base_movimentos, hoje_ts)
    pdd_df = build_pdd(backlog_df, lgd_rates)
    pdd = pdd_df["PDD"].sum() if not pdd_df.empty else 0.0

    period_start_ts = pd.Timestamp(start_date) if period_active and start_date else None
    period_end_exclusive = pd.Timestamp(end_date) + pd.Timedelta(days=1) if period_active and end_date else None
    period_end_ts = pd.Timestamp(end_date) if period_active and end_date else None

    cf, eficiencia, eficiencia_recente = build_cashflow(dataset_base_movimentos, dataset_base_contratos, hoje_ts,
                                                     period_start=period_start_ts, period_end=period_end_exclusive)
    recovery = build_recovery_curve(dataset_base_movimentos)
    dow = build_dow_analysis(dataset_base_movimentos)
    agentes = build_agent_performance(dataset_base_contratos, dataset_base_movimentos, usuarios)
    monthly_return = build_monthly_return(dataset_base_movimentos, period_start=period_start_ts, period_end=period_end_exclusive)
    monthly_eff = build_monthly_efficiency(dataset_base_movimentos, period_start=period_start_ts, period_end=period_end_exclusive)
    fpd = build_fpd(dataset_base_movimentos, dataset_base_contratos)
    roll_rate = build_roll_rate(dataset_base_movimentos)
    concentracao = build_concentration(dataset_base_movimentos)

    aberto_80_89 = dataset_base_movimentos.loc[dataset_base_movimentos["vencido"] & dataset_base_movimentos["dias_atraso"].between(80, 89), "areceber"].sum()
    futuro_30d = hoje_ts + pd.Timedelta(days=30)
    open_next_30 = dataset_base_movimentos.loc[dataset_base_movimentos["a_vencer"] & (dataset_base_movimentos["dtvenc"] <= futuro_30d), "areceber"].sum()

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

    # =========================================================================
    # CORREÇÃO: Chamar as funções com os períodos corretos
    # =========================================================================
    viab = build_viability_analysis(
        dataset_base_contratos, 
        dataset_base_movimentos, 
        pdd_total,
        period_start=period_start_ts, 
        period_end=period_end_exclusive
    )
    
    monthly_profit = build_monthly_profit(
        dataset_base_movimentos, 
        period_start=period_start_ts, 
        period_end=period_end_exclusive
    )

    # ---- Dados auxiliares para o relatório ---------------------------------
    new_ct = contratos[contratos["dtinicio"].notna()].copy()
    if period_active:
        new_ct = new_ct[(new_ct["dtinicio"] >= period_start_ts) & (new_ct["dtinicio"] < period_end_exclusive)].copy()

    seg = movimentos.groupby("nome_estabelecimento_norm").agg(
        Recebido=("valorrecebido", "sum"),
        Em_aberto=("areceber", lambda s: s[movimentos.loc[s.index, "status_pago"] == False].sum()),
        Clientes=("idcliente", "nunique"),
    ).reset_index().rename(columns={"nome_estabelecimento_norm": "Segmento"})
    seg["Total"] = seg["Recebido"] + seg["Em_aberto"]
    seg["Eficiencia %"] = seg["Recebido"] / seg["Total"].replace(0, np.nan)
    seg = seg.sort_values("Total", ascending=False)
    prio = build_priority_clients(movimentos)

    ctx = {
        "today": today,
        "hoje": hoje_ts,
        "apply_period": apply_period,
        "start_date": start_date,
        "end_date": end_date,
        "movimentos": movimentos,
        "movimentos_total": movimentos_total,
        "contratos": contratos,
        "contratos_total": contratos_total,
        "contratos_excluidos": contratos_excluidos,
        "programado": programado, "recebido": recebido, "aberto": aberto,
        "vencido": vencido, "a_vencer": a_vencer, "aberto_90": aberto_90,
        "aberto_80_89": aberto_80_89, "open_next_30": open_next_30,
        "parcelas_vencidas": parcelas_vencidas, "parcelas_pagas": parcelas_pagas,
        "parcelas_abertas": parcelas_abertas,
        "programado_total": programado_total, "recebido_total": recebido_total,
        "aberto_total": aberto_total, "principal": principal, "principal_total": principal_total,
        "juros_previstos": juros_previstos, "juros_realizados": juros_realizados,
        "desconto_total": desconto_total, "pdd": pdd, "pdd_total": pdd_total,
        "fpd": fpd, "concentracao": concentracao, "cf": cf,
        "eficiencia": eficiencia, "eficiencia_recente": eficiencia_recente,
        "recovery": recovery, "dow": dow, "agentes": agentes,
        "best_agente": best_agente, "pior_agente": pior_agente,
        "best_dow": best_dow, "worst_dow": worst_dow,
        "viab": viab, "monthly_profit": monthly_profit,
        "new_ct": new_ct, "seg": seg, "prio": prio,
        "backlog_df": backlog_df, "pdd_df": pdd_df, "roll_rate": roll_rate,
        "monthly_eff": monthly_eff,
    }

    # ---- Seletor e exibicao do relatorio -----------------------------------
    scope = st.sidebar.radio(
        "Escopo do relatório",
        ["Geral (todas as páginas)", "Por página"],
        key="rel_scope",
    )
    if scope.startswith("Por página"):
        paginas_opcoes = ["Visao Geral", "Fluxo de Caixa", "Risco & Cobranca", "Agentes",
                          "Carteira", "Rentabilidade", "Viabilidade & Lucro"]
        paginas_sel = st.sidebar.multiselect("Páginas do relatório", paginas_opcoes, default=paginas_opcoes)
    else:
        paginas_sel = []
    gerar_click = st.sidebar.button("📄 Gerar relatório", use_container_width=True)

    if gerar_click:
        conteudo = gerar_relatorio_geral(ctx, paginas_sel)
        with st.expander(f"📄 Relatorio ({'Geral' if scope.startswith('Geral') else 'Por página'})", expanded=True):
            st.download_button(
                "⬇️ Baixar relatório (.md)",
                data=conteudo.encode("utf-8"),
                file_name=f"relatorio_analise_{date.today().strftime('%Y%m%d')}.md",
                mime="text/markdown",
            )
            st.markdown(conteudo)

    st.title("🎬 Painel Financeiro - Microcredito Diario")
    st.caption("Carteira de microcredito (90 parcelas diarias - Pix). Modo de demonstração com dados fictícios.")
    
    if use_fake_data and FAKER_AVAILABLE:
        st.info("📊 **Modo de Demonstração:** Exibindo dados gerados aleatoriamente com Faker. Os dados são para fins de demonstração apenas.")

    paginas = ["Visao Geral", "Fluxo de Caixa", "Risco & Cobranca", "Agentes", "Carteira",
               "Rentabilidade", "Viabilidade & Lucro", "Simulador Financeiro",
               "Modelos Preditivos",
               "Visualizacoes", "Dados"]
    aba = st.sidebar.radio("📑 Navegação", paginas, index=0, key="nav_aba")

    # =========================================================================
    # TAB 1 - VISAO GERAL
    # =========================================================================
    if aba == "Visao Geral":
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
        st.info(f"**Periodo:** {_rep_period(ctx)} | "
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
    elif aba == "Fluxo de Caixa":
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
    elif aba == "Risco & Cobranca":
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
    elif aba == "Agentes":
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
    elif aba == "Carteira":
        st.subheader("Originacao e maturacao")
        new_ct = contratos[contratos["dtinicio"].notna()].copy()
        period_start_ts = pd.Timestamp(start_date) if start_date else None
        period_end_exclusive = pd.Timestamp(end_date) + pd.Timedelta(days=1) if end_date else None
        if apply_period and period_start_ts and period_end_exclusive:
            new_ct = new_ct[(new_ct["dtinicio"] >= period_start_ts) & (new_ct["dtinicio"] < period_end_exclusive)].copy()

        if not new_ct.empty:
            stats_start = period_start_ts if apply_period and period_start_ts else new_ct["dtinicio"].min()
            stats_end = pd.Timestamp(end_date) if apply_period and end_date else new_ct["dtinicio"].max()
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

            st.markdown("##### Novos contratos por categoria / subcategoria")
            seg_new = []
            if "categoria_estabelecimento" in new_ct.columns:
                seg_new.append(("categoria_estabelecimento", "Categoria"))
            if "subcategoria_estabelecimento" in new_ct.columns:
                seg_new.append(("subcategoria_estabelecimento", "Subcategoria"))
            for col, label in seg_new:
                by_seg = build_new_contract_stats(new_ct, stats_start, stats_end, dimension=col)
                if by_seg.empty:
                    continue
                by_seg = by_seg.rename(columns={"Valor_originado": "Valor originado (R$)", "Ticket_medio": "Ticket medio (R$)"})
                by_seg["Valor originado (R$)"] = by_seg["Valor originado (R$)"].round(2)
                by_seg["Ticket medio (R$)"] = by_seg["Ticket medio (R$)"].round(2)
                st.markdown(f"**{label}**")
                show(by_seg.head(10))

        st.markdown("---")
        st.subheader("Resumo por categoria e subcategoria")
        if "categoria_estabelecimento" in movimentos.columns and "subcategoria_estabelecimento" in movimentos.columns:
            seg_cart = movimentos[["categoria_estabelecimento", "subcategoria_estabelecimento", "valorrecebido", "areceber", "status_pago", "idcliente"]].copy()
            seg_cart["subcategoria_estabelecimento"] = seg_cart["subcategoria_estabelecimento"].fillna("OUTROS")
            seg_cart["categoria_estabelecimento"] = seg_cart["categoria_estabelecimento"].fillna("OUTROS")
            seg_tbl = seg_cart.groupby(["categoria_estabelecimento", "subcategoria_estabelecimento"], dropna=False).agg(
                Recebido=("valorrecebido", "sum"),
                Em_aberto=("areceber", lambda s: s[~seg_cart.loc[s.index, "status_pago"]].sum()),
                Clientes=("idcliente", "nunique"),
            ).reset_index()
            seg_tbl["Em_aberto"] = seg_tbl.apply(lambda r: seg_cart.loc[(seg_cart["categoria_estabelecimento"] == r["categoria_estabelecimento"]) & (seg_cart["subcategoria_estabelecimento"] == r["subcategoria_estabelecimento"]) & (~seg_cart["status_pago"]), "areceber"].sum(), axis=1)
            seg_tbl["Total"] = seg_tbl["Recebido"] + seg_tbl["Em_aberto"]
            seg_tbl["Eficiencia %"] = seg_tbl["Recebido"] / seg_tbl["Total"].replace(0, np.nan)
            seg_tbl = seg_tbl.sort_values("Total", ascending=False).reset_index(drop=True)
            seg_tbl["Eficiencia %"] = (seg_tbl["Eficiencia %"] * 100).round(1)
            seg_tbl[["Recebido", "Em_aberto", "Total"]] = seg_tbl[["Recebido", "Em_aberto", "Total"]].round(2)
            seg_tbl = seg_tbl.rename(columns={"categoria_estabelecimento": "Categoria", "subcategoria_estabelecimento": "Subcategoria"})
            show(seg_tbl[["Categoria", "Subcategoria", "Recebido", "Em_aberto", "Total", "Clientes", "Eficiencia %"]])

            fig_seg = px.bar(
                seg_tbl.sort_values("Total", ascending=False).head(15),
                x="Subcategoria",
                y="Total",
                color="Categoria",
                title="Top subcategorias por valor total",
                text="Total",
            )
            fig_seg.update_traces(texttemplate="R$ %{text:,.0f}", textposition="outside")
            fig_seg.update_layout(height=380, xaxis_tickangle=-25)
            st.plotly_chart(fig_seg, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Dados de categoria/subcategoria não disponíveis para o filtro atual.")

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
        seg_dim = st.selectbox("Agrupar por:", ["subcategoria_estabelecimento", "categoria_estabelecimento", "nome_estabelecimento_norm"], index=0)
        seg = build_segment_summary(movimentos, dimension=seg_dim)
        if seg.empty:
            st.info("Sem dados para o agrupamento selecionado.")
        else:
            view_seg = seg.copy()
            view_seg["Eficiencia %"] = (view_seg["Eficiencia %"] * 100).round(1)
            view_seg[["Recebido", "Em_aberto", "Total"]] = view_seg[["Recebido", "Em_aberto", "Total"]].round(2)
            show(view_seg)

    # =========================================================================
    # TAB 6 - RENTABILIDADE
    # =========================================================================
    elif aba == "Rentabilidade":
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

        st.markdown("---")
        st.subheader("Perfil de comportamento de pagamento")
        st.caption("Quem é o melhor pagador e quem é o mais inadimplente (por idade, sexo+idade e segmento).")

        def render_perfil_pagamento(titulo, profile, label_cols):
            st.markdown(f"##### {titulo}")
            if profile is None or profile.empty:
                st.info("Sem dados para esta dimensão.")
                return
            view = profile.copy()
            for col in ["Eficiencia %", "% Parcelas pagas", "% Aberto 90+"]:
                view[col] = (view[col] * 100).round(1)
            for col in ["Recebido", "Em_aberto", "Total", "Aberto_90"]:
                view[col] = view[col].round(2)
            show(view)
            if len(profile) > 1:
                chart_df = profile.copy()
                chart_df["Eficiencia %"] = (chart_df["Eficiencia %"] * 100).round(1)
                xcol = label_cols[-1] if label_cols else profile.columns[0]
                label_lbl = " + ".join(label_cols) if label_cols else "Grupo"
                fig_fp = px.bar(chart_df, x=chart_df[xcol].astype(str), y="Eficiencia %",
                                color="Eficiencia %", color_continuous_scale="RdYlGn",
                                text="Eficiencia %", title=f"Eficiência de pagamento ({label_lbl})")
                fig_fp.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig_fp.update_layout(height=300, xaxis_tickangle=-20,
                                     yaxis_title="Eficiência %", xaxis_title=label_lbl)
                st.plotly_chart(fig_fp, use_container_width=True, config={"displayModeBar": False})
            melhor, pior, igual, m_label, p_label = comparar_pagadores(profile, label_cols)
            if melhor is not None:
                if igual:
                    st.success(f"⚖️ **Pagadores semelhantes:** não há diferença relevante de comportamento "
                               f"(~{melhor['Eficiencia %']:.1%} de eficiência).")
                else:
                    st.success(f"🏆 **Melhor pagador:** {m_label} — eficiência {melhor['Eficiencia %']:.1%}, "
                               f"{melhor['% Parcelas pagas']:.1%} das parcelas pagas, atraso médio {melhor['Atraso medio (dias)']}d.")
                    st.error(f"⚠️ **Mais inadimplente:** {p_label} — eficiência {pior['Eficiencia %']:.1%}, "
                             f"{pior['% Aberto 90+']:.1%} do aberto em 90+, atraso médio {pior['Atraso medio (dias)']}d.")

        colP1, colP2 = st.columns(2)
        with colP1:
            age_prof = build_payment_profile(movimentos, ["faixa_idade"])
            render_perfil_pagamento("Por faixa etária", age_prof, ["faixa_idade"])
        with colP2:
            genage_prof = build_payment_profile(movimentos, ["genero_cat", "faixa_idade"])
            render_perfil_pagamento("Por sexo + faixa etária", genage_prof, ["genero_cat", "faixa_idade"])
        seg_cat = movimentos.groupby(["categoria_estabelecimento", "subcategoria_estabelecimento"], dropna=False).agg(
            Recebido=("valorrecebido", "sum"),
            Em_aberto=("areceber", lambda s: s[~movimentos.loc[s.index, "status_pago"]].sum()),
            Atraso_90=("areceber", lambda s: s[movimentos.loc[s.index, "atraso_90"]].sum()),
            Clientes=("idcliente", "nunique"),
        ).reset_index()
        seg_cat["Categoria"] = seg_cat["categoria_estabelecimento"].fillna("OUTROS")
        seg_cat["Subcategoria"] = seg_cat["subcategoria_estabelecimento"].fillna("OUTROS")
        seg_cat["Total"] = seg_cat["Recebido"] + seg_cat["Em_aberto"]
        seg_cat["Eficiencia %"] = seg_cat["Recebido"] / seg_cat["Total"].replace(0, np.nan)
        seg_cat["Inadimplencia"] = seg_cat["Atraso_90"]
        seg_cat = seg_cat.sort_values("Total", ascending=False).reset_index(drop=True)
        if not seg_cat.empty:
            seg_view = seg_cat[["Categoria", "Subcategoria", "Recebido", "Em_aberto", "Inadimplencia", "Eficiencia %"]].copy()
            seg_view["Eficiencia %"] = (seg_view["Eficiencia %"] * 100).round(1)
            seg_view[["Recebido", "Em_aberto", "Inadimplencia"]] = seg_view[["Recebido", "Em_aberto", "Inadimplencia"]].round(2)
            st.markdown("##### Segmentos por categoria e subcategoria")
            show(seg_view.head(20))

            seg_plot = seg_cat.sort_values("Total", ascending=False).head(12).copy()
            seg_plot["Inadimplencia"] = seg_plot["Inadimplencia"].fillna(0)
            fig_seg = px.bar(
                seg_plot,
                x="Subcategoria",
                y=["Recebido", "Inadimplencia"],
                barmode="group",
                color_discrete_sequence=[COLORS["azul"], COLORS["vermelho"]],
                title="Recebido vs inadimplência por subcategoria",
            )
            fig_seg.update_layout(height=340, xaxis_tickangle=-20, legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig_seg, use_container_width=True, config={"displayModeBar": False})

        st.markdown("---")
        st.subheader("Inadimplência vs Recebimento por faixa de valor do contrato")
        st.caption("Quais faixas de VALOR tendem a MAIS recebimento e a MAIS inadimplência (vencido 90+).")

        def render_coorte(titulo, coorte, label_cols):
            st.markdown(f"##### {titulo}")
            if coorte is None or coorte.empty:
                st.info("Sem dados para esta dimensão.")
                return
            view = coorte.copy()
            for col in ["Recebido_geral", "Vencido", "Atraso_90", "Inadimplencia"]:
                view[col] = view[col].round(0)
            view["% Inadimplencia"] = (view["% Inadimplencia"] * 100).round(1)
            show(view)
            if len(coorte) > 1:
                xcol = label_cols[-1] if label_cols else coorte.columns[0]
                label_lbl = " + ".join(label_cols) if label_cols else "Grupo"
                cdf = coorte.copy()
                cdf["Recebido_geral"] = cdf["Recebido_geral"].round(0)
                cdf["Inadimplencia"] = cdf["Inadimplencia"].round(0)
                cdf["_x"] = cdf[xcol].astype(str)
                fig_c = px.bar(cdf, x="_x", y=["Recebido_geral", "Inadimplencia"],
                               barmode="group", color_discrete_sequence=SEQUENCE,
                               title=f"Recebido vs Inadimplência ({label_lbl})")
                fig_c.update_layout(height=320, xaxis_tickangle=-20,
                                    xaxis_title=label_lbl, yaxis_title="R$")
                st.plotly_chart(fig_c, use_container_width=True, config={"displayModeBar": False})
            mais_rec = top_coorte(coorte, "Recebido_geral", label_cols)
            mais_inad = top_coorte(coorte, "Inadimplencia", label_cols)
            maior_risco = top_coorte(coorte, "% Inadimplencia", label_cols)
            if mais_rec:
                st.success("💰 **Mais recebimento (geral):** " +
                           " • ".join(f"{l} (R${v:,.0f})" for l, v in mais_rec))
            if mais_inad:
                st.error("⚠️ **Mais inadimplência em valor:** " +
                         " • ".join(f"{l} (R${v:,.0f})" for l, v in mais_inad))
            if maior_risco:
                st.warning("📛 **Maior risco (% da carteira em inadimplência):** " +
                           " • ".join(f"{l} ({v:.1%})" for l, v in maior_risco))

        colC1, colC2 = st.columns(2)
        with colC1:
            valor_coorte = build_coorte_valor_contrato(contratos)
            if valor_coorte is not None and not valor_coorte.empty:
                render_coorte("Por faixa de valor do contrato", valor_coorte, ["Faixa Valor"])
        with colC2:
            valor_profile_r = build_perfil_valor_contrato(contratos)
            if valor_profile_r is not None and not valor_profile_r.empty:
                vcr = valor_profile_r.copy()
                vcr["Default_%"] = (vcr["Default_%"] * 100).round(1)
                show(vcr[["Faixa Valor", "Contratos", "Clientes", "Valor_total",
                          "Recebido", "Aberto", "Vencido", "Default_%"]])
                faixa_mais_risco = valor_profile_r.loc[valor_profile_r["Default_%"].idxmax()]
                st.warning(f"📛 **Maior default 90d (risco):** {faixa_mais_risco['Faixa Valor']} "
                           f"({faixa_mais_risco['Default_%']:.1%} dos contratos)")
            else:
                st.info("Sem dados para esta dimensão.")

        st.markdown("---")
        st.subheader("Estratégia de valor inicial do contrato")
        st.caption("Quanto oferecer inicialmente a um cliente conforme seu perfil, e como escalar conforme o relacionamento.")

        regras_estr = pd.DataFrame([
            {"Perfil": "Novo", "Valor inicial": "30% do teto", "Escalonamento": "+25% a cada contrato pago em dia",
             "Observação": "Cliente novo, sem histórico de pagamento"},
            {"Perfil": "Risco", "Valor inicial": "25% do teto", "Escalonamento": "+20% a cada contrato pago em dia",
             "Observação": "Histórico com inadimplência 90+"},
            {"Perfil": "Recorrente", "Valor inicial": "55% do teto", "Escalonamento": "+25% a cada contrato pago em dia",
             "Observação": "Relação recorrente, sem default"},
            {"Perfil": "Confiável", "Valor inicial": "75% do teto", "Escalonamento": "+25% a cada contrato pago em dia",
             "Observação": "Relação sólida, ≥2 contratos pagos"},
        ])
        show(regras_estr)

        st.markdown("##### Simulador de valor inicial")
        clientes_disp = sorted(clientes["id"].astype(int).tolist()) if clientes is not None and len(clientes) else []
        sel_id = st.selectbox("Escolha um cliente (por id)", clientes_disp) if clientes_disp else None
        req_val = st.number_input("Valor que o cliente precisa (R$)", min_value=100.0, value=3000.0, step=100.0)
        if sel_id is not None:
            perfil = perfil_cliente(contratos, sel_id)
            rec = recomendar_valor_contrato(perfil, valor_solicitado=req_val)
            st.write(f"**Perfil do cliente #{sel_id}:** {perfil['n_contratos']} contrato(s), "
                     f"default 90d = {perfil['default']}, recebimento médio = "
                     f"{perfil['receb_pct']:.1%}" if perfil["receb_pct"] is not None else
                     f"**Perfil do cliente #{sel_id}:** sem histórico (cliente novo).")
            st.info(f"**Classificação de risco:** 🎯 {rec['classe']} — {rec['descricao']}")
            st.success(f"💡 **Valor inicial sugerido:** R$ {rec['valor_inicial']:,.0f} "
                       f"({rec['pct_inicial']:.0%} do teto de R$ {rec['teto']:,.0f})")
            st.caption("Plano de escalonamento (conforme cada contrato é pago em dia):")
            plano_txt = "\n".join(
                f"- **Etapa {p['etapa']}:** até R$ {p['capacidade']:,.0f} ({p['pct_teto']:.0%} do teto) — "
                f"quando {p['condicao']}" for p in rec["plano"])
            st.markdown(plano_txt)

        st.markdown("---")
        st.subheader("Simulador de viabilidade por perfil")
        st.caption("Verifique se um perfil (sexo, idade, segmento, valor) tende à inadimplência e se é viável conceder o contrato.")

        genero_opts = ["Todos", "Masculino", "Feminino"]
        categorias_opts = ["Todas"] + sorted(movimentos["categoria_estabelecimento"].dropna().unique().astype(str))
        subcategorias_opts = ["Todas"] + sorted(movimentos["subcategoria_estabelecimento"].dropna().unique().astype(str))

        cvin1, cvin2, cvin3, cvin4, cvin5 = st.columns(5)
        with cvin1:
            s_genero = st.selectbox("Sexo", genero_opts, index=0)
        with cvin2:
            s_idade = st.number_input("Idade (anos)", min_value=16, max_value=90, value=26, step=1)
        with cvin3:
            s_categoria = st.selectbox("Categoria", categorias_opts, index=0)
        with cvin4:
            s_subcategoria = st.selectbox("Subcategoria", subcategorias_opts, index=0)
        with cvin5:
            s_valor = st.number_input("Valor do contrato (R$)", min_value=100.0, value=3000.0, step=100.0)

        res = analisar_viabilidade_perfil(movimentos, contratos,
                                          genero=s_genero, idade=s_idade,
                                          categoria=s_categoria if s_categoria != "Todas" else None,
                                          subcategoria=s_subcategoria if s_subcategoria != "Todas" else None,
                                          valor=s_valor)

        colv1, colv2 = st.columns([1, 1])
        with colv1:
            st.metric("Amostra comparável (clientes)", f"{res['amostra_clientes']}")
            st.metric("Contratos comparáveis", f"{res['amostra_contratos']}")
            st.metric("Eficiência de pagamento",
                      f"{res['eficiencia']:.1%}" if res["eficiencia"] is not None else "n/d")
            st.metric("Inadimplência (% do movimentado)",
                      f"{res['inadimplencia_pct']:.1%}" if res["inadimplencia_pct"] is not None else "n/d")
        with colv2:
            st.metric("Default 90d na base", f"{res['default_base_pct']:.1%}"
                      if res["default_base_pct"] is not None else "n/d")
            st.metric("Inadimplência grave 90+ (do vencido)", f"{res['grave_90_pct']:.1%}"
                      if res["grave_90_pct"] is not None else "n/d")
            if res["tendencia"] == "Alta":
                st.error(f"🚨 **Tendência: {res['tendencia']} à inadimplência**")
            elif res["tendencia"] == "Média":
                st.warning(f"⚠️ **Tendência: {res['tendencia']} à inadimplência**")
            elif res["tendencia"] == "Baixa":
                st.success(f"🟢 **Tendência: {res['tendencia']} à inadimplência**")
            else:
                st.info(f"**Tendência: {res['tendencia']}**")

        st.markdown("**Veredito de viabilidade:**")
        if res["tendencia"] == "Alta":
            st.error(f"❌ {res['veredito']}")
        elif res["tendencia"] == "Média":
            st.warning(f"🟠 {res['veredito']}")
        elif res["tendencia"] == "Baixa":
            st.success(f"✅ {res['veredito']}")
        else:
            st.info(res["veredito"])

        if res["valor_inicial_sugerido"]:
            st.info(f"💡 **Valor inicial recomendado para este perfil:** R$ {res['valor_inicial_sugerido']:,.0f} "
                    f"de um total de R$ {s_valor:,.0f} solicitado.")

        if res["eficiencia"] is not None and movimentos is not None and len(movimentos):
            geral = movimentos[~movimentos["dtvenc"].isna()].copy()
            gen_rec = float(geral["valorrecebido"].sum())
            gen_ab = float(geral.loc[~geral["status_pago"], "areceber"].sum())
            gen_ef = gen_rec / (gen_rec + gen_ab) if (gen_rec + gen_ab) > 0 else None
            if gen_ef is not None:
                bar_df = pd.DataFrame({
                    "Grupo": ["Perfil selecionado", "Carteira geral"],
                    "Eficiência %": [res["eficiencia"] * 100, gen_ef * 100],
                })
                fig_vb = px.bar(bar_df, x="Grupo", y="Eficiência %", color="Grupo",
                                color_discrete_sequence=SEQUENCE, text="Eficiência %",
                                title="Eficiência: Perfil selecionado vs Carteira geral")
                fig_vb.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig_vb.update_layout(height=320, showlegend=False)
                st.plotly_chart(fig_vb, use_container_width=True, config={"displayModeBar": False})

    # =========================================================================
    # TAB 8 - VIABILIDADE & LUCRO
    # =========================================================================
    elif aba == "Viabilidade & Lucro":
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

        st.markdown("---")
        st.subheader("Análise por faixas de valor (passo de R$ 500)")
        st.caption("Agrupa contratos por faixa de R$ 500 e permite escolher quais faixas analisar "
                   "(ex.: analisar a inadimplência de R$ 500 a R$ 5.000).")

        faixas_full = build_perfil_por_faixas(contratos, passo=500)
        if faixas_full is not None and not faixas_full.empty:
            all_faixas = faixas_full["Faixa Valor"].tolist()
            colF1, colF2, colF3 = st.columns([2, 2, 1])
            with colF1:
                fronteira_min = int(faixas_full["lo"].min())
                fronteira_max = int(faixas_full["hi"].max())
                sel_min = st.number_input("Faixa inicial (R$)", min_value=fronteira_min,
                                          max_value=fronteira_max, value=min(fronteira_min, 500), step=500,
                                          key="faixa_min")
                sel_min = int(sel_min // 500) * 500
            with colF2:
                sel_max = st.number_input("Faixa final (R$)", min_value=fronteira_min,
                                          max_value=fronteira_max,
                                          value=min(max(fronteira_max, 5000), fronteira_max) if fronteira_max >= 5000 else fronteira_max,
                                          step=500, key="faixa_max")
                sel_max = int(sel_max // 500) * 500
            with colF3:
                passo_faixa = st.selectbox("Passo (R$)", [100, 500, 1000], index=1, key="faixa_passo")

            faixas_filtradas = [f for f in all_faixas
                                if faixas_full.loc[faixas_full["Faixa Valor"] == f, "lo"].iloc[0] >= sel_min
                                and faixas_full.loc[faixas_full["Faixa Valor"] == f, "hi"].iloc[0] <= sel_max]

            fdf = faixas_full[faixas_full["Faixa Valor"].isin(faixas_filtradas)].copy()
            if fdf.empty:
                st.info("Nenhuma faixa no intervalo selecionado.")
            else:
                fview = fdf.copy()
                fview["Default_%"] = (fview["Default_%"] * 100).round(1)
                fview["% Inadimplencia"] = (fview["% Inadimplencia"] * 100).round(1)
                show(fview[["Faixa Valor", "Contratos", "Clientes", "Valor_total",
                            "Recebido", "Aberto", "Vencido", "Default_%", "% Inadimplencia"]]
                     .rename(columns={"Default_%": "Default 90d (%)", "% Inadimplencia": "Inadimplência (%)"}))

                colfg1, colfg2 = st.columns([1, 1])
                with colfg1:
                    fig_fx = px.bar(fdf, x="Faixa Valor", y="Valor_total",
                                    color="Default_%",
                                    color_continuous_scale="RdYlGn_r",
                                    title="Valor originado por faixa (cor = taxa de default)")
                    fig_fx.update_layout(height=360, xaxis_tickangle=-45)
                    st.plotly_chart(fig_fx, use_container_width=True, config={"displayModeBar": False})

                st.markdown("##### Inadimplência e recebimento por faixa de valor (linha inteira)")
                fplot = fdf.copy()
                fplot["Inadimplência %"] = (fplot["% Inadimplencia"] * 100).fillna(0)
                fplot["Recebido (R$)"] = fplot["Recebido"]
                fig_fl = go.Figure()
                fig_fl.add_trace(go.Scatter(x=fplot["Faixa Valor"], y=fplot["Inadimplência %"],
                                            mode="lines+markers+text", name="Inadimplência (%)",
                                            text=[f"{v:.1f}%" for v in fplot["Inadimplência %"]],
                                            textposition="top center", line=dict(color=COLORS["vermelho"], width=3),
                                            yaxis="y"))
                fig_fl.add_trace(go.Scatter(x=fplot["Faixa Valor"], y=fplot["Recebido (R$)"],
                                            mode="lines+markers", name="Recebido (R$)",
                                            line=dict(color=COLORS["azul"], width=2), yaxis="y2"))
                fig_fl.update_layout(
                    title="Evolução por faixa de valor (inadimplência e recebimento)",
                    height=420, xaxis_tickangle=-45,
                    yaxis=dict(title="Inadimplência (%)"),
                    yaxis2=dict(title="Recebido (R$)", overlaying="y", side="right"),
                    legend=dict(orientation="h", y=1.12))
                st.plotly_chart(fig_fl, use_container_width=True, config={"displayModeBar": False})

                if fdf["Default_%"].notna().any():
                    pior = fdf.loc[fdf["Default_%"].idxmax()]
                    st.warning(f"📛 **Faixa com maior risco (default 90d):** {pior['Faixa Valor']} "
                               f"({pior['Default_%']:.1%} dos contratos)")
                if fdf["% Inadimplencia"].notna().any():
                    pior_inad = fdf.loc[fdf["% Inadimplencia"].idxmax()]
                    st.error(f"⚠️ **Faixa com maior inadimplência (% do movimentado):** {pior_inad['Faixa Valor']} "
                             f"({pior_inad['% Inadimplencia']:.1%})")
        else:
            st.info("Sem dados de valor de contrato disponíveis.")

    # =========================================================================
    # TAB - SIMULADOR FINANCEIRO (previsão de juros / crescimento de capital)
    # =========================================================================
    elif aba == "Simulador Financeiro":
        st.subheader("📈 Simulador Financeiro — Projeção de Capital da Carteira")
        st.caption("Simula o crescimento do capital sob diferentes cenários de rentabilidade, "
                   "inadimplência, custos, impostos e reinvestimento. Independente dos dados do banco.")

        with st.expander("Como ler esta aba", expanded=False):
            st.markdown("""
            **Esta aba projeta quanto seu capital pode crescer mês a mês.**

            - **Cenário Ideal:** juros compostos puros — `VF = VP × (1 + taxa)^n`. Todo o retorno é reinvestido.
            - **Cenário Realista:** o retorno bruto é reduzido por inadimplência, perdas, impostos e custos
              operacionais, e apenas uma fração é efetivamente reinvestida.
            - **Sensibilidade:** mostra o capital final variando taxa × inadimplência.
            - **Ponto de equilíbrio:** maior inadimplência que o negócio suporta antes de parar de crescer.
            """)

        ci1, ci2, ci3, ci4 = st.columns(4)
        cap_inicial = ci1.number_input("Capital inicial (R$)", min_value=1000.0, value=250000.0, step=10000.0,
                                       key="sim_cap")
        taxa_bruta = ci2.number_input("Retorno bruto mensal (%)", min_value=0.1, value=8.0, step=0.5,
                                      key="sim_taxa") / 100.0
        n_meses = ci3.number_input("Meses de projeção", min_value=1, max_value=60, value=24, step=1,
                                   key="sim_meses")
        reinv = ci4.number_input("Fração reinvestida (%)", min_value=0, max_value=100, value=100, step=5,
                                 key="sim_reinv") / 100.0

        cj1, cj2, cj3 = st.columns(3)
        inad = cj1.number_input("Inadimplência/perdas (% do retorno)", min_value=0.0, value=20.0, step=1.0,
                                key="sim_inad") / 100.0
        custos = cj2.number_input("Custos operacionais (% do retorno)", min_value=0.0, value=15.0, step=1.0,
                                  key="sim_custos") / 100.0
        impostos = cj3.number_input("Impostos (% do retorno)", min_value=0.0, value=5.0, step=1.0,
                                    key="sim_imp") / 100.0

        # --- Calcula cenários mês a mês ---
        taxa_ideal = taxa_bruta * reinv
        cap_ideal = []
        cap_real = []
        juros_linha = []
        perdas_linha = []
        custos_linha = []
        meses = list(range(0, n_meses + 1))

        for m in range(n_meses + 1):
            cap_ideal.append(cap_inicial * (1 + taxa_ideal) ** m)
            if m == 0:
                cap_real.append(cap_inicial)
                juros_linha.append(0.0)
                perdas_linha.append(0.0)
                custos_linha.append(0.0)
            else:
                cap_prev = cap_real[-1]
                juros = cap_prev * taxa_bruta
                perda = juros * inad
                custo = juros * custos
                imp = juros * impostos
                juros_linha.append(juros)
                perdas_linha.append(perda + imp)
                custos_linha.append(custo)
                liquido = juros - perda - custo - imp
                cap_real.append(cap_prev + liquido * reinv)

        df_sim = pd.DataFrame({
            "Mês": meses,
            "Capital Ideal (R$)": cap_ideal,
            "Capital Realista (R$)": cap_real,
            "Juros no mês (R$)": juros_linha,
            "Perdas + Impostos (R$)": perdas_linha,
            "Custos (R$)": custos_linha,
        })
        for col in ["Capital Ideal (R$)", "Capital Realista (R$)", "Juros no mês (R$)",
                    "Perdas + Impostos (R$)", "Custos (R$)"]:
            df_sim[col] = df_sim[col].round(2)

        st.markdown("---")
        capt1, capt2 = st.columns(2)
        capt1.metric("Capital final — cenário ideal", fmt_brl(cap_ideal[-1]))
        capt2.metric("Capital final — cenário realista", fmt_brl(cap_real[-1]))

        # --- Marcos de capital ---
        marcos = {500000: "R$ 500 mil", 750000: "R$ 750 mil", 1000000: "R$ 1 milhão",
                  2000000: "R$ 2 milhões", 5000000: "R$ 5 milhões"}
        lin_real = pd.Series(cap_real)
        lin_ideal = pd.Series(cap_ideal)
        st.markdown("**Mês em que o capital atinge cada marco (— = não atinge no período):**")
        cols_marco = st.columns(4)
        marco_items = list(marcos.items())
        for i, (mvalor, mnome) in enumerate(marco_items):
            mes_real = lin_real[lin_real >= mvalor].index.min()
            mes_ideal = lin_ideal[lin_ideal >= mvalor].index.min()
            with cols_marco[i % 4]:
                st.markdown(f"**{mnome}**")
                st.markdown(f"- Realista: `mês {mes_real}`" if pd.notna(mes_real) else "- Realista: não atinge")
                st.markdown(f"- Ideal: `mês {mes_ideal}`" if pd.notna(mes_ideal) else "- Ideal: não atinge")

        st.markdown("---")
        st.markdown("### Evolução do capital")
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(x=df_sim["Mês"], y=df_sim["Capital Ideal (R$)"],
                                     mode="lines+markers", name="Cenário ideal (juros compostos)",
                                     line=dict(color=COLORS["azul"], width=2)))
        fig_sim.add_trace(go.Scatter(x=df_sim["Mês"], y=df_sim["Capital Realista (R$)"],
                                     mode="lines+markers", name="Cenário realista (com perdas/custos)",
                                     line=dict(color=COLORS["laranja"], width=2)))
        fig_sim.update_layout(title=f"Projeção de capital — {n_meses} meses (início R$ {cap_inicial:,.0f})",
                              height=400, xaxis_title="Mês", yaxis_title="Capital (R$)",
                              legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig_sim, use_container_width=True, config={"displayModeBar": False})

        with st.expander("Ver tabela mês a mês", expanded=False):
            show(df_sim)

        st.markdown("---")
        st.markdown("### Análise de sensibilidade (capital final após o período)")
        st.caption("Varia o retorno bruto × inadimplência para ver como o capital final se comporta.")
        sens_taxas = [0.03, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12]
        sens_inads = [0.0, 0.10, 0.20, 0.30, 0.40]

        sens_rows = []
        for t in sens_taxas:
            for iad in sens_inads:
                c = cap_inicial
                for _ in range(n_meses):
                    j = c * t
                    c = c + (j - j * iad - j * custos - j * impostos) * reinv
                sens_rows.append({"Taxa mensal": f"{t:.1%}", "Inadimplência": f"{iad:.0%}",
                                  "Capital final (R$)": c})
        df_sens = pd.DataFrame(sens_rows)
        df_sens["Capital final (R$)"] = df_sens["Capital final (R$)"].round(0).astype(int)

        fig_sens = px.density_heatmap(
            df_sens, x="Inadimplência", y="Taxa mensal", z="Capital final (R$)",
            color_continuous_scale="RdYlGn_r", title="Capital final (R$) — taxa × inadimplência")
        fig_sens.update_layout(height=420)
        st.plotly_chart(fig_sens, use_container_width=True, config={"displayModeBar": False})

        with st.expander("Ver tabela de sensibilidade", expanded=False):
            show(df_sens)

        st.markdown("---")
        st.markdown("### Qual taxa de inadimplência máxima o negócio aguenta?")
        st.caption("Maior inadimplência com a qual o capital ainda cresce (líquido de custos e impostos) "
                   "considerando a fração reinvestida.")
        custo_total_pct = custos + impostos
        if reinv > 0 and (1 - custo_total_pct) > 0:
            inad_max = max(0.0, 1 - custo_total_pct / reinv)
            st.success(f"**O capital deixa de crescer quando a inadimplência + perdas supera "
                       f"`{(reinv - custo_total_pct):.1%}` do retorno bruto.** "
                       f"Ou seja, a inadimplência máxima sustentável é **aproximadamente {inad_max:.0%}** "
                       f"do retorno bruto (considerando {custos:.0%} de custos e {impostos:.0%} de impostos "
                       f"sobre o retorno, reinvestindo {reinv:.0%}).")
        else:
            st.info("Não é possível calcular o ponto de equilíbrio com os parâmetros atuais "
                    "(ajuste a fração reinvestida e os custos).")

    # =========================================================================
    # TAB 9 - MODELOS PREDITIVOS
    # =========================================================================
    elif aba == "Modelos Preditivos":
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
    elif aba == "Visualizacoes":
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
    elif aba == "Dados":
        st.subheader("Dados brutos")
        
        # Mostrar dados com base no filtro aplicado
        st.markdown("##### Dados do período filtrado")
        if st.checkbox("Mostrar movimentacoes (periodo filtrado)", value=False):
            show(movimentos)
        if st.checkbox("Mostrar contratos (periodo filtrado)", value=False):
            show(contratos)
        
        st.markdown("##### Dados completos (sem filtro)")
        if st.checkbox("Mostrar todos os contratos (portfolio completo)", value=False):
            show(contratos_total)
        if st.checkbox("Mostrar todos os movimentos (portfolio completo)", value=False):
            show(movimentos_total)

if __name__ == "__main__":
    main()