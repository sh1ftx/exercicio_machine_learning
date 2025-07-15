import pandas as pd
import matplotlib.pyplot as plt 

# Verificando a versão do pandas
print("Testando a versão do pandas")
print(pd.__version__)

# Lendo o arquivo CpdV com separador correto
print("\nLendo o arquivo .CSV")
df = pd.read_csv('../database/database_wine_red.csv', sep=';')

# Renomeando para nomes mais simples
df.columns = [
    'acidez_fixa',
    'acidez_volatil',
    'acido_citrico',
    'acucar_residual',
    'cloretos',
    'dioxido_enxofre_livre',
    'dioxido_enxofre_total',
    'densidade',
    'ph',
    'sulfatos',
    'alcool',
    'qualidade'
]

# Diagnóstico inicial
print("\nColunas renomeadas:")
print(df.columns.tolist())

print("\nValores nulos por coluna:")
print(df.isnull().sum())

print("\nTipos de dados:")
print(df.dtypes)

# Tratando dados nulos - preenchendo com média
df.fillna(df.mean(numeric_only=True), inplace=True)

# Verificando valores fora do comum (exemplo: pH fora de 0-14)
df = df[(df['ph'] >= 0) & (df['ph'] <= 14)]

# Corrigindo valores negativos indevidos (exemplo: álcool)
df.loc[df['alcool'] < 0, 'alcool'] = df[df['alcool'] >= 0]['alcool'].mean()

# Exibindo estatísticas finais
print("\nResumo estatístico após limpeza:")
print(df.describe())

# Testes gerais
