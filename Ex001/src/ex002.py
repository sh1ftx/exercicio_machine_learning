import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
import numpy as np
import os
import time
import sys

console = Console()

# Caminho absoluto do CSV
csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'database', 'maternal_risk.csv'))

def type_writer(text, delay=0.03):
    """Simula efeito de texto sendo escrito letra por letra no terminal."""
    for char in text:
        console.print(char, end="", soft_wrap=True, highlight=False)
        time.sleep(delay)
    console.print()  # pular linha no final

def load_dataset():
    with console.status("[bold cyan]Carregando dataset...[/bold cyan]", spinner="dots"):
        if not os.path.isfile(csv_path):
            console.print(f"[bold red]Arquivo não encontrado:[/bold red] {csv_path}")
            raise FileNotFoundError(csv_path)
        df = pd.read_csv(csv_path)
    console.print("[bold green]✔ Dataset carregado com sucesso![/bold green]\n")
    return df

metadata = {
    'Variable Name': ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate', 'RiskLevel'],
    'Role': ['Feature', 'Feature', 'Feature', 'Feature', 'Feature', 'Feature', 'Target'],
    'Type': ['Integer'] * 6 + ['Categorical'],
    'Demographic': ['Age', '', '', '', '', '', 'Risk category'],
    'Description': [
        'Idade da mulher em anos',
        'Pressão sistólica (valor superior) em mmHg',
        'Pressão diastólica (valor inferior) em mmHg',
        'Nível de glicose no sangue em mmol/L',
        'Temperatura corporal em graus Celsius',
        'Frequência cardíaca em batimentos por minuto',
        'Nível de risco durante a gestação'
    ],
    'Units': ['', 'mmHg', 'mmHg', 'mmol/L', '°C', 'bpm', ''],
    'Missing Values': ['no'] * 7
}

def print_rich_table(data_dict, title, header_style="bold blue", max_col_width=30):
    table = Table(title=title, header_style=header_style, box=box.SIMPLE_HEAVY, show_lines=True)
    for key in data_dict.keys():
        table.add_column(key, style="cyan", max_width=max_col_width, overflow="fold")
    for i in range(len(data_dict['Variable Name'])):
        table.add_row(*[str(data_dict[key][i]) for key in data_dict])
    console.print(table)

def print_dataset_head(df, title="Primeiras Linhas do Dataset", max_col_width=20):
    console.print(Panel(f"[bold magenta]Mostrando as primeiras 5 linhas do dataset[/bold magenta]", style="magenta"))
    table = Table(show_header=True, header_style="bold green", box=box.SIMPLE_HEAVY, show_lines=True)
    for col in df.columns:
        table.add_column(col, style="yellow", max_width=max_col_width, overflow="fold")
    for _, row in df.head().iterrows():
        table.add_row(*[str(val) for val in row])
    console.print(table)
    legenda = (
        "[bold yellow]Legenda das Colunas:[/bold yellow]\n"
        "[cyan]Age[/cyan]: Idade da mulher em anos\n"
        "[cyan]SystolicBP[/cyan]: Pressão arterial sistólica (mmHg)\n"
        "[cyan]DiastolicBP[/cyan]: Pressão arterial diastólica (mmHg)\n"
        "[cyan]BS[/cyan]: Glicose no sangue (mmol/L)\n"
        "[cyan]BodyTemp[/cyan]: Temperatura corporal (°C)\n"
        "[cyan]HeartRate[/cyan]: Frequência cardíaca (bpm)\n"
        "[cyan]RiskLevel[/cyan]: Categoria de risco (low, mid, high)\n"
    )
    console.print(Panel(legenda, title="📖 Legenda do Dataset", border_style="yellow"))

def print_descriptive_stats(df, title="Estatísticas Descritivas", max_col_width=20):
    console.print(Panel(f"[bold magenta]Estatísticas descritivas do dataset[/bold magenta]", style="magenta"))
    desc = df.describe()
    table = Table(show_header=True, header_style="bold red", box=box.SIMPLE_HEAVY, show_lines=True)
    for col in desc.columns:
        table.add_column(col, style="yellow", max_width=max_col_width, overflow="fold")
    for _, row in desc.iterrows():
        table.add_row(*[f"{val:.2f}" for val in row])
    console.print(table)
    legenda = (
        "[bold yellow]Legenda das Estatísticas:[/bold yellow]\n"
        "- count: quantidade de registros\n"
        "- mean: média\n"
        "- std: desvio padrão\n"
        "- min: valor mínimo\n"
        "- 25%, 50%, 75%: percentis\n"
        "- max: valor máximo\n"
    )
    console.print(Panel(legenda, title="📖 Legenda das Estatísticas", border_style="yellow"))

def print_classification_report(report, classes, title="Relatório de Classificação", max_col_width=15):
    console.print(Panel(f"[bold magenta]Relatório de classificação do modelo[/bold magenta]", style="magenta"))
    table = Table(show_header=True, header_style="bold yellow", box=box.SIMPLE_HEAVY, show_lines=True)
    metrics = list(next(iter(report.values())).keys())
    table.add_column("Classe", style="cyan", max_width=max_col_width)
    for metric in metrics:
        table.add_column(metric.capitalize(), style="magenta", max_width=max_col_width)

    for label in classes:
        row = [label] + [f"{report[label][metric]:.2f}" for metric in metrics]
        table.add_row(*row)
    for key in ['accuracy', 'macro avg', 'weighted avg']:
        if key in report:
            if key == 'accuracy':
                acc_val = f"{report[key]:.2f}"
                table.add_row("Accuracy", acc_val, "", "", "")
            else:
                row = [key.capitalize()] + [f"{report[key][metric]:.2f}" for metric in metrics]
                table.add_row(*row)
    console.print(table)

    legenda = (
        "[bold yellow]Legenda do relatório:[/bold yellow]\n"
        "- Precision: proporção de predições corretas para a classe\n"
        "- Recall: proporção de casos reais da classe corretamente detectados\n"
        "- F1-score: média harmônica entre precision e recall\n"
        "- Support: número de amostras reais daquela classe\n"
    )
    console.print(Panel(legenda, title="📖 Legenda do Relatório", border_style="yellow"))

def print_confusion_matrix(cm, classes, title="Matriz de Confusão"):
    console.print(Panel(f"[bold magenta]{title}[/bold magenta]", style="magenta"))
    table = Table(show_header=True, header_style="bold blue", box=box.SIMPLE_HEAVY, show_lines=True)
    table.add_column("Real \\ Predito", style="cyan", justify="center")
    for c in classes:
        table.add_column(c, justify="center", style="green")
    for i, row in enumerate(cm):
        row_data = [str(v) for v in row]
        table.add_row(classes[i], *row_data)
    console.print(table)

    legenda = (
        "[bold yellow]Legenda da Matriz de Confusão:[/bold yellow]\n"
        "- Linhas representam a classe real\n"
        "- Colunas representam a classe predita\n"
        "- Diagonal principal: acertos\n"
        "- Valores fora da diagonal: erros de classificação\n"
    )
    console.print(Panel(legenda, title="📖 Legenda da Matriz de Confusão", border_style="yellow"))

def print_feature_importance(importances, features, title="Importância das Variáveis"):
    console.print(Panel(f"[bold magenta]{title}[/bold magenta]", style="magenta"))
    table = Table(box=box.SIMPLE_HEAVY, show_lines=True)
    table.add_column("Variável", style="cyan", no_wrap=True)
    table.add_column("Importância", style="green", justify="right")
    table.add_column("Barra Visual", style="yellow")

    max_importance = max(importances)
    for feat, imp in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
        bar_length = int((imp / max_importance) * 20)
        bar = "█" * bar_length + "-" * (20 - bar_length)
        table.add_row(feat, f"{imp:.4f}", bar)
    console.print(table)

def main():
    df = load_dataset()

    # Renomear colunas para nomes amigáveis
    df.rename(columns={
        'A1': 'Age', 'A2': 'SystolicBP', 'A3': 'DiastolicBP',
        'A4': 'BS', 'A5': 'BodyTemp', 'A6': 'HeartRate', 'A7': 'RiskLevel'
    }, inplace=True)

    # Exibir metadados
    print_rich_table(metadata, "📋 Metadados das Variáveis")

    # Exibir primeiras linhas do dataset com legenda
    print_dataset_head(df)

    # Estatísticas descritivas com legenda
    print_descriptive_stats(df)

    # Explicação didática do passo de preparação dos dados
    type_writer(
        "\nAgora vamos preparar os dados para o modelo de machine learning.\n"
        "Selecionamos as colunas com informações que ajudam a prever o risco (idade, pressão arterial, glicose, etc).\n"
        "E também vamos transformar as categorias de risco (low, mid, high) em números para o modelo entender.\n"
    )

    X = df[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']]
    y = df['RiskLevel']

    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    n_runs = 1000
    accuracies = []

    type_writer(
        f"\nVamos treinar e testar o modelo {n_runs} vezes, cada vez com uma divisão diferente dos dados.\n"
        "Isso serve para garantir que o resultado final seja confiável e não dependente de uma única divisão.\n"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        transient=True,
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Treinando e testando o modelo...", total=n_runs)
        for i in range(n_runs):
            # Divide dados treino/teste aleatoriamente
            X_train, X_test, y_train_enc, y_test_enc = train_test_split(X, y_enc, test_size=0.2)
            # Cria e treina o modelo Random Forest
            model = RandomForestClassifier()
            model.fit(X_train, y_train_enc)
            # Faz a predição nos dados de teste
            y_pred = model.predict(X_test)
            # Calcula a acurácia da predição
            acc = accuracy_score(y_test_enc, y_pred)
            accuracies.append(acc)
            progress.advance(task)

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    type_writer(
        f"\nApós {n_runs} treinamentos e testes, a média da acurácia obtida foi de {mean_acc*100:.2f}%.\n"
        f"O desvio padrão, que mostra a variação entre os resultados, foi de {std_acc*100:.2f}%.\n"
        "Isso significa que nosso modelo é, em média, essa porcentagem correto, e que a variação dos resultados é relativamente pequena.\n"
    )

    console.print(Panel(
        f"[bold green]Acurácia média após {n_runs} rodadas: {mean_acc*100:.2f}%[/bold green]\n"
        f"[bold yellow]Desvio padrão da acurácia: {std_acc*100:.2f}%[/bold yellow]",
        title="🎯 Resultado da Avaliação",
        border_style="green"
    ))

    type_writer(
        "\nAgora vamos fazer um treinamento final usando uma divisão fixa dos dados para analisar com calma os resultados.\n"
        "Com isso, podemos gerar relatórios detalhados, ver onde o modelo acerta e erra, e entender quais variáveis são mais importantes para a decisão.\n"
    )

    # Treinamento final com seed fixa para relatório detalhado
    X_train, X_test, y_train_enc, y_test_enc = train_test_split(X, y_enc, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train_enc)
    y_pred = model.predict(X_test)

    # Relatório de classificação
    report = classification_report(y_test_enc, y_pred, target_names=label_encoder.classes_, output_dict=True)
    print_classification_report(report, label_encoder.classes_)

    # Matriz de confusão
    cm = confusion_matrix(y_test_enc, y_pred)
    print_confusion_matrix(cm, label_encoder.classes_)

    # Importância das variáveis
    print_feature_importance(model.feature_importances_, X.columns)

if __name__ == "__main__":
    main()
