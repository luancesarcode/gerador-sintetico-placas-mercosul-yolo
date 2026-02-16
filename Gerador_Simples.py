import argparse
import random
import shutil
import string
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DEFAULT_PLATE_TEMPLATE = "assets/placa_mercosul.png"
DEFAULT_FONT_PATH = "assets/fonts/FE-Schrift.TTF"
DEFAULT_PLATE_TEMPLATES_DIR = "assets/plates"
DEFAULT_NUM_IMAGES = 500
DEFAULT_FONT_SIZE = 340
DEFAULT_TEXT_X = 260
DEFAULT_TEXT_Y = 220
DEFAULT_CHAR_SPACING = 12


def gerar_texto_placa():
    return (
        "".join(random.choices(string.ascii_uppercase, k=3))
        + random.choice("0123456789")
        + random.choice(string.ascii_uppercase)
        + "".join(random.choices("0123456789", k=2))
    )


def listar_arquivos_imagem(dir_path):
    if not dir_path:
        return []
    path = Path(dir_path)
    if not path.is_dir():
        return []
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files = []
    for ext in exts:
        files.extend(path.glob(ext))
        files.extend(path.glob(ext.upper()))
    return sorted(set(files))


def carregar_templates(template_padrao, pasta_templates):
    caminhos_templates = listar_arquivos_imagem(pasta_templates)
    templates = []
    for caminho_template in caminhos_templates:
        try:
            with Image.open(caminho_template) as img:
                templates.append(img.convert("RGBA"))
        except Exception:
            continue

    if templates:
        return templates

    template_padrao = Path(template_padrao)
    if not template_padrao.is_file():
        raise FileNotFoundError(
            f"Template de placa nao encontrado: {template_padrao}"
        )
    with Image.open(template_padrao) as img:
        return [img.convert("RGBA")]


def extrair_prefixo_numerico(nome_base):
    prefixo, separador, _ = nome_base.partition("_")
    if separador and prefixo.isdigit():
        return int(prefixo)
    return None


def proximo_indice_inicial(stems_usados):
    maior_indice = 0
    for nome_base in stems_usados:
        indice = extrair_prefixo_numerico(nome_base)
        if indice is not None and indice > maior_indice:
            maior_indice = indice
    return maior_indice + 1


def stem_placa_unico(texto_placa, stems_usados, indice):
    while True:
        nome_base = f"{indice:05d}_{texto_placa}"
        if nome_base not in stems_usados:
            stems_usados.add(nome_base)
            return nome_base, indice + 1
        indice += 1


def desenhar_texto_na_placa(
    placa_rgba,
    texto_placa,
    fonte,
    text_x,
    text_y,
    char_spacing,
):
    placa_saida = placa_rgba.copy()
    desenho = ImageDraw.Draw(placa_saida)
    cursor_x = text_x

    for caractere in texto_placa:
        desenho.text((cursor_x, text_y), caractere, font=fonte, fill=(0, 0, 0, 255))
        caixa_texto = desenho.textbbox((cursor_x, text_y), caractere, font=fonte)
        largura_char = caixa_texto[2] - caixa_texto[0]
        cursor_x += largura_char + char_spacing

    return placa_saida


def reconstruir_arquivo_numeros(pasta_imagens, caminho_numeros):
    linhas = []
    for caminho_png in sorted(pasta_imagens.glob("*.png")):
        nome_base = caminho_png.stem
        _, separador, texto_placa = nome_base.partition("_")
        if separador:
            linhas.append(f"{caminho_png.name};{texto_placa}")
        else:
            linhas.append(f"{caminho_png.name};")
    conteudo = "\n".join(linhas)
    if conteudo:
        conteudo += "\n"
    caminho_numeros.write_text(conteudo, encoding="utf-8")


def analisar_argumentos():
    analisador = argparse.ArgumentParser(
        description=(
            "Gera apenas placas em PNG, sem fundo adicional, sem distorcao "
            "e mantendo a resolucao original do template."
        )
    )
    analisador.add_argument("--num-images", type=int, default=DEFAULT_NUM_IMAGES)
    analisador.add_argument("--plate-template", default=DEFAULT_PLATE_TEMPLATE)
    analisador.add_argument("--plate-templates-dir", default=DEFAULT_PLATE_TEMPLATES_DIR)
    analisador.add_argument("--font-path", default=DEFAULT_FONT_PATH)
    analisador.add_argument("--font-size", type=int, default=DEFAULT_FONT_SIZE)
    analisador.add_argument("--text-x", type=int, default=DEFAULT_TEXT_X)
    analisador.add_argument("--text-y", type=int, default=DEFAULT_TEXT_Y)
    analisador.add_argument("--char-spacing", type=int, default=DEFAULT_CHAR_SPACING)
    analisador.add_argument("--output-dir", default="dataset")
    analisador.add_argument("--seed", type=int, default=None)
    analisador.add_argument(
        "--clean",
        action="store_true",
        help="Remove a pasta de imagens antes de gerar.",
    )
    return analisador.parse_args()


def principal():
    argumentos = analisar_argumentos()

    if argumentos.num_images < 0:
        raise ValueError("num-images deve ser >= 0.")
    if argumentos.char_spacing < 0:
        raise ValueError("char-spacing deve ser >= 0.")

    if argumentos.seed is not None:
        random.seed(argumentos.seed)

    caminho_fonte = Path(argumentos.font_path)
    if not caminho_fonte.is_file():
        raise FileNotFoundError(f"Arquivo de fonte nao encontrado: {caminho_fonte}")

    templates_placa = carregar_templates(
        argumentos.plate_template, argumentos.plate_templates_dir
    )
    fonte = ImageFont.truetype(str(caminho_fonte), argumentos.font_size)

    diretorio_saida = Path(argumentos.output_dir)
    pasta_imagens = diretorio_saida / "images"
    caminho_numeros = diretorio_saida / "numeros_gerados.txt"

    if argumentos.clean and pasta_imagens.exists():
        shutil.rmtree(pasta_imagens)

    pasta_imagens.mkdir(parents=True, exist_ok=True)

    stems_imagens = {p.stem for p in pasta_imagens.glob("*.png")}
    faltantes = max(0, argumentos.num_images - len(stems_imagens))
    indice_nome = proximo_indice_inicial(stems_imagens)

    for _ in range(faltantes):
        texto_placa = gerar_texto_placa()
        nome_base, indice_nome = stem_placa_unico(
            texto_placa, stems_imagens, indice_nome
        )
        caminho_imagem = pasta_imagens / f"{nome_base}.png"

        template = random.choice(templates_placa)
        placa_saida = desenhar_texto_na_placa(
            template,
            texto_placa,
            fonte,
            argumentos.text_x,
            argumentos.text_y,
            argumentos.char_spacing,
        )
        placa_saida.save(caminho_imagem, format="PNG")

    reconstruir_arquivo_numeros(pasta_imagens, caminho_numeros)
    print("Placas PNG geradas com sucesso.")
    print(f"Imagens: {pasta_imagens}")
    print(f"Numeros: {caminho_numeros}")

