"""
Pipeline de preparação de dados para Nova-U.
Suporta: PDF, Markdown, código-fonte, JSON, JSONL, texto puro.

Uso:
  # PDF de livros/documentos → dataset de treino
  python scripts/prepare_data.py --pdf livro.pdf --output data/meu_dataset.jsonl

  # Diretório com PDFs, códigos, docs
  python scripts/prepare_data.py --input-dir ./documentos/ --output data/dataset.jsonl

  # Apenas extrair texto de PDFs (sem formato de conversa)
  python scripts/prepare_data.py --pdf livro.pdf --output data/texto.txt --plain-text
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

# --- System prompt padrão Novo Seguros ---
SYSTEM_PROMPT = (
    "Você é o Novo Cortex, uma IA da Novo Seguros. Você atua como um assistente "
    "inteligente e especialista em desenvolvimento de software e tecnologia, "
    "com foco exclusivo em seguros e insurtech.\n\n"
    "Diretrizes de Comportamento:\n"
    "1. Identidade e Escopo: Seu nome é Novo Cortex. Sua expertise é restrita a "
    "tecnologia, arquitetura de sistemas e desenvolvimento de software aplicados ao mercado de seguros (Insurtech).\n"
    "2. Restrição de Assunto: Se o usuário solicitar informações, orientações ou "
    "discussões que fujam do tema de tecnologia e seguros, você deve responder "
    "estritamente: 'Não estou autorizada a falar sobre esse assunto. Aguardo sua próxima pergunta.'\n"
    "3. Clareza e Objetividade: Responda diretamente ao que foi solicitado. "
    "Seja preciso, estruturado e evite repetições ou textos desnecessários (fluff).\n"
    "4. Raciocínio Técnico: Ao oferecer ajuda em codificação ou análise, priorize "
    "passos concretos, exemplos de código funcionais e decisões fundamentadas em vez de descrições genéricas.\n"
    "5. Tratamento de Ambiguidade: Se um pedido técnico for incompleto, indique o "
    "que falta e assuma a premissa mínima razoável para fornecer uma solução imediata e utilizável.\n"
    "6. Idioma: Todas as interações devem ser realizadas obrigatoriamente em português do Brasil (pt-BR)."
)

IGNORE_DIRS = {"venv", ".venv", "node_modules", ".git", "__pycache__", ".next", "dist", "build"}
CODE_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".c", ".cpp", ".java"}
MARKDOWN_EXTENSIONS = {".md", ".mdx", ".markdown"}


# --- Extratores ---

def extract_pdf(filepath):
    try:
        import fitz
    except ImportError:
        print("[ERRO] PyMuPDF (fitz) não está instalado. Instale com: pip install PyMuPDF", file=sys.stderr)
        return ""
    try:
        doc = fitz.open(filepath)
        pages = []
        for page in doc:
            text = page.get_text()
            if text.strip():
                pages.append(text.strip())
        doc.close()
        return "\n\n".join(pages)
    except Exception as e:
        print(f"[AVISO] PDF falhou: {filepath}: {e}", file=sys.stderr)
        return ""


def extract_text_file(filepath):
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


# --- Chunking ---

def chunk_text(text, max_chars=3000, overlap=200):
    if len(text) <= max_chars:
        return [text]
    chunks = []
    paragraphs = text.split("\n\n")
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current += para + "\n\n"
        else:
            if current.strip():
                chunks.append(current.strip())
            if len(para) > max_chars:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                sub = ""
                for s in sentences:
                    if len(sub) + len(s) + 1 <= max_chars:
                        sub += s + " "
                    else:
                        if sub.strip():
                            chunks.append(sub.strip())
                        sub = s + " "
                current = sub if sub.strip() else ""
            else:
                current = para + "\n\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks


# --- Geração de conversas ---

def file_type_label(filepath):
    ext = Path(filepath).suffix.lower()
    labels = {
        ".pdf": "documento PDF",
        ".md": "documentação Markdown",
        ".txt": "arquivo de texto",
    }
    for code_ext in CODE_EXTENSIONS:
        labels[code_ext] = f"código {code_ext[1:]}"
    return labels.get(ext, "documento")


def generate_conversations(filepath, text, max_chunk_chars=3000, include_system=True):
    filename = Path(filepath).name
    ftype = file_type_label(filepath)
    ext = Path(filepath).suffix.lower()
    chunks = chunk_text(text, max_chars=max_chunk_chars)
    for chunk in chunks:
        if not chunk.strip():
            continue
        if ext == ".pdf":
            user_msg = f"Com base no seguinte trecho do {ftype} '{filename}', explique o conteúdo:\n\n{chunk}"
        elif ext in MARKDOWN_EXTENSIONS:
            user_msg = f"Com base no seguinte trecho do {ftype} '{filename}', explique o conteúdo:\n\n{chunk}"
        elif ext in CODE_EXTENSIONS:
            user_msg = f"Explique o seguinte {ftype} do arquivo '{filename}':\n\n```\n{chunk}\n```"
        else:
            user_msg = f"Explique o conteúdo do arquivo '{filename}':\n\n{chunk}"
        assistant_msg = f"Com base no {ftype} '{filename}', aqui está a informação:\n\n{chunk}"
        conv = {"conversations": []}
        if include_system:
            conv["conversations"].append({"role": "system", "content": SYSTEM_PROMPT})
        conv["conversations"].append({"role": "user", "content": user_msg})
        conv["conversations"].append({"role": "assistant", "content": assistant_msg})
        yield conv


# --- Pipeline ---

def run_pipeline(args):
    seen_hashes = set()
    total = 0
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    mode = "w" if not args.append else "a"
    out = open(args.output, mode, encoding="utf-8")

    # PDF individual
    if args.pdf:
        for pdf_path in args.pdf:
            print(f"  Lendo PDF: {pdf_path}")
            text = extract_pdf(pdf_path)
            if not text.strip():
                print(f"    [AVISO] PDF vazio: {pdf_path}", file=sys.stderr)
                continue
            print(f"    {len(text)} caracteres extraídos")
            if args.plain_text:
                out.write(text + "\n")
                total += 1
            else:
                for conv in generate_conversations(pdf_path, text, args.chunk_size, not args.no_system):
                    h = hashlib.md5(json.dumps(conv, sort_keys=True).encode()).hexdigest()
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        out.write(json.dumps(conv, ensure_ascii=False) + "\n")
                        total += 1
            print(f"    Gerados exemplos até agora: {total}")

    # Diretório
    if args.input_dir:
        files = []
        for d in args.input_dir:
            for root, dirs, filenames in os.walk(d):
                dirs[:] = [x for x in dirs if x not in IGNORE_DIRS]
                for fn in filenames:
                    ext = Path(fn).suffix.lower()
                    if ext in {".pdf"} | MARKDOWN_EXTENSIONS | CODE_EXTENSIONS | {".txt", ".json", ".jsonl"}:
                        files.append(os.path.join(root, fn))
        files.sort()
        print(f"  Encontrados {len(files)} arquivos")
        for fpath in files:
            print(f"  Processando: {fpath}")
            ext = Path(fpath).suffix.lower()
            if ext == ".pdf":
                text = extract_pdf(fpath)
            else:
                text = extract_text_file(fpath)
            if not text.strip():
                continue
            if args.plain_text:
                out.write(text + "\n")
                total += 1
            else:
                for conv in generate_conversations(fpath, text, args.chunk_size, not args.no_system):
                    h = hashlib.md5(json.dumps(conv, sort_keys=True).encode()).hexdigest()
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        out.write(json.dumps(conv, ensure_ascii=False) + "\n")
                        total += 1
            print(f"    Total: {total}")

    out.close()
    print(f"\nFinalizado! {total} exemplos em {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Prepara dados para Nova-U")
    parser.add_argument("--pdf", nargs="+", help="Arquivo(s) PDF para extrair")
    parser.add_argument("--input-dir", "-i", nargs="+", help="Diretório(s) com arquivos fonte")
    parser.add_argument("--output", "-o", default="data/dataset.jsonl", help="Arquivo de saída")
    parser.add_argument("--chunk-size", type=int, default=3000)
    parser.add_argument("--no-system", action="store_true", help="Remove system prompt")
    parser.add_argument("--plain-text", action="store_true", help="Saída como texto puro (não JSONL)")
    parser.add_argument("--append", action="store_true", help="Adicionar ao arquivo existente")
    args = parser.parse_args()

    if not args.pdf and not args.input_dir:
        parser.print_help()
        sys.exit(1)

    run_pipeline(args)


if __name__ == "__main__":
    main()
