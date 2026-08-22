"""
dividir_dataset.py

Divide um dataset organizado em pastas por classe (sem divisao previa)
em tres particoes: treino, validacao e teste.

Motivacao: o dataset DrGFreeman traz apenas tres pastas ('rock',
'paper', 'scissors') sem separacao. Precisamos de tres particoes
distintas para que o early stopping use a validacao e o teste seja
tocado apenas na avaliacao final, evitando o vazamento de selecao que
existia no pipeline anterior (onde o teste servia tambem de validacao).

Caracteristicas:
  - Estratificado: cada classe eh dividida na mesma proporcao, de modo
    que as tres particoes fiquem balanceadas.
  - Reproduzivel: com a mesma semente, a divisao eh sempre identica.
  - Nao destrutivo: os arquivos sao COPIADOS, o dataset original
    permanece intacto.

Estrutura esperada na entrada:
    <ENTRADA>/
    ├── paper/
    ├── rock/
    └── scissors/

Estrutura gerada na saida:
    <SAIDA>/
    ├── train/{paper,rock,scissors}/
    ├── val/{paper,rock,scissors}/
    └── test/{paper,rock,scissors}/

Uso:
    python dividir_dataset.py --input data/raw/rps-real-original \
                              --output data/raw/rps-real
"""

import argparse
import os
import random
import shutil
import sys

EXTENSOES_VALIDAS = (".jpg", ".jpeg", ".png", ".bmp")


def listar_classes(pasta_entrada: str) -> list[str]:
    """Retorna as subpastas (classes) em ordem alfabetica.

    A ordem alfabetica importa: eh a mesma que o Keras usa para atribuir
    os indices 0, 1, 2 -- e esses indices precisam corresponder ao vetor
    CLASSES[] do firmware.
    """
    if not os.path.isdir(pasta_entrada):
        print(f"[ERRO] Pasta de entrada nao encontrada: {pasta_entrada}")
        sys.exit(1)

    classes = sorted(
        d for d in os.listdir(pasta_entrada)
        if os.path.isdir(os.path.join(pasta_entrada, d))
    )
    if not classes:
        print(f"[ERRO] Nenhuma subpasta encontrada em {pasta_entrada}")
        sys.exit(1)
    return classes


def listar_imagens(pasta_classe: str) -> list[str]:
    return sorted(
        arq for arq in os.listdir(pasta_classe)
        if arq.lower().endswith(EXTENSOES_VALIDAS)
    )


def dividir(
    pasta_entrada: str,
    pasta_saida: str,
    prop_treino: float,
    prop_val: float,
    semente: int,
) -> None:
    classes = listar_classes(pasta_entrada)
    print(f"\nClasses encontradas (ordem alfabetica): {classes}")
    print("Confira se essa ordem corresponde ao vetor CLASSES[] do firmware.\n")

    rng = random.Random(semente)
    resumo = []

    for nome_classe in classes:
        origem = os.path.join(pasta_entrada, nome_classe)
        imagens = listar_imagens(origem)

        if not imagens:
            print(f"[AVISO] Nenhuma imagem em '{nome_classe}', pulando.")
            continue

        # Embaralha com semente fixa: mesma entrada -> mesma divisao.
        rng.shuffle(imagens)

        total = len(imagens)
        n_treino = int(total * prop_treino)
        n_val = int(total * prop_val)
        # O teste recebe o resto, garantindo que nenhuma imagem se perca
        # por arredondamento.
        particoes = {
            "train": imagens[:n_treino],
            "val": imagens[n_treino:n_treino + n_val],
            "test": imagens[n_treino + n_val:],
        }

        for nome_particao, arquivos in particoes.items():
            destino = os.path.join(pasta_saida, nome_particao, nome_classe)
            os.makedirs(destino, exist_ok=True)
            for arq in arquivos:
                shutil.copy2(
                    os.path.join(origem, arq),
                    os.path.join(destino, arq),
                )

        resumo.append({
            "classe": nome_classe,
            "total": total,
            "train": len(particoes["train"]),
            "val": len(particoes["val"]),
            "test": len(particoes["test"]),
        })
        print(f"  {nome_classe:<12} total={total:<6} "
              f"train={len(particoes['train']):<6} "
              f"val={len(particoes['val']):<5} "
              f"test={len(particoes['test'])}")

    # ---- Resumo final ----
    tot = {k: sum(r[k] for r in resumo) for k in ("total", "train", "val", "test")}
    print("\n" + "-" * 52)
    print(f"  {'TOTAL':<12} total={tot['total']:<6} "
          f"train={tot['train']:<6} val={tot['val']:<5} test={tot['test']}")
    print("-" * 52)
    print(f"\nProporcao real: "
          f"{tot['train'] / tot['total'] * 100:.1f}% / "
          f"{tot['val'] / tot['total'] * 100:.1f}% / "
          f"{tot['test'] / tot['total'] * 100:.1f}%")
    print(f"Saida gerada em: {pasta_saida}")
    print(f"Semente usada: {semente} (guarde este valor para reproduzir)")

    # Grava o resumo em disco, para citar no artigo sem depender da memoria.
    caminho_resumo = os.path.join(pasta_saida, "divisao_resumo.txt")
    with open(caminho_resumo, "w", encoding="utf-8") as f:
        f.write(f"Divisao do dataset (semente={semente})\n")
        f.write(f"Proporcao alvo: treino={prop_treino:.2f} "
                f"val={prop_val:.2f} teste={1 - prop_treino - prop_val:.2f}\n\n")
        f.write(f"{'classe':<14}{'total':>8}{'train':>8}{'val':>8}{'test':>8}\n")
        for r in resumo:
            f.write(f"{r['classe']:<14}{r['total']:>8}{r['train']:>8}"
                    f"{r['val']:>8}{r['test']:>8}\n")
        f.write(f"{'TOTAL':<14}{tot['total']:>8}{tot['train']:>8}"
                f"{tot['val']:>8}{tot['test']:>8}\n")
    print(f"Resumo salvo em : {caminho_resumo}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Divide um dataset por classe em treino/validacao/teste."
    )
    parser.add_argument("--input", required=True,
                        help="Pasta com as subpastas de classe (dataset original).")
    parser.add_argument("--output", required=True,
                        help="Pasta onde a estrutura dividida sera criada.")
    parser.add_argument("--train", type=float, default=0.70,
                        help="Proporcao de treino (padrao: 0.70)")
    parser.add_argument("--val", type=float, default=0.15,
                        help="Proporcao de validacao (padrao: 0.15)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semente para a divisao (padrao: 42)")
    args = parser.parse_args()

    if args.train + args.val >= 1.0:
        print("[ERRO] train + val deve ser menor que 1.0 "
              "(o restante fica para o teste).")
        sys.exit(1)

    if os.path.exists(args.output) and os.listdir(args.output):
        print(f"[AVISO] A pasta de saida '{args.output}' nao esta vazia.")
        resposta = input("        Continuar e possivelmente sobrescrever? (s/N) ")
        if resposta.strip().lower() != "s":
            print("        Cancelado.")
            sys.exit(0)

    dividir(args.input, args.output, args.train, args.val, args.seed)


if __name__ == "__main__":
    main()
