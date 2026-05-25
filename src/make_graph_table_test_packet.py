from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "data" / "split_label_train_V3.snappy.parquet"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "graph_table_test_data"


def stable_entity_split(entity_id: str) -> str:
    """Повторяет разбиение обучения: один клиент целиком попадает в одну выборку."""
    bucket = int(hashlib.md5(str(entity_id).encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "valid"
    return "test"


def select_assignment_evaluation_profiles(
    eligible: pd.DataFrame,
    n_profiles: int,
    existing_share: float,
    random_state: int,
) -> tuple[set[str], dict[str, int | str | float]]:
    """Формирует тестовый пакет с известными случаями существующих и новых клиентов.

    Существующий клиент: в пакет отдаём один из его повторных профилей, а минимум
    один профиль оставляем в истории. Новый клиент: берём клиента с единственным
    профилем, поэтому совпадающего клиента в истории нет.
    """
    if not 0.0 <= existing_share <= 1.0:
        raise ValueError("--existing-share должен находиться в диапазоне 0..1.")

    profile_entity = eligible[["profile_id", "entity_id"]].dropna().drop_duplicates().copy()
    profile_entity["profile_id"] = profile_entity["profile_id"].astype(str)
    profile_entity["entity_id"] = profile_entity["entity_id"].astype(str)
    rng = random_state

    existing_candidates: list[str] = []
    for _, group in profile_entity.groupby("entity_id", sort=False):
        profiles = group["profile_id"].sample(frac=1.0, random_state=rng).tolist()
        if len(profiles) >= 2:
            # Первый профиль клиента остаётся в истории как правильная цель поиска.
            existing_candidates.extend(profiles[1:])

    entity_sizes = profile_entity.groupby("entity_id")["profile_id"].nunique()
    singleton_entities = set(entity_sizes[entity_sizes.eq(1)].index)
    new_candidates = profile_entity.loc[
        profile_entity["entity_id"].isin(singleton_entities), "profile_id"
    ].tolist()

    max_controlled_profiles = len(existing_candidates) + len(new_candidates)
    if n_profiles > max_controlled_profiles:
        raise ValueError(
            f"Для проверки назначения доступно максимум {max_controlled_profiles:,} профилей: "
            f"{len(existing_candidates):,} существующих и {len(new_candidates):,} новых."
        )

    requested_existing = round(n_profiles * existing_share)
    n_existing = min(requested_existing, len(existing_candidates))
    n_new = n_profiles - n_existing
    if n_new > len(new_candidates):
        n_new = len(new_candidates)
        n_existing = n_profiles - n_new
    if n_existing > len(existing_candidates):
        raise ValueError("Недостаточно профилей существующих клиентов для указанного размера пакета.")

    existing_selected = set(pd.Series(existing_candidates).sample(n=n_existing, random_state=random_state))
    new_selected = set(pd.Series(new_candidates).sample(n=n_new, random_state=random_state + 1))
    return existing_selected | new_selected, {
        "requested_profiles": n_profiles,
        "requested_existing_share": existing_share,
        "existing_profiles_in_packet": n_existing,
        "new_profiles_in_packet": n_new,
        "max_existing_profiles_available": len(existing_candidates),
        "max_new_profiles_available": len(new_candidates),
        "max_controlled_profiles_available": max_controlled_profiles,
    }


def make_test_packet(
    input_path: Path,
    out_dir: Path,
    n_profiles: int,
    random_state: int,
    entity_split: str,
    existing_share: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"make_graph_table_test_packet_{datetime.now():%Y%m%d_%H%M%S}.log"
    def log(message: str) -> None:
        line = f"[{datetime.now():%H:%M:%S}] {message}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"log_path={log_path}")
    log(f"читаем исходные данные={input_path}")
    df = pd.read_parquet(input_path)
    log(f"raw rows={len(df):,} columns={df.shape[1]:,}")
    if "profile_id" not in df.columns:
        raise ValueError("Входной parquet должен содержать profile_id.")

    eligible = df
    if entity_split != "all":
        if "entity_id" not in df.columns:
            raise ValueError("--entity-split требует колонку entity_id во входном parquet.")
        eligible = df[df["entity_id"].astype(str).map(stable_entity_split).eq(entity_split)].copy()
        log(f"entity_split={entity_split} eligible_profiles={eligible['profile_id'].nunique():,}")

    sampled_profile_ids, selection_summary = select_assignment_evaluation_profiles(
        eligible,
        n_profiles=n_profiles,
        existing_share=existing_share,
        random_state=random_state,
    )
    packet = eligible[eligible["profile_id"].astype(str).isin(sampled_profile_ids)].copy()
    log(
        f"sampled_profiles={len(sampled_profile_ids):,} "
        f"packet_rows={len(packet):,}"
    )

    suffix = f"_{entity_split}" if entity_split != "all" else ""
    packet_path = out_dir / f"raw_packet{suffix}_{n_profiles}_profiles.parquet"
    summary_path = out_dir / f"raw_packet{suffix}_{n_profiles}_summary.json"

    packet.to_parquet(packet_path, index=False)

    summary = {
        "source": str(input_path),
        "packet_path": str(packet_path),
        "rows": int(len(packet)),
        "profiles": int(packet["profile_id"].nunique()),
        "entity_id_available": "entity_id" in packet.columns,
        "entities": int(packet["entity_id"].nunique()) if "entity_id" in packet.columns else None,
        "entity_split": entity_split,
        "random_state": int(random_state),
        **selection_summary,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    log(f"packet_path={packet_path}")
    log(f"summary_path={summary_path}")
    log("done")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Создать размеченный пакет профилей для проверки graph-table inference.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-profiles", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--entity-split",
        choices=["train", "valid", "test", "all"],
        default="test",
        help="Для финальной проверки используйте test: эти клиенты не участвовали в обучении и выборе XGBoost.",
    )
    parser.add_argument(
        "--existing-share",
        type=float,
        default=0.25,
        help="Доля профилей существующих клиентов в тестовом пакете; остальные - новые клиенты.",
    )
    args = parser.parse_args()
    make_test_packet(
        args.input,
        args.out_dir,
        args.n_profiles,
        args.random_state,
        args.entity_split,
        args.existing_share,
    )


if __name__ == "__main__":
    main()
