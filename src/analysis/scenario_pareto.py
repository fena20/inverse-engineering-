#!/usr/bin/env python3
"""Generate a scenario-level decarbonization Pareto chart from Persian/English inputs."""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_SCENARIOS = [
    {
        "سناریو": "۱",
        "استراتژی محوری": "ارتقای سریع (EPC C)",
        "هزینه خالص (£)": "1.2k−2.5k",
        "کاهش کربن": "۱۵٪",
        "شاخص اقتصادی": "بازگشت سرمایه بسیار سریع",
    },
    {
        "سناریو": "۲",
        "استراتژی محوری": "اولویت پوسته (Solid Wall)",
        "هزینه خالص (£)": "12k−18k",
        "کاهش کربن": "۳۰-۴۰٪",
        "شاخص اقتصادی": "سرمایه‌گذاری بلندمدت/ارزش افزوده",
    },
    {
        "سناریو": "۳",
        "استراتژی محوری": "الکتریکی‌سازی (ASHP)",
        "هزینه خالص (£)": "5k (خالص)",
        "کاهش کربن": "۶۰-۷۰٪",
        "شاخص اقتصادی": "کربن‌زدایی عمیق/وابسته به یارانه",
    },
    {
        "سناریو": "۴",
        "استراتژی محوری": "مسیر متعادل (Hybrid)",
        "هزینه خالص (£)": "8k−12k",
        "کاهش کربن": "۴۵-۵۵٪",
        "شاخص اقتصادی": "بهینه از نظر عملیاتی",
    },
    {
        "سناریو": "۵",
        "استراتژی محوری": "استاندارد پیشرو (NZEB)",
        "هزینه خالص (£)": "40k−80k",
        "کاهش کربن": ">۸۵٪",
        "شاخص اقتصادی": "غیر‌اقتصادی/زیست‌محیطی محض",
    },
    {
        "سناریو": "۶",
        "استراتژی محوری": "تولیدکننده (Prosumer)",
        "هزینه خالص (£)": "12k−18k",
        "کاهش کربن": "خالص صفر",
        "شاخص اقتصادی": "استقلال از شبکه برق",
    },
]

PERSIAN_TO_EN = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")


def normalize_number_text(value: str) -> str:
    txt = str(value).translate(PERSIAN_TO_EN)
    txt = txt.replace("−", "-").replace("–", "-").replace("٪", "%")
    return txt


def parse_cost_kgbp(value: str) -> float:
    txt = normalize_number_text(value).lower().replace("£", "")
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", txt)]
    if not nums:
        raise ValueError(f"Unable to parse cost: {value}")
    if "k" in txt:
        return sum(nums) / len(nums)
    return (sum(nums) / len(nums)) / 1000.0


def parse_reduction_pct(value: str) -> float:
    txt = normalize_number_text(value).lower()
    if "خالص صفر" in str(value):
        return 100.0
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", txt)]
    if not nums:
        raise ValueError(f"Unable to parse carbon reduction: {value}")
    if txt.strip().startswith(">"):
        return nums[0]
    return sum(nums) / len(nums)


def economic_score(text: str) -> int:
    t = str(text)
    if "بسیار سریع" in t:
        return 5
    if "بهینه" in t or "استقلال" in t:
        return 4
    if "بلندمدت" in t or "وابسته" in t:
        return 3
    if "غیر" in t:
        return 1
    return 2


def load_scenarios(input_csv: Path | None) -> pd.DataFrame:
    if input_csv is None:
        return pd.DataFrame(DEFAULT_SCENARIOS)
    return pd.read_csv(input_csv)


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean["scenario_id"] = clean["سناریو"].astype(str).map(normalize_number_text)
    clean["cost_mid_kgbp"] = clean["هزینه خالص (£)"].map(parse_cost_kgbp)
    clean["carbon_reduction_pct"] = clean["کاهش کربن"].map(parse_reduction_pct)
    clean["economic_score"] = clean["شاخص اقتصادی"].map(economic_score)
    return clean.sort_values("scenario_id")


def plot_scenarios(df: pd.DataFrame, output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(13, 8))

    sizes = 120 + (df["economic_score"] * 45)
    sc = ax.scatter(
        df["cost_mid_kgbp"],
        df["carbon_reduction_pct"],
        s=sizes,
        c=df["economic_score"],
        cmap="viridis",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.95,
    )

    ax.plot(df["cost_mid_kgbp"], df["carbon_reduction_pct"], "--", color="#555", alpha=0.6)

    for _, row in df.iterrows():
        ax.annotate(
            f"S{row['scenario_id']}\n{row['استراتژی محوری']}",
            (row["cost_mid_kgbp"], row["carbon_reduction_pct"]),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=9,
        )

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Economic Attractiveness Score (1=Low, 5=High)")

    ax.set_xlabel("Net Lifecycle Cost Midpoint (£k)")
    ax.set_ylabel("Carbon Reduction (%)")
    ax.set_title("Scenario-Level Pareto Analysis for Retrofit Strategies")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 105)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate custom scenario Pareto chart.")
    parser.add_argument("--input", type=Path, default=None, help="Optional CSV with Persian columns")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/thesis_figures"),
        help="Directory to save figure and parsed table",
    )
    args = parser.parse_args()

    raw_df = load_scenarios(args.input)
    prepared = prepare_dataframe(raw_df)

    output_png = args.output_dir / "fig7_4_custom_scenario_pareto.png"
    output_csv = args.output_dir / "table7_2_custom_scenario_inputs_parsed.csv"

    plot_scenarios(prepared, output_png)
    prepared.to_csv(output_csv, index=False)

    print(f"Saved: {output_png}")
    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    main()
