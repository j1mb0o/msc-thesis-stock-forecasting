import numpy as np
import pandas as pd


def generate_latex_table(
    data,
    idx_structure,
    idx_names,
    columns,
    caption="My Table",
    label="tab:mytable",
    filename="output_table.tex",
):
    """
    Generates a LaTeX table matching the specific 'Forecast | Model | 1..10' structure.
    Uses \adjustbox to fit content to \textwidth.
    """

    # 1. Construct MultiIndex for Rows (Forecast, Model)
    index = pd.MultiIndex.from_tuples(idx_structure, names=idx_names)

    # 2. Create DataFrame
    df = pd.DataFrame(data, index=index, columns=columns)

    # 3. Generate just the tabular code using Pandas
    # We set caption=None and positioning=None to get ONLY the tabular environment
    # This allows us to wrap it in \adjustbox manually below.
    latex_tabular = df.style.format(precision=2).to_latex(
        position=None,  # Do not generate \begin{table} wrapper yet
        caption=None,  # Do not generate \caption yet
        label=None,  # Do not generate \label yet
        hrules=True,  # Adds \toprule, \midrule, \bottomrule
        multirow_align="c",  # Center vertical alignment for merged cells
        multicol_align="c",  # Center horizontal alignment
        column_format="cc"
        + "c" * len(columns),  # Force 'c' for all columns (Index + Data)
    )

    # 4. Manually construct the Table Wrapper with \adjustbox
    # This gives us exact control over the layout requested.
    full_latex = f"""\\begin{{table}}[h!]
  \\centering
  \\caption{{{caption}}}
  \\label{{{label}}}
  \\adjustbox{{max width=\\textwidth}}{{
{latex_tabular}
  }}
\\end{{table}}"""

    # 5. Save to file
    with open(filename, "w") as f:
        f.write(full_latex)

    print(f"Successfully generated table: {filename}")
    print("-" * 30)
    print(full_latex)


if __name__ == "__main__":
    # --- NEW TESTING VALUES (Simulation) ---

    # 1. Column Headers (1 to 10)
    cols = [str(i) for i in range(1, 11)]

    # 2. Row Indices (Forecast, Model)
    forecasts = ["1d", "2d", "3d"]
    models = ["Arima", "FM", "Naive"]

    idx_structure = []
    for f in forecasts:
        for m in models:
            idx_structure.append((f, m))

    # 3. Data: generating random float numbers to simulate RMSE values
    # 9 rows x 10 columns
    np.random.seed(42)  # Seed for reproducibility
    raw_data = np.random.uniform(low=0.5, high=2.5, size=(9, 10))

    # 4. Run the Generator
    generate_latex_table(
        data=raw_data,
        idx_structure=idx_structure,
        idx_names=["Forecast", "Model"],
        columns=cols,
        caption="Forecast Accuracy Metrics (RMSE)",
        label="tab:forecast_rmse_test",
        filename="thesis_table_test.tex",
    )
