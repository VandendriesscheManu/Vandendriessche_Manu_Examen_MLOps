import pandas as pd
from pathlib import Path

input_path = Path("data/got_persona_dataset_100.xlsx")
output_dir = Path("data_tabular")
output_dir.mkdir(exist_ok=True)

# Excel → DataFrame
df = pd.read_excel(input_path, engine="openpyxl")

# Opslaan als Parquet (tabulair & Azure-vriendelijk)
df.to_parquet(output_dir / "got_persona.parquet", index=False)

print("✅ Excel succesvol omgezet naar Parquet")
