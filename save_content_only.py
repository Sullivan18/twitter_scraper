import pandas as pd
import sys

def save_content_only(input_csv, output_csv):
    # Carrega o CSV original
    df = pd.read_csv(input_csv)
    
    # Seleciona apenas a coluna "Content"
    content_df = df[['Content']]
    
    # Salva a nova DataFrame em um novo CSV
    content_df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python save_content_only.py <input_csv> <output_csv>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    
    save_content_only(input_csv, output_csv)
    print(f"CSV com conteúdo apenas salvo em: {output_csv}")
