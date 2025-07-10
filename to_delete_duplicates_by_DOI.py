import pandas as pd
import os

def remove_duplicates_by_doi(input_file: str = "combined_dataset.xlsx",
                             output_file: str = "processed_dataset.xlsx"):
    """
    Reads combined_dataset.xlsx, drops duplicate DOIs, 
    and writes the cleaned DataFrame to processed_dataset.xlsx.
    """
    # make sure input exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # load
    df = pd.read_excel(input_file)
    if "DOI" not in df.columns:
        raise ValueError("Column 'DOI' not found in input file.")

    # drop duplicates
    df_clean = df.drop_duplicates(subset="DOI", keep="first")

    # write out
    df_clean.to_excel(output_file, index=False)
    print(f"✅  Duplicates removed. Clean file written to: {output_file}")


if __name__ == "__main__":
    remove_duplicates_by_doi()
