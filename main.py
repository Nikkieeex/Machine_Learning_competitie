from __init__ import data_reader
from eda import distribution

def main():
    data = data_reader("data/data-studenten.csv")
    pdf_export_dir = "eda_plots"
    distribution(data, pdf_export_dir)
    return

if __name__ == "__main__":
   main()
