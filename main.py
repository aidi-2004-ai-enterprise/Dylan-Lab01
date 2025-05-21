from palmerpenguins import load_penguins
import pandas as pd

def main():
    # Load the penguins dataset
    penguins = load_penguins()
    
    # penguins is a pandas DataFrame
    print("Penguins dataset loaded successfully!")
    print(penguins.head())  # show first 5 rows

if __name__ == "__main__":
    main()
