from src.preprocessing import *
from src.regression import *
from src.tree import *





if __name__ == "__main__":
    df = load_player_data("QB")
    train_regression(df)
    train_tree_gridsearch(df)

    