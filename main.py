from src.preprocessing import *




if __name__ == "__main__":
    df = add_rolling_mean(load_player_data("QB"), 5, "fantasy_points_ppr", "ppr_avg_5")
    print(df, df.columns)
    