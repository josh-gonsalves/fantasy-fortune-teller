import nflreadpy as nfl
import numpy as np
import polars as pl

# Columns from the dataset that we are interested in
PLAYER_COLS = [
            "player_id",            # str
            "player_display_name",  # str
            "position",             # str
            "team",                 # str
            "opponent_team",        # str
            "week",                 # int
            "season",               # int
            "fantasy_points_ppr"    # float
            ]
GAME_COLS = [
            "game_id",              # str
            "week",                 # int
            "season",               # int
            "home_team",            # str
            "away_team",            # str
            "home_rest",            # int
            "away_rest",            # int
            "roof",                 # str
            "surface",              # str
            "temp",                 # int
            "wind",                 # int
            "div_game"              # int
            ]

"""
    load_player_data(): Returns a polars.Dataframe object
        Pulls per-game player data across all available seasons
        Pulls per-game conditions and team data

"""
def load_player_data(position: str, start=1999, end=2025):
    # Initially load data into two dataframes
    player_df = nfl.load_player_stats(np.arange(start, end + 1))
    game_df = nfl.load_schedules(np.arange(start, end + 1))

    # Filter both by regular season; no playoff games are included
    player_df = (
        player_df
        .filter((pl.col("season_type") == "REG") & (pl.col("position") == position))
        .select(PLAYER_COLS)
    )
    game_df = (
        game_df
        .filter(pl.col("game_type") == "REG")
        .select(GAME_COLS)
    )

    # Match up data where the game is the same
    home_join = join(player_df, game_df, "home")
    away_join = join(player_df, game_df, "away")

    # Combine the two tables and ensure that the game exists
    # Creates what is in essence an inner join
    df = (
        pl.concat([home_join, away_join])
        .filter(pl.col("game_id").is_not_null())
    )

    return df

"""
    add_rolling_mean(): Adds a rolling mean of whatever size is necessary
        Initially sorts the data chronologically by player
        window_size can be whatever length
        ***any null values still exist
"""
def add_rolling_mean(df: pl.DataFrame, window_size: int, mean_col: str, col_alias: str):
    df = df.sort(["player_id", "season", "week"])
    return (
        df
        .with_columns(
            pl.col(mean_col)
            .shift(1)
            .rolling_mean(window_size, min_periods=1)
            .over("player_id")
            .alias(col_alias)
        )
    )

def encode_categoricals(df: pl.DataFrame):
    pass

"""
----------------------------------------------------------------
| Helper Functions                                             |
----------------------------------------------------------------
"""

"""
    join(): Inner join on game, week, and season
    - Because the left join replaces "gtype_team" with "team",
        it is necessary to also drop the opposite (away for h-
        ome and vice versa) "gtype_team" column. The game type
        is encoded within the "is_home" attribute
"""
def join(leftdf: pl.DataFrame, rightdf: pl.DataFrame, gtype: str, join: str = "left"):
    # Initialize values for use later
    val = 0
    team = "home_team"
    if gtype == "home":
        val = 1
        team = "away_team"
    
    # Return computed join with additional and dropped columns 
    return (    
        leftdf
        .join(
            rightdf,
            left_on = ["team", "week", "season"],
            right_on = [gtype + "_team", "week", "season"],
            how = join
        )
        .with_columns([
            pl.lit(val).alias("is_home"),
            pl.col(gtype + "_rest").alias("team_rest"),
        ])
        .drop([team])
        
    )
        

