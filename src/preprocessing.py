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
            "temp",                 # int
            "wind",                 # int
            "div_game"              # int
            ]
OPP_COLS = [
            "def_tackles_for_loss",  # int
            "def_fumbles_forced",    # int
            "def_sacks",             # int
            "def_qb_hits",           # int
            "def_interceptions",     # int
            "def_pass_defended",     # int
            "team",                  # str
            "week",                  # int
            "season"                 # int
]

# Features and target
FEATURE_COLS = [
            "season",
            "home_rest",
            "away_rest",
            "temp",
            "wind",
            "div_game",
            "is_home",
            "team_rest",
            "ppr_avg_2",
            "ppr_avg_8",
            "def_sacks_avg_10",
            "def_qb_hits_avg_10",
            "def_interceptions_avg_10",
            "def_pass_defended_avg_10",
            "def_fumbles_forced_avg_10",
            "def_tackles_for_loss_avg_10"
]

TARGET_COL = "fantasy_points_ppr"

"""
    load_player_data(): Returns a polars.Dataframe object
        Pulls per-game player data across all available seasons
        Pulls per-game conditions and team data

"""
def load_player_data(position: str, start=1999, end=2025):
    # Initially load data into three dataframes
    player_df = nfl.load_player_stats(np.arange(start, end + 1))
    game_df = nfl.load_schedules(np.arange(start, end + 1))
    opp_df = nfl.load_team_stats(np.arange(start, end + 1))


    # Filter all by regular season; no playoff games are included
    player_df = (
        player_df
        .filter((pl.col("season_type") == "REG") 
                & (pl.col("position") == position) 
                & (pl.col("fantasy_points_ppr").is_not_null())
                & (pl.col("fantasy_points_ppr").is_not_nan()))
        .select(PLAYER_COLS)
    )
    game_df = (
        game_df
        .filter(pl.col("game_type") == "REG")
        .select(GAME_COLS)
    )
    opp_df = (
        opp_df
        .filter((pl.col("season_type") == "REG"))
        .select(OPP_COLS)
    )

    # Add rolling means for defensive stats
    # These have large window sizes to take into account season averages
    opp_df = opp_df.sort(["team", "season", "week"])
    for stat in ["def_sacks","def_qb_hits","def_interceptions","def_pass_defended", "def_fumbles_forced", "def_tackles_for_loss"]:
        opp_df = add_rolling_mean_def(opp_df, 10, stat, f"{stat}_avg_10")

    # Match up data where the game is the same
    home_join = join(player_df, game_df, "home")
    away_join = join(player_df, game_df, "away")

    # Combine the two tables and ensure that the game exists
    # Creates what is in essence an inner join
    df = (
        pl.concat([home_join, away_join])
        .filter(pl.col("game_id").is_not_null())
    )

    # Add rolling means for player points
    # One mean has a window of 8 games and the other has a window of 2
    # This gives two emphases on a player's long and short term performance
    df = add_rolling_mean_player(df, 8, "fantasy_points_ppr", "ppr_avg_8")
    df = add_rolling_mean_player(df, 2, "fantasy_points_ppr", "ppr_avg_2")

    df = df.join(
            opp_df,
            left_on = ["opponent_team", "week", "season"],
            right_on = ["team", "week", "season"],
            how = "left"
        )
    
    # Handle nulls and encode categoricals
    df = handle_nulls(df)
    df = encode_categoricals(df)
    df = df.select(FEATURE_COLS + [TARGET_COL])

    return df

"""
    add_rolling_mean_player(): Adds a rolling mean of whatever size is necessary for a player's stat
        Initially sorts the data chronologically by player
        window_size can be whatever length
        ***any null values still exist
"""
def add_rolling_mean_player(df: pl.DataFrame, window_size: int, mean_col: str, col_alias: str):
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


"""
    add_rolling_mean_def(): Adds a rolling mean of whatever size is necessary for a team's stat
        Initially sorts the data chronologically by team
        window_size can be whatever length
        ***any null values still exist
"""
def add_rolling_mean_def(df: pl.DataFrame, window_size: int, mean_col: str, col_alias: str):
    df = df.sort(["team", "season", "week"])
    return (
        df
        .with_columns(
            pl.col(mean_col)
            .shift(1)
            .rolling_mean(window_size, min_periods=1)
            .over(["team", "season"])
            .alias(col_alias)
        )
    )


"""
    encode_categoricals(): Cleans up data so that no strs exist in the data
        The two different encoding options are one hots and ordinals
"""
def encode_categoricals(df: pl.DataFrame):

    # The var 'position' ends up being dropped at the moment but would be useful for future projects
    one_hots = ["position"]
    ordinals = ["team", "opponent_team"]

    # Encoding
    df = df.with_columns([
        pl.col(col).cast(pl.Categorical) for col in one_hots
    ]).to_dummies(columns=one_hots)
    df = df.with_columns([
        pl.col(col)
        .cast(pl.Categorical)
        .to_physical()
        .alias(f"{col}_id")
        for col in ordinals
    ]).drop(ordinals)

    return df

"""
    handle_nulls(): Ensures that there are no null values in the data
        Player averages like 'ppr_avg_2' are populated with 0's
        Individual game stats like 'temp' are populated with the median
"""
def handle_nulls(df: pl.DataFrame):
    # Columns to be zeroed
    zero_cols = [
                "ppr_avg_2",
                "ppr_avg_8",
                "def_sacks_avg_10",
                "def_qb_hits_avg_10",
                "def_interceptions_avg_10",
                "def_pass_defended_avg_10",
                "def_fumbles_forced_avg_10", 
                "def_tackles_for_loss_avg_10"]
    df = df.with_columns([
        pl.col(c).fill_null(0) for c in zero_cols
    ])
    # Columns to be median-ed
    median_cols = ["temp", "wind", "team_rest"]
    df = df.with_columns([
        pl.col(c).fill_null(pl.col(c).median()) for c in median_cols
    ])

    return df



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
        

