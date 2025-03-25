# Fantasy Football

A repository with my attempts to automatically select my fantasy football team for me. The rough aim is to:

* Use historic player data alongside points scored data to build a model that can predict future points scored.
* Predict points for future game weeks
* Run an optimisation algorithm to select the best team for the upcoming game week(s)

## Installation

Firstly clone this repository to your local machine. Then navigate to the root of the repository. For this project, we're using python 3.12. I'd suggest using pyenv to manage your python versions. Once you have python 3.12 as the local version, you'll also need to make sure you have poetry installed to manage virtual environments. If you don't already, you can install it following the instructions here.

You'll also need to have a .env file with your github API key inside it. To get a github API key, follow the instructions [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens). Once you have your API key, create a .env file in the root of the repository and add the following line:

```bash
GITHUB_API_KEY=<your_github_api_key>
```

Once all this is done, you can install the dependencies for this project by running the following command:

```bash
poetry install
```

When this command has run, you can then activate the virtual environment by running:

```bash
poetry shell
```

This will activate the virtual environment, and you can run the main script by running:

```bash
python main.py
```

## Resources

For the data I've heavily leaned on the excellent Fantasy Premier League repository by [Vaastav Anand](https://github.com/vaastav/Fantasy-Premier-League). This repository contains a wealth of data on the Fantasy Premier League, including player data, fixture data, and points scored data.

## Current State

This is very early state, and the code is very much a work in progress. The current state of the code is as follows:

* `main.py` this script will download all the csv data within the `data` of the Fantasy Premier League repository and save it on your local filesystem using the same structure as the repository itself.
* `fantasy_football/data_extraction.py` contains functions for downloading data from the FPL API.
* `fantasy_football/data_transformation.py` contains functions for transforming the raw data into useful formats.
* `fantasy_football/fpl.py` contains functions for interacting with the FPL API to get player data, team information, and fixtures.
* `fantasy_football/prediction.py` contains a simple prediction model for player points.
* `fantasy_football/optimization.py` contains functions for optimizing team selection using linear programming.

## Usage

### Downloading Data

```bash
# Download all historical data files
python main.py --download-all

# Update only the current season data
python main.py --update-current-season
```

### Team Optimization

You can use the optimization feature to generate the optimal team for a given gameweek:

```bash
# Optimize team for gameweek 5 with 1 free transfer
python main.py --optimize-team 5 --free-transfers 1

# Optimize team for gameweek 5 with 2 free transfers
python main.py --optimize-team 5 --free-transfers 2
```

The optimization algorithm will:
1. Use your current team as a starting point
2. Consider transfer options to maximize expected points
3. Select the best 15-player squad
4. Choose the optimal starting 11
5. Select captain and vice-captain
6. Account for the penalty for extra transfers beyond your free allowance

The output will show:
- Recommended transfers (if any)
- The full optimized 15-player squad
- The suggested starting 11 (marked with *)
- Captain (C) and vice-captain (V) selections
- Expected points for the gameweek
