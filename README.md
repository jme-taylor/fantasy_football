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
