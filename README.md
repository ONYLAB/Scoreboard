# Scoreboard for COVID-19 Epidemiological Models Deposited to Covid19 Forecast Hub 

Epidemiological models of COVID-19 have proliferated quickly. It is important to understand how well these models estimate the true values of key underlying variables: number infected, recovered, and died. Some models may predict well for short term but not well for long term. Yet appreciating the model predictive performance is essential to inform public health decisions including those that manage the delicate tradeoffs between the economy and public health. Here we developed a scoring system to assess the predictions of COVID-19 epidemiological models. We use model forecast cumulative distributions uploaded to the COVID-19 Forecast Hub by model developers. Our score computes the log likelihood for the model forecasts by way of comparing those to the observed COVID-19 positive cases and deaths. Scores are updated continuously as new data become available. Our scoring workflow keeps track of how models improve (or degrade) over time assisting public health policy makers and other decision makers understand the model uncertainty and thinking about the potential impact of these uncertainties while making the decision. Additionally, publicly available information on model performance can aid in further developing modeling frameworks, as well as inform policy makers as to which modeling paradigms are critical to consider.

### Requirements to run code here:
- python 3.5+
- Install the Python package:
  - From source install the scoreboard:
    - `git clone https://github.com/ONYLAB/Scoreboard`
    - `cd Scoreboard`
    - `pip install -e .  # This installs the covid-sicr package from source`
  - From source downlaod covid19 forecast hub:
    - `git clone https://github.com/reichlab/covid19-forecast-hub/`

### Important Notebook:
- New data can be downloaded and score can be obtained by running `Notebooks/RunScores.ipynb`:

This code is open source under the MIT License.
Correspondence on modeling and the code should be directed to carsonc at nih dot gov or osman dot yogurtcu at fda dot hhs dot gov.
