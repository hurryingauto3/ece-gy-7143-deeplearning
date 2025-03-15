from kaggle.api.kaggle_api_extended import KaggleApi

def submit_to_kaggle(filename, message):
    api = KaggleApi()
    api.authenticate()
    api.competition_submit(filename, f"{message}", 'deep-learning-spring-2025-project-1')
