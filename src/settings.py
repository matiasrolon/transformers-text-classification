# PATHS
PATH_FOLDER_DATA = "../dataset/multilabel-emotions-clasifications"
PATH_TEST = PATH_FOLDER_DATA+ "/test.tsv"
PATH_TRAIN = PATH_FOLDER_DATA + "/train.tsv"
PATH_EMOTIONS = PATH_FOLDER_DATA + "/emotions.txt"
PATH_PREPROCESSING_DATA = './results/preprocessEmotions.csv'

# BERT
BERT_MODEL_NAME = 'bert-base-cased'
MAX_TOKEN_COUNT = 1024
N_EPOCHS = 10
BATCH_SIZE = 12

# OTHERS
PREPROCESS_DATA = False
LABEL_COLUMNS = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride',
    'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']