import argparse
import os
import pandas as pd
import pathlib
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import tqdm


from dataset_orm import *

DB_FILE = 'wordbank.db'
DB_PATH = pathlib.Path(__file__).parent.parent.absolute() / 'data' / DB_FILE
engine = create_engine(f'sqlite:///{DB_PATH}')
Session = sessionmaker(bind=engine)

parser = argparse.ArgumentParser(description='ORM helper')
parser.add_argument('--create-tables', action='store_true')
parser.add_argument('--load-words', action='store_true')
WORDBANK_WORDS_FILE = os.path.join(CURRENT_DIR, 'worbank_with_category.tsv')
parser.add_argument('--words-file', default=WORDBANK_WORDS_FILE)
parser.add_argument('--load-babi', action='store_true')
BABI_FILE = os.path.join(CURRENT_DIR, 'babi_line_wordbank_cleaned_data.tsv')
parser.add_argument('--babi-file', default=BABI_FILE)
parser.add_argument('--load-childes', action='store_true')
CHILDES_FILE = os.path.join(CURRENT_DIR, 'childes_wordbank_cleaned_data.tsv')
parser.add_argument('--childes-file', default=CHILDES_FILE)


def create_tables():
    Base.metadata.create_all(engine)


def load_wordbank_words(file_path=WORDBANK_WORDS_FILE):
    session = Session()
    words_df = pd.read_csv(file_path, sep='\t')
    for idx, (word_text, category_name) in tqdm.tqdm(words_df.iterrows(), total=len(words_df)):
        word_text = word_text.strip().lower()
        category_name = category_name.strip().lower()

        word_query = session.query(WordbankWord).filter(WordbankWord.word == word_text)
        word = word_query.one_or_none()

        if word is not None:
            if word.category.name != category_name:
                print(f'For {word} we currently have category {word.category}, but found category name {category_name}...')
                continue

        category_query = session.query(WordbankCategory).filter(WordbankCategory.name == category_name)
        category = category_query.one_or_none()

        if category is None:
            category = WordbankCategory(name=category_name)
            session.add(category)

        word = WordbankWord(word=word_text, category=category)
        session.add(word)
    
    session.commit()


BABI_DATASET_NAME = 'bAbI'


def load_babi_data(file_path=BABI_FILE):
    session = Session()
    babi_df = pd.read_csv(file_path, sep='\t')

    original_dataset = session.query(OriginalDataset).filter(OriginalDataset.name == BABI_DATASET_NAME).one_or_none()
    if original_dataset is None:
        original_dataset = OriginalDataset(name=BABI_DATASET_NAME)
        session.add(original_dataset)

    babi_sentence_type = session.query(SentenceType).filter(SentenceType.type_name == BABI_DATASET_NAME).one_or_none()
    if babi_sentence_type is None:
        babi_sentence_type = SentenceType(type_name=BABI_DATASET_NAME)
        session.add(babi_sentence_type)

    for idx, (sentence_text, matches, _, _, _, _) in tqdm.tqdm(babi_df.iterrows(), total=len(babi_df)):
        sentence_text = sentence_text.strip().lower()
        wordbank_words = [word.strip() for word in matches.lower().split(',')]

        sentence_query = session.query(Sentence).filter(Sentence.text == sentence_text).filter(Sentence.original_dataset == original_dataset)
        sentence = sentence_query.one_or_none()

        if sentence is not None:
            for sentence_word in sentence.wordbank_words:
                if not any([sentence_word.word == w for w in wordbank_words]):
                    print(f'Sentence "{sentence_text}" (idx {idx}) found in DB (id {sentence.id}) words {[wbw.word for wbw in sentence.wordbank_words]} but in file with words {wordbank_words}...')

            for file_word in wordbank_words:
                if not any([file_word == w.word for w in sentence.wordbank_words]):
                    print(f'Sentence "{sentence_text}" (idx {idx}) found in DB (id {sentence.id}) with words {[wbw.word for wbw in sentence.wordbank_words]} but in file with words {wordbank_words}...')

            continue
            
        for word_text in wordbank_words:
            word_query = session.query(WordbankWord).filter(WordbankWord.word == word_text)
            word = word_query.one_or_none()
            if word is None:
                raise ValueError(f'Sentence "{sentence_text}" (idx {idx}), found match word "{word_text}" that does not exist in the DB. Aborting...')

        word_objects = session.query(WordbankWord).filter(WordbankWord.word.in_(wordbank_words)).all()
        sentence = Sentence(original_dataset=original_dataset, text=sentence_text, 
                            wordbank_words=word_objects, sentence_type=babi_sentence_type, 
                            target_child_age=None)
        session.add(sentence)

    session.commit()


CHILDES_DATASET_NAME = 'CHILDES'


def load_childes_data(file_path=CHILDES_FILE):
    session = Session()
    childes_df = pd.read_csv(file_path, sep='\t')

    original_dataset = session.query(OriginalDataset).filter(OriginalDataset.name == CHILDES_DATASET_NAME).one_or_none()
    if original_dataset is None:
        original_dataset = OriginalDataset(name=CHILDES_DATASET_NAME)
        session.add(original_dataset)

    for idx, (sentence_text, matches, _, _, _, target_child_age, sentence_type_name) in tqdm.tqdm(childes_df.iterrows(), total=len(childes_df)):
        sentence_text = sentence_text.strip().lower()
        wordbank_words = [word.strip() for word in matches.lower().split(',')]
        sentence_type_name = sentence_type_name.strip().lower()

        sentence_query = session.query(Sentence).filter(Sentence.text == sentence_text).filter(Sentence.original_dataset == original_dataset)
        sentence = sentence_query.one_or_none()

        if sentence is not None:
            for sentence_word in sentence.wordbank_words:
                if not any([sentence_word.word == w for w in wordbank_words]):
                    print(f'Sentence "{sentence_text}" (idx {idx}) found in DB (id {sentence.id}) words {[wbw.word for wbw in sentence.wordbank_words]} but in file with words {wordbank_words}')

            for file_word in wordbank_words:
                if not any([file_word == w.word for w in sentence.wordbank_words]):
                    print(f'Sentence "{sentence_text}" (idx {idx}) found in DB (id {sentence.id}) with words {[wbw.word for wbw in sentence.wordbank_words]} but in file with words {wordbank_words}')

            if sentence.sentence_type.type_name != sentence_type_name:
                print(f'Sentence "{sentence_text}" (idx {idx}) found in DB (id {sentence.id}) with type name {sentence.sentence_type.type_name} but in file with type name {sentence_type_name}')

            continue
            
        for word_text in wordbank_words:
            word_query = session.query(WordbankWord).filter(WordbankWord.word == word_text)
            word = word_query.one_or_none()
            if word is None:
                raise ValueError(f'Sentence "{sentence_text}" (idx {idx}), found match word "{word_text}" that does not exist in the DB. Aborting...')

        word_objects = session.query(WordbankWord).filter(WordbankWord.word.in_(wordbank_words)).all()
        sentence_type = session.query(SentenceType).filter(SentenceType.type_name == sentence_type_name).one_or_none()
        if sentence_type is None:
            sentence_type = SentenceType(type_name=sentence_type_name)
            session.add(sentence_type)
        
        sentence = Sentence(original_dataset=original_dataset, text=sentence_text, 
                            wordbank_words=word_objects, sentence_type=sentence_type, 
                            target_child_age=target_child_age)
        session.add(sentence)

    session.commit()


if __name__ == '__main__':
    args = parser.parse_args()

    if args.create_tables:
        create_tables()

    if args.load_words:
        load_wordbank_words(args.words_file)

    if args.load_babi:
        load_babi_data(args.babi_file)

    if args.load_childes:
        load_childes_data(args.childes_file)
