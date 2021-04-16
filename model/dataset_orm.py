from sqlalchemy import Column, Integer, Float, String, ForeignKey, Table
from sqlalchemy.orm import declarative_base, relationship


Base = declarative_base()


class WordbankCategory(Base):
    __tablename__ = 'wordbank_categories'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)

    def __repr__(self):
        return f'<WordbankCategory(id={self.id}, name="{self.name}">'

    def __str_(self):
        return self.name


class OriginalDataset(Base):
    __tablename__ = 'original_datasets'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)

    def __repr__(self):
        return f'<OriginalDataset(id={self.id}, name="{self.name}">'

    def __str__(self):
        return self.name


words_sentences = Table('words_sentences', Base.metadata,
    Column('word_id', ForeignKey('wordbank_words.id'), primary_key=True),
    Column('sentence_id', ForeignKey('sentences.id'), primary_key=True),
)


class WordbankWord(Base):
    __tablename__ = 'wordbank_words'

    id = Column(Integer, primary_key=True)
    word = Column(String, nullable=False)

    category_id = Column(Integer, ForeignKey('wordbank_categories.id'), nullable=False)
    category = relationship('WordbankCategory', back_populates='words')

    sentences = relationship('Sentence', secondary=words_sentences, back_populates='wordbank_words')

    def __repr__(self):
        return f'<WordbankWord(id={self.id}, name="{self.name}", category="{self.category}">'

    def __str__(self):
        return f'{self.word} ({self.category})'


WordbankCategory.words = relationship('WordbankWord', order_by=WordbankWord.id, back_populates='category')


class SentenceType(Base):
    __tablename__ = 'sentence_types'

    id = Column(Integer, primary_key=True)
    type_name = Column(String, nullable=False)

    def __repr__(self):
        return f'<SentenceType(id={self.id}, type_name="{self.type_name}">'

    def __str__(self):
        return self.type_name


class Sentence(Base):
    __tablename__ = 'sentences'

    id = Column(Integer, primary_key=True)
    text = Column(String)

    original_dataset_id = Column(Integer, ForeignKey('original_datasets.id'))
    original_dataset = relationship('OriginalDataset', back_populates='sentences')

    wordbank_words = relationship('WordbankWord', secondary=words_sentences, back_populates='sentences')

    sentence_type_id = Column(Integer, ForeignKey('sentence_types.id'))
    sentence_type = relationship('SentenceType', back_populates='sentences')

    target_child_age = Column(Float)

    def __repr__(self):
        return f'<Sentence(id={self.id}, original_dataset="{self.original_dataset}", text="{self.text}">'

    def __str__(self):
        return f'{self.text} ({self.original_dataset})'


OriginalDataset.sentences = relationship('Sentence', order_by=Sentence.id, back_populates='original_dataset')
SentenceType.sentences = relationship('Sentence', order_by=Sentence.id, back_populates='sentence_type')


