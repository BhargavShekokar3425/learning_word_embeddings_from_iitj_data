"""
Word2Vec Model Training Module
training CBOW and Skip-gram models.
We experiment with different hyperparameters as required by the assignment.

Author: Bhargav Shekokar (B23CS1008)
Course: NLU Assignment 2
"""

import os
import json
from gensim.models import Word2Vec
import numpy as np
from tqdm import tqdm
import time


class Word2VecTrainer:
    """
    Trains Word2Vec models with various configurations.
    Handles both CBOW and Skip-gram with different hyperparameters.
    """
    
    def __init__(self, corpus_path="./data/processed/cleaned_corpus.txt"):
        """
        Initialize trainer with path to cleaned corpus.
        
        Args:
            corpus_path: Path to the preprocessed corpus file
                         (one sentence per line, words space-separated)
        """
        self.corpus_path = corpus_path
        self.sentences = self._load_corpus()
        self.models = {}  # store trained models here
        self.results = []  # store experiment results
        
        # create output directory for models
        os.makedirs("./models", exist_ok=True)
        os.makedirs("./outputs", exist_ok=True)
        
    def _load_corpus(self):
        """
        Loads the processed corpus from file.
        Returns list of sentences where each sentence is a list of words.
        """
        sentences = []
        
        print(f"Loading corpus from {self.corpus_path}...")
        
        try:
            with open(self.corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    words = line.strip().split()
                    if len(words) >= 2:  # need at least 2 words for context
                        sentences.append(words)
        except FileNotFoundError:
            print(f"ERROR: Corpus file not found at {self.corpus_path}")
            print("Make sure to run data collection and preprocessing first!")
            return []
        
        print(f"Loaded {len(sentences)} sentences")
        return sentences
    
    def train_model(self, model_type='cbow', vector_size=100, window=5, 
                    min_count=1, negative=5, epochs=10):
        """
        Trains a single Word2Vec model with given hyperparameters.
        
        Args:
            model_type: 'cbow' or 'skipgram'
            vector_size: Dimension of word vectors (embedding size)
            window: Context window size
            min_count: Ignore words with frequency lower than this
                       (set to 1 for small corpora to keep more vocab)
            negative: Number of negative samples (0 means no negative sampling)
            epochs: Number of training iterations
            
        Returns:
            Trained Word2Vec model
        """
        
        if not self.sentences:
            print("No sentences loaded! Can't train.")
            return None
        
        # sg=0 means CBOW, sg=1 means Skip-gram
        # this naming is a bit confusing but that's how gensim does it
        sg = 0 if model_type.lower() == 'cbow' else 1
        
        model_name = f"{model_type}_dim{vector_size}_win{window}_neg{negative}"
        print(f"\nTraining {model_name}...")
        
        start_time = time.time()
        
        # the actual training happens here
        # workers=4 uses multiple CPU cores for faster training
        model = Word2Vec(
            sentences=self.sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            negative=negative,
            sg=sg,
            workers=4,
            epochs=epochs,
            seed=42  # for reproducibility
        )
        
        training_time = time.time() - start_time
        
        # store model
        self.models[model_name] = model
        
        # save model to disk
        model_path = f"./models/{model_name}.model"
        model.save(model_path)
        
        # record results
        result = {
            'model_name': model_name,
            'model_type': model_type,
            'vector_size': vector_size,
            'window': window,
            'negative_samples': negative,
            'min_count': min_count,
            'epochs': epochs,
            'vocabulary_size': len(model.wv),
            'training_time': round(training_time, 2),
            'model_path': model_path
        }
        self.results.append(result)
        
        print(f"  ✓ Vocab size: {len(model.wv)}")
        print(f"  ✓ Training time: {training_time:.2f}s")
        print(f"  ✓ Saved to: {model_path}")
        
        return model
    
    def run_experiments(self):
        """
        Runs all the experiments required by the assignment.
        Trains multiple models with different hyperparameters.
        
        We're experimenting with:
        - Model type: CBOW vs Skip-gram
        - Embedding dimensions: 50, 100, 200
        - Window sizes: 3, 5, 7
        - Negative samples: 5, 10
        """
        
        print("\n" + "=" * 60)
        print("WORD2VEC EXPERIMENTS")
        print("=" * 60)
        
        # hyperparameter configurations to try
        # not going crazy here - just enough to show we experimented
        
        embedding_dims = [50, 100, 200]
        window_sizes = [3, 5, 7]
        negative_samples = [5, 10]
        model_types = ['cbow', 'skipgram']
        
        # we'll do a subset of combinations to keep it manageable
        # full grid would be 2 x 3 x 3 x 2 = 36 models - too many!
        
        experiments = [
            # vary embedding dimension (fix window=5, negative=5)
            {'model_type': 'cbow', 'vector_size': 50, 'window': 5, 'negative': 5},
            {'model_type': 'cbow', 'vector_size': 100, 'window': 5, 'negative': 5},
            {'model_type': 'cbow', 'vector_size': 200, 'window': 5, 'negative': 5},
            
            {'model_type': 'skipgram', 'vector_size': 50, 'window': 5, 'negative': 5},
            {'model_type': 'skipgram', 'vector_size': 100, 'window': 5, 'negative': 5},
            {'model_type': 'skipgram', 'vector_size': 200, 'window': 5, 'negative': 5},
            
            # vary window size (fix dim=100, negative=5)
            {'model_type': 'cbow', 'vector_size': 100, 'window': 3, 'negative': 5},
            {'model_type': 'cbow', 'vector_size': 100, 'window': 7, 'negative': 5},
            
            {'model_type': 'skipgram', 'vector_size': 100, 'window': 3, 'negative': 5},
            {'model_type': 'skipgram', 'vector_size': 100, 'window': 7, 'negative': 5},
            
            # vary negative samples (fix dim=100, window=5)
            {'model_type': 'cbow', 'vector_size': 100, 'window': 5, 'negative': 10},
            {'model_type': 'skipgram', 'vector_size': 100, 'window': 5, 'negative': 10},
        ]
        
        print(f"\nRunning {len(experiments)} experiments...")
        print("-" * 60)
        
        for i, config in enumerate(experiments, 1):
            print(f"\n[Experiment {i}/{len(experiments)}]")
            self.train_model(**config)
        
        print("\n" + "=" * 60)
        print("ALL EXPERIMENTS COMPLETED!")
        print("=" * 60)
        
        return self.results
    
    def print_results_table(self):
        """
        Prints a nice formatted table of all experiment results.
        Good for copy-pasting into the report.
        """
        if not self.results:
            print("No results yet. Run experiments first!")
            return
        
        print("\n" + "=" * 90)
        print("EXPERIMENT RESULTS SUMMARY")
        print("=" * 90)
        
        # header
        header = f"{'Model Type':<12} {'Dim':<6} {'Window':<8} {'Neg Samples':<12} {'Vocab':<8} {'Time(s)':<10}"
        print(header)
        print("-" * 90)
        
        # rows
        for r in self.results:
            row = f"{r['model_type']:<12} {r['vector_size']:<6} {r['window']:<8} {r['negative_samples']:<12} {r['vocabulary_size']:<8} {r['training_time']:<10}"
            print(row)
        
        print("=" * 90 + "\n")
        
        # save results to JSON file
        results_path = "./outputs/experiment_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {results_path}")
    
    def get_model(self, model_name):
        """
        Returns a trained model by name.
        If not in memory, tries to load from disk.
        """
        if model_name in self.models:
            return self.models[model_name]
        
        # try loading from disk
        model_path = f"./models/{model_name}.model"
        if os.path.exists(model_path):
            model = Word2Vec.load(model_path)
            self.models[model_name] = model
            return model
        
        print(f"Model '{model_name}' not found!")
        return None
    
    def get_best_models(self):
        """
        Returns the best CBOW and Skip-gram models.
        "Best" here means the one with dim=100, window=5, neg=5
        (standard configuration that usually works well)
        """
        cbow_model = self.get_model("cbow_dim100_win5_neg5")
        skipgram_model = self.get_model("skipgram_dim100_win5_neg5")
        
        return cbow_model, skipgram_model


def load_sentences_from_corpus(corpus_path="./data/processed/cleaned_corpus.txt"):
    """
    Helper function to load sentences for training.
    Useful if you want to train models separately.
    """
    sentences = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            if len(words) >= 2:
                sentences.append(words)
    return sentences


def train_models(corpus_path="./data/processed/cleaned_corpus.txt"):
    """
    Main function to train all models.
    Call this after preprocessing is done.
    """
    trainer = Word2VecTrainer(corpus_path)
    
    if not trainer.sentences:
        return None, None
    
    # run all experiments
    results = trainer.run_experiments()
    
    # print results table
    trainer.print_results_table()
    
    # return the best models for further analysis
    return trainer.get_best_models()


if __name__ == "__main__":
    train_models()
