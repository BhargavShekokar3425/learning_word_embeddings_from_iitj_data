"""
Semantic Analysis Module for Word2Vec Assignment

Author: Bhargav Shekokar (B23CS1008)
Course: NLU Assignment 2
"""

import os
import numpy as np
from gensim.models import Word2Vec
from collections import defaultdict
import json


class SemanticAnalyzer:
    """
    Analyzes semantic relationships captured by word embeddings.
    Does similarity search, analogies, and other fun stuff.
    """
    
    def __init__(self, cbow_model=None, skipgram_model=None):
        """
        Initialize with trained models.
        
        Args:
            cbow_model: Trained CBOW Word2Vec model
            skipgram_model: Trained Skip-gram Word2Vec model
        """
        self.cbow = cbow_model
        self.skipgram = skipgram_model
        
        # store results for report
        self.similarity_results = {}
        self.analogy_results = {}
        
        # create output directory
        os.makedirs("./outputs", exist_ok=True)
        
    def load_model(self, model_path):
        """
        Loads a Word2Vec model from disk.
        """
        try:
            return Word2Vec.load(model_path)
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            return None
    
    def load_default_models(self):
        """
        Loads the default (best) models from the models directory.
        Useful when you want to analyze without retraining.
        """
        cbow_path = "./models/cbow_dim100_win5_neg5.model"
        skipgram_path = "./models/skipgram_dim100_win5_neg5.model"
        
        if os.path.exists(cbow_path):
            self.cbow = self.load_model(cbow_path)
            print(f"Loaded CBOW model from {cbow_path}")
        
        if os.path.exists(skipgram_path):
            self.skipgram = self.load_model(skipgram_path)
            print(f"Loaded Skip-gram model from {skipgram_path}")
        
        return self.cbow, self.skipgram
    
    def find_similar_words(self, word, model, top_n=5):
        """
        Finds top N most similar words using cosine similarity.
        
        Args:
            word: Target word
            model: Word2Vec model to use
            top_n: Number of similar words to return
            
        Returns:
            List of (word, similarity_score) tuples
        """
        if model is None:
            print("Model not loaded!")
            return []
        
        # check if word exists in vocabulary
        if word not in model.wv:
            print(f"'{word}' not in vocabulary!")
            # try lowercase
            word_lower = word.lower()
            if word_lower in model.wv:
                word = word_lower
            else:
                return []
        
        # gensim makes this super easy
        similar = model.wv.most_similar(word, topn=top_n)
        return similar
    
    def analyze_target_words(self):
        """
        Analyzes the target words required by the assignment:
        - research
        - student
        - phd
        - exam (listed twice in assignment, we'll just do it once)
        
        Reports top 5 nearest neighbors for each.
        """
        
        target_words = ['research', 'student', 'phd', 'exam']
        
        print("\n" + "=" * 70)
        print("SEMANTIC SIMILARITY ANALYSIS")
        print("=" * 70)
        
        results = {
            'cbow': {},
            'skipgram': {}
        }
        
        for word in target_words:
            print(f"\n{'='*50}")
            print(f"Target Word: '{word}'")
            print(f"{'='*50}")
            
            # CBOW results
            if self.cbow:
                print(f"\n  CBOW Model - Top 5 Similar Words:")
                print("  " + "-" * 40)
                cbow_similar = self.find_similar_words(word, self.cbow, top_n=5)
                if cbow_similar:
                    for similar_word, score in cbow_similar:
                        print(f"    {similar_word:<20} (similarity: {score:.4f})")
                    results['cbow'][word] = cbow_similar
                else:
                    print(f"    Word '{word}' not found in CBOW vocabulary")
            
            # Skip-gram results
            if self.skipgram:
                print(f"\n  Skip-gram Model - Top 5 Similar Words:")
                print("  " + "-" * 40)
                sg_similar = self.find_similar_words(word, self.skipgram, top_n=5)
                if sg_similar:
                    for similar_word, score in sg_similar:
                        print(f"    {similar_word:<20} (similarity: {score:.4f})")
                    results['skipgram'][word] = sg_similar
                else:
                    print(f"    Word '{word}' not found in Skip-gram vocabulary")
        
        self.similarity_results = results
        
        # save results to file
        self._save_similarity_results()
        
        return results
    
    def _save_similarity_results(self):
        """
        Saves similarity results to a JSON file.
        """
        # convert tuples to lists for JSON serialization
        serializable_results = {}
        for model_type, words in self.similarity_results.items():
            serializable_results[model_type] = {}
            for word, similar_list in words.items():
                serializable_results[model_type][word] = [
                    {'word': w, 'similarity': float(s)} 
                    for w, s in similar_list
                ]
        
        output_path = "./outputs/similarity_results.json"
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nSimilarity results saved to {output_path}")
    
    def word_analogy(self, word_a, word_b, word_c, model, top_n=5):
        """
        Performs word analogy: A is to B as C is to ?
        Uses the classic vector arithmetic: result = B - A + C
        
        Example: UG is to BTech as PG is to ? (should give MTech)
        
        Args:
            word_a, word_b, word_c: Words for the analogy
            model: Word2Vec model to use
            top_n: Number of results to return
            
        Returns:
            List of (word, score) tuples
        """
        if model is None:
            return []
        
        # convert to lowercase and check vocabulary
        word_a = word_a.lower()
        word_b = word_b.lower()
        word_c = word_c.lower()
        
        missing_words = []
        for w in [word_a, word_b, word_c]:
            if w not in model.wv:
                missing_words.append(w)
        
        if missing_words:
            print(f"  Words not in vocabulary: {missing_words}")
            return []
        
        # gensim has a built-in method for this
        # it computes: positive - negative and finds most similar
        try:
            results = model.wv.most_similar(
                positive=[word_b, word_c],
                negative=[word_a],
                topn=top_n
            )
            return results
        except Exception as e:
            print(f"  Analogy failed: {e}")
            return []
    
    def run_analogy_experiments(self):
        """
        Runs the analogy experiments required by the assignment.
        
        Analogies to try:
        1. UG : BTech :: PG : ? (expected: MTech)
        2. student : learn :: faculty : ? (expected: teach/research)
        3. phd : thesis :: mtech : ? (expected: project)
        
        And a few more interesting ones.
        """
        
        print("\n" + "=" * 70)
        print("WORD ANALOGY EXPERIMENTS")
        print("=" * 70)
        
        # define analogy experiments
        # format: (word_a, word_b, word_c, expected_result, explanation)
        analogies = [
            # required by assignment
            ('ug', 'btech', 'pg', 'mtech', 
             'UG is to BTech as PG is to ?'),
            
            # more analogies to show understanding
            ('student', 'learn', 'faculty', 'teach',
             'Student is to Learn as Faculty is to ?'),
            
            ('phd', 'research', 'btech', 'project',
             'PhD is to Research as BTech is to ?'),
            
            ('mtech', 'postgraduate', 'btech', 'undergraduate',
             'MTech is to Postgraduate as BTech is to ?'),
            
            ('semester', 'exam', 'course', 'assignment',
             'Semester is to Exam as Course is to ?'),
            
            ('department', 'professor', 'hostel', 'student',
             'Department is to Professor as Hostel is to ?'),
        ]
        
        results = {
            'cbow': [],
            'skipgram': []
        }
        
        for word_a, word_b, word_c, expected, explanation in analogies:
            print(f"\n{'='*60}")
            print(f"Analogy: {explanation}")
            print(f"Format: {word_a} : {word_b} :: {word_c} : ?")
            print(f"Expected: {expected}")
            print(f"{'='*60}")
            
            analogy_result = {
                'analogy': f"{word_a} : {word_b} :: {word_c} : ?",
                'explanation': explanation,
                'expected': expected
            }
            
            # CBOW results
            if self.cbow:
                print(f"\n  CBOW Results:")
                print("  " + "-" * 40)
                cbow_results = self.word_analogy(word_a, word_b, word_c, 
                                                  self.cbow, top_n=5)
                if cbow_results:
                    for i, (word, score) in enumerate(cbow_results, 1):
                        marker = " ✓" if word == expected else ""
                        print(f"    {i}. {word:<20} (score: {score:.4f}){marker}")
                    
                    analogy_result['cbow_results'] = [
                        {'word': w, 'score': float(s)} 
                        for w, s in cbow_results
                    ]
                    analogy_result['cbow_correct'] = expected in [w for w, s in cbow_results]
                else:
                    print("    (Could not compute - missing words)")
                    analogy_result['cbow_results'] = []
                    analogy_result['cbow_correct'] = False
                
                results['cbow'].append(analogy_result.copy())
            
            # Skip-gram results
            if self.skipgram:
                print(f"\n  Skip-gram Results:")
                print("  " + "-" * 40)
                sg_results = self.word_analogy(word_a, word_b, word_c, 
                                               self.skipgram, top_n=5)
                if sg_results:
                    for i, (word, score) in enumerate(sg_results, 1):
                        marker = " ✓" if word == expected else ""
                        print(f"    {i}. {word:<20} (score: {score:.4f}){marker}")
                    
                    analogy_result['skipgram_results'] = [
                        {'word': w, 'score': float(s)} 
                        for w, s in sg_results
                    ]
                    analogy_result['skipgram_correct'] = expected in [w for w, s in sg_results]
                else:
                    print("    (Could not compute - missing words)")
                    analogy_result['skipgram_results'] = []
                    analogy_result['skipgram_correct'] = False
                
                results['skipgram'].append(analogy_result.copy())
        
        self.analogy_results = results
        
        # save results
        self._save_analogy_results()
        
        # print summary
        self._print_analogy_summary()
        
        return results
    
    def _save_analogy_results(self):
        """
        Saves analogy results to JSON file.
        """
        output_path = "./outputs/analogy_results.json"
        with open(output_path, 'w') as f:
            json.dump(self.analogy_results, f, indent=2)
        
        print(f"\nAnalogy results saved to {output_path}")
    
    def _print_analogy_summary(self):
        """
        Prints a summary of analogy experiment results.
        """
        print("\n" + "=" * 70)
        print("ANALOGY EXPERIMENTS SUMMARY")
        print("=" * 70)
        
        print("""
Discussion:
-----------
Word analogies test whether the model captures semantic relationships.
The classic example is: king - man + woman ≈ queen

In our IIT Jodhpur context, we expect relationships like:
- Academic levels (UG/PG, BTech/MTech)
- Roles (student/faculty)
- Activities (learn/teach, study/research)

The results may not always match expected outcomes because:
1. Our corpus is relatively small (domain-specific)
2. Some relationships might not be explicitly stated in the text
3. Word frequencies affect embedding quality

Generally, Skip-gram tends to perform better on analogy tasks,
especially for less frequent words, while CBOW is faster to train.
        """)
    
    def get_word_vector(self, word, model):
        """
        Gets the vector for a word.
        Useful for manual inspection or custom analysis.
        """
        if model is None or word not in model.wv:
            return None
        return model.wv[word]
    
    def compute_similarity(self, word1, word2, model):
        """
        Computes cosine similarity between two words.
        """
        if model is None:
            return None
        
        word1, word2 = word1.lower(), word2.lower()
        
        if word1 not in model.wv or word2 not in model.wv:
            return None
        
        return model.wv.similarity(word1, word2)
    
    def explore_vocabulary(self, model, n=20):
        """
        Shows some vocabulary statistics.
        Useful for understanding what the model learned.
        """
        if model is None:
            return
        
        vocab = list(model.wv.key_to_index.keys())
        print(f"\nVocabulary size: {len(vocab)}")
        print(f"Sample words: {vocab[:n]}")


def run_semantic_analysis(cbow_model=None, skipgram_model=None):
    """
    Main function to run all semantic analysis.
    
    Args:
        cbow_model: Trained CBOW model (or None to load from disk)
        skipgram_model: Trained Skip-gram model (or None to load from disk)
    """
    analyzer = SemanticAnalyzer(cbow_model, skipgram_model)
    
    # load models if not provided
    if cbow_model is None or skipgram_model is None:
        analyzer.load_default_models()
    
    # check if we have models to work with
    if analyzer.cbow is None and analyzer.skipgram is None:
        print("ERROR: No models available for analysis!")
        print("Train models first using word2vec_trainer.py")
        return None
    
    # run similarity analysis
    similarity_results = analyzer.analyze_target_words()
    
    # run analogy experiments
    analogy_results = analyzer.run_analogy_experiments()
    
    return analyzer


if __name__ == "__main__":
    # run semantic analysis on trained models
    run_semantic_analysis()
