import numpy as np
from collections import defaultdict, Counter

def compute_cider_score(generated_caption, reference_captions, tokenizer=None):
    def get_ngrams(caption, n):
        words = caption.split()
        return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

    def compute_tf(captions, n):
        tf_scores = []
        for caption in captions:
            ngrams = get_ngrams(caption, n)
            tf = Counter(ngrams)
            tf_scores.append(tf)
        return tf_scores

    def compute_idf(reference_captions, n):
        all_ngrams = defaultdict(int)
        for ref in reference_captions:
            ngrams = set(get_ngrams(ref, n))
            for ngram in ngrams:
                all_ngrams[ngram] += 1
        num_docs = len(reference_captions)
        idf = {ngram: np.log(float(num_docs) / (count + 1)) for ngram, count in all_ngrams.items()}
        return idf

    def compute_tf_idf(tf_scores, idf):
        tf_idf_scores = []
        for tf in tf_scores:
            tf_idf = {ngram: freq * idf.get(ngram, 0) for ngram, freq in tf.items()}
            tf_idf_scores.append(tf_idf)
        return tf_idf_scores

    def cosine_similarity(vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])
        sum1 = sum([vec1[x]**2 for x in vec1.keys()])
        sum2 = sum([vec2[x]**2 for x in vec2.keys()])
        denominator = np.sqrt(sum1) * np.sqrt(sum2)
        if denominator == 0:
            return 0.0
        return float(numerator) / denominator

    # Tokenize the captions using PTBTokenizer
    if not tokenizer is None:
        tokenized_generated = tokenizer.tokenize({'image1': [{'caption': generated_caption}]})['image1'][0] # tokenized sentence
        tokenized_references = tokenizer.tokenize({'image1': [{'caption': ref} for ref in reference_captions]})['image1'] # tokenized sentences
    else:
        tokenized_generated = generated_caption
        tokenized_references = reference_captions
    # Parameters
    n = 1

    # Compute tf for generated caption and reference captions
    gen_tf = compute_tf([tokenized_generated], n)[0]
    ref_tfs = compute_tf([ref for ref in tokenized_references], n)

    # Compute idf from reference captions
    idf = compute_idf([ref for ref in tokenized_references], n)

    # Compute tf-idf for generated caption and reference captions
    gen_tf_idf = compute_tf_idf([gen_tf], idf)[0]
    ref_tf_idfs = compute_tf_idf(ref_tfs, idf)

    # Compute cosine similarities
    similarities = [cosine_similarity(gen_tf_idf, ref_tf_idf) for ref_tf_idf in ref_tf_idfs]
    cider_score = np.mean(similarities)
    
    return cider_score