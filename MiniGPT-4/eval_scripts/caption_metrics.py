from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

def spice_score(generated_caption, reference_captions):
    
    tokenizer = PTBTokenizer()
    tokenized_generated = tokenizer.tokenize({'image1': [{'caption': generated_caption}]})
    tokenized_references = tokenizer.tokenize({'image1': [{'caption': ref} for ref in reference_captions]})

    spice = Spice()

    score, scores = spice.compute_score(tokenized_references, tokenized_generated)

    return score