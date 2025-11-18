import os
import subprocess
import tempfile

# path to the stanford corenlp jar
STANFORD_CORENLP_3_4_1_JAR = 'stanford-corenlp-3.4.1.jar'

# punctuations to be removed from the sentences
PUNCTUATIONS = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-",
                ".", "?", "!", ",", ":", "-", "--", "...", ";"]


class PTBTokenizer:
    """Python wrapper of Stanford PTBTokenizer"""

    def __init__(self, _source='gts'):
        self.source = _source

    def tokenize(self, captions_for_image):
        cmd = ['java', '-cp', STANFORD_CORENLP_3_4_1_JAR,
               'edu.stanford.nlp.process.PTBTokenizer',
               '-preserveLines', '-lowerCase']

        # ======================================================
        # prepare data for PTB Tokenizer
        # ======================================================

        if self.source == 'gts':
            image_id = [k for k, v in captions_for_image.items() for _ in range(len(v))]
            sentences = '\n'.join([c['caption'].replace('\n', ' ') for k, v in captions_for_image.items() for c in v])
            final_tokenized_captions_for_image = {}

        elif self.source == 'res':
            index = [i for i, v in enumerate(captions_for_image)]
            image_id = [v["image_id"] for v in captions_for_image]
            sentences = '\n'.join(v["caption"].replace('\n', ' ') for v in captions_for_image)
            final_tokenized_captions_for_index = []

        # ======================================================
        # save sentences to temporary file
        # ======================================================
        path_to_jar_dir_name = os.path.dirname(os.path.abspath(__file__))
        # tmp_file = tempfile.NamedTemporaryFile(delete=False, dir=path_to_jar_dir_name)
        tmp_file = tempfile.NamedTemporaryFile(delete=False, dir=path_to_jar_dir_name, mode='w', encoding='utf-8')
        tmp_file.write(sentences)
        tmp_file.close()

        # ======================================================
        # tokenize sentence
        # ======================================================
        cmd.append(os.path.basename(tmp_file.name))
        p_tokenizer = subprocess.Popen(cmd, cwd=path_to_jar_dir_name, stdout=subprocess.PIPE)
        token_lines = p_tokenizer.communicate(input=sentences.rstrip())[0].decode('utf-8')
        lines = token_lines.split('\n')
        # remove temp file
        os.remove(tmp_file.name)

        # ======================================================
        # create dictionary for tokenized captions
        # ======================================================
        if self.source == 'gts':
            for k, line in zip(image_id, lines):
                if k not in final_tokenized_captions_for_image:
                    final_tokenized_captions_for_image[k] = []
                tokenized_caption = ' '.join([w for w in line.rstrip().split(' ') if w not in PUNCTUATIONS])
                final_tokenized_captions_for_image[k].append(tokenized_caption)

            return final_tokenized_captions_for_image

        elif self.source == 'res':
            for k, img, line in zip(index, image_id, lines):
                tokenized_caption = ' '.join([w for w in line.rstrip().split(' ') if w not in PUNCTUATIONS])
                final_tokenized_captions_for_index.append({'image_id': img, 'caption': [tokenized_caption]})

            return final_tokenized_captions_for_index
