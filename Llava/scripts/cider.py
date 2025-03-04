# coding: utf-8
import argparse
import json
from scripts.pyciderevalcap.eval import CIDErEvalCap as ciderEval
# from pyciderevalcap.eval import CIDErEvalCap as ciderEval
from collections import defaultdict
"""
Load the reference and candidate json files, which are to be evaluated using CIDEr.

Reference file: list of dict('image_id': image_id, 'caption': caption).
Candidate file: list of dict('image_id': image_id, 'caption': caption).

"""

def readfile(path_to_ref_file, path_to_cand_file):
    if path_to_ref_file.endswith('json'):
        ref_list = json.loads(open(path_to_ref_file, 'r', encoding="UTF-8").read())
    else:
        ref_list = []
        with open(path_to_ref_file, 'r', encoding="UTF-8") as file:
            for line in file:
                ref_list.append(json.loads(line))
    if path_to_cand_file.endswith('json'):
        cand_list = json.loads(open(path_to_cand_file, 'r', encoding="UTF-8").read())
    else:
        cand_list = []
        with open(path_to_cand_file, 'r', encoding="UTF-8") as file:
            for line in file:
                cand_list.append(json.loads(line))

    gts = defaultdict(list)

    for l in ref_list:
        for caption in l['caption']:
            gts[l['image_id']].append({"caption": caption})

    res = cand_list
    return gts, res

def calculate_cider(args):

    # load the configuration file
    df_mode = 'coco-val'

    # Print the parameters
    print("Running CIDEr with the following settings")
    print("*****************************")
    print("Reference File:%s" % args.refpath)
    print("Candidate File:%s" % args.candpath)
    print("Result File:%s" % args.resultfile)
    print("IDF:%s" % df_mode)
    print("*****************************")

    # load reference and candidate sentences
    gts, res = readfile(args.refpath, args.candpath)

    # calculate cider scores
    scorer = ciderEval(gts, res, df_mode)
    # scores: dict of list with key = metric and value = score given to each
    # candidate
    scores = scorer.evaluate()

    with open(args.resultfile, 'w') as outfile:
        json.dump(scores, outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refpath", type=str)
    parser.add_argument("--candpath", type=str)
    parser.add_argument("--resultfile", type=str)
    args = parser.parse_args()

    calculate_cider(args)