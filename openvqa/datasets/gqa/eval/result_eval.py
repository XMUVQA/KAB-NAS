# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from openvqa.datasets.gqa.eval.gqa_eval import GQAEval
import json, pickle
import numpy as np


def eval(__C,  dataset, ans_ix_list, result_eval_file, log_file, valid=True):
    result_eval_file = result_eval_file + '.json'

    qid_list = [qid for qid in dataset.qid_list]
    ans_size = dataset.ans_size

    result = [{
        'questionId': qid_list[ix],
        'prediction': dataset.ix_to_ans[ans_ix_list[ix]],
    } for ix in range(len(qid_list))]

    print('Save the result to file: {}'.format(result_eval_file))
    json.dump(result, open(result_eval_file, 'w'))



    if valid:
        # create vqa object and vqaRes object
        ques_file_path = __C.RAW_PATH[__C.DATASET][__C.SPLIT['val']]
        choices_path = None
        if __C.SPLIT['val'] + '_choices' in __C.RAW_PATH[__C.DATASET]:
            choices_path = __C.RAW_PATH[__C.DATASET][__C.SPLIT['val'] + '_choices']

        eval_gqa = GQAEval(__C, result_eval_file, ques_file_path, choices_path, EVAL_CONSISTENCY=False)
        result_string, detail_result_string = eval_gqa.get_str_result()

        print('Write to log file: {}'.format(log_file))
        logfile = open(log_file, 'a+')

        for result_string_ in result_string:
            logfile.write(result_string_)
            logfile.write('\n')
            print(result_string_)

        for detail_result_string_ in detail_result_string:
            logfile.write(detail_result_string_)
            logfile.write("\n")

        logfile.write('\n')
        logfile.close()
        return eval_gqa.scores['accuracy']


