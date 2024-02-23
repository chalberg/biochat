import chromadb
from pytrials.client import ClinicalTrials
from argparse import ArgumentParser

def init_db():
    pass

def update_db():
    pass

def get_clinical_data(test):
    if test:
        # load sample NCTIds
        with open('data/sample_studies.txt', 'r') as sample_trials:
            trials = sample_trials.read()
            ids = list(trials.splitlines())

        ct = ClinicalTrials()
        for id in ids[:10]:
            print(id)
            study = ct.get_study_fields(search_expr='{}'.format(id),
                                        fields=['NCTId','BriefTitle','BriefSummary'],
                                        fmt='csv',
                                        max_studies=1)
            print(study[1][3]) # brief summary
            

if __name__=="__main__":
    parser = ArgumentParser()

    parser.add_argument('--update_db', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()
    
    get_clinical_data(args.test)