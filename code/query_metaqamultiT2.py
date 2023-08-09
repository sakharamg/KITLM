import regex as re
from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher
import pandas as pd
from tqdm import tqdm
import itertools
hops = ['2hop', '3hop']
splits = ["dev", "test", "train"]
# HOP = 1
TOP_K = 3
all_entities = [word.lower() for word in pd.read_csv("../data/wikimovies/entities.txt",
                                                     sep="\t", header=0, names=["id", "entity"])["entity"].to_list()]

with Run().context(RunConfig(experiment='metaqa')):
    searcher = Searcher(index='wikimovies.nbits=2')


def query_colbert_fancier(query, searcher=searcher, k=5, hops=1, limit=20):
    visited_entities = [[re.search('\[(.*)\]', query).group(1).lower()]]
    all_visited_entities = []
    all_visited_entities.append(visited_entities[0][0])

    visited_pids = []
    og_query = query
    # print(visited_entities)
    for hop in range(hops):
        curr_passages = []
        results = searcher.search(query, k=k)
        curr = []
        for id, (passage_id, passage_rank, passage_score) in enumerate(zip(*results)):
            if passage_id not in visited_pids:
                visited_pids.append(passage_id)
                curr_passage = " "+searcher.collection[passage_id][:-1]+" ."
                # print(hop+1, curr_passage)
                if any(" "+entity+" " in curr_passage for entity in visited_entities[-1]):
                    # print("out",hop+1, curr_passage)
                    curr_passages.append(curr_passage)
                    # query+=curr_passage
                    for entity in all_entities:
                        if " "+entity+" " in curr_passage and entity not in visited_entities[-1] and not any(entity in enTT for enTT in all_visited_entities):
                            curr.append(entity)
        if len(curr) == 0:
            results = searcher.search(query, k=limit)
            for id, (passage_id, passage_rank, passage_score) in enumerate(zip(*results)):
                if passage_id not in visited_pids:
                    visited_pids.append(passage_id)
                    curr_passage = " " + \
                        searcher.collection[passage_id][:-1]+" ."
                    # print(hop+1, curr_passage)
                    if any(" "+entity+" " in curr_passage for entity in visited_entities[-1]):
                        # print("in",hop+1, curr_passage)
                        curr_passages.append(curr_passage)
                        for entity in all_entities:
                            if " "+entity+" " in curr_passage and entity not in visited_entities[-1] and not any(entity in enTT for enTT in all_visited_entities):
                                curr.append(entity)
        k *= 2
        # print(visited_entities)
        # print(query)
        visited_entities.append(list(set(curr)))
        all_visited_entities.extend(curr)
        query += ''.join(curr_passages)
    return str.strip(query.replace(og_query, ""))
    # return [searcher.collection[passage_id] for passage_id in visited_pids]


for hop in hops:
    for split in splits:
        questions = []
        answers = []
        contexts = []
        # qas = open("../data/metaqa/"+hop+"/qa_"+split +
        #            ".txt", 'r').read().splitlines()
        qas = pd.read_csv("../data/metaqa/"+hop+"/qa_"+split +
                          ".txt", sep='\t', header=0, names=["query", 'answer'])
        # queries = qas["query"].to_list()
        # anss = qas["answer"].to_list()
        for ind in tqdm(qas.index, desc=hop+" and "+split):
            # for index, row in tqdm(qas.iterrows(), desc=hop+" and "+split):
            query, ans = qas['query'][ind], qas['answer'][ind]
            top_k_passages = query_colbert_fancier(
                query, hops=int(hop[0]), searcher=searcher, k=TOP_K)
            context = top_k_passages.replace("\t", " ")
            questions.append(query)
            answers.append(ans)
            contexts.append(context)
            # if ind == 20:
            #     break
        details = {
            'qid': [a for a in range(len(questions))],
            'question': questions,
            'answer': answers,
            'context': contexts,
        }

        # creating a Dataframe object with skipping
        # one column i.e skipping age column.
        df = pd.DataFrame(details, columns=[
                          'qid', 'question', 'answer', 'context'])
        df.to_csv("../data/metaqa/"+hop+"/qa_"+split+"_triples_multitop"+str(TOP_K)+".tsv",
                  index=False, sep='\t', header=False)
