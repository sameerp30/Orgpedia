from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Trainer

if __name__=='__main__':
    with Run().context(RunConfig(nranks=4, experiment="marathi-synthetic")):

        config = ColBERTConfig(
            bsize=32,
            root="/raid/nlp/sameer/mahGRs/indiccolbert/mar_Deva-nllb1.3b-moses/colbert-50000",
            maxsteps=350
        )
        trainer = Trainer(
            triples="/raid/nlp/sameer/mahGRs/synthetic_triples_with_date.json",
            queries="/raid/nlp/sameer/mahGRs/synthetic_titles_with_date.tsv",
            collection="/raid/nlp/sameer/mahGRs/index_marathi_finance.tsv",
            config=config,
        )

        checkpoint_path = trainer.train()

        print(f"Saved checkpoint to {checkpoint_path}...")