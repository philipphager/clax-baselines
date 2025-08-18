from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import List, Set

import hydra
from omegaconf import DictConfig, OmegaConf
from progress_table import ProgressTable
from pyclick.click_models import CM, PBM, UBM, DBN, SDBN, DCM, CCM
from pyclick.click_models.CTR import GCTR, DCTR, RCTR
from pyclick.click_models.Evaluation import LogLikelihood, Perplexity, PerplexityCond
from pyclick.search_session import SearchSession
from pyclick.utils.Utils import Utils

from dataset import YandexRelPredChallengeParser

name2model = {
    "GCTR": GCTR,
    "DCTR": DCTR,
    "RCTR": RCTR,
    "CM": CM,
    "PBM": PBM,
    "UBM": UBM,
    "DBN": DBN,
    "SDBN": SDBN,
    "DCM": DCM,
    "CCM": CCM,
}


def evaluate(model_name, model, sessions, stage):
    loglikelihood = LogLikelihood()
    ppl = Perplexity()
    cond_ppl = PerplexityCond()

    logger = ProgressTable(
        columns=[
            "model",
            f"{stage}_ll",
            f"{stage}_ppl",
            f"{stage}_cond_ppl",
        ],
    )
    logger.update_from_dict(
        {
            "model": model_name,
            f"{stage}_ll": loglikelihood.evaluate(model, sessions),
            f"{stage}_ppl": ppl.evaluate(model, sessions)[0],
            f"{stage}_cond_ppl": cond_ppl.evaluate(model, sessions)[0],
        }
    )
    logger.close()
    return logger.to_df()


def get_unique_queries(
    search_sessions: List[SearchSession],
    min_sessions: int = 1,
) -> Set[str]:
    query_counter = Counter()

    for search_session in search_sessions:
        query_counter.update([search_session.query])

    return {
        query_id for query_id, count in query_counter.items() if count >= min_sessions
    }


@hydra.main(config_name="config", config_path="config", version_base="1.2")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    result_dir = Path(f"results/{config.experiment}")
    result_dir.mkdir(exist_ok=True, parents=True)
    model_name = config.model.lower()

    parser = YandexRelPredChallengeParser()
    train_sessions = parser.parse(config.dataset, sessions_range=config.train_sessions)
    val_sessions = parser.parse(config.dataset, sessions_range=config.val_sessions)
    test_sessions = parser.parse(config.dataset, sessions_range=config.test_sessions)

    if config.min_train_sessions_per_eval_query > 0:
        train_queries = get_unique_queries(
            train_sessions,
            min_sessions=config.min_train_sessions_per_eval_query,
        )
        val_sessions = Utils.filter_sessions(val_sessions, train_queries)
        test_sessions = Utils.filter_sessions(test_sessions, train_queries)
        print(
            f"Evaluating {len(train_queries)} unique train queries, resulting in "
            f"{len(val_sessions)} val sessions, {len(test_sessions)} test sessions"
        )

    model = name2model[config.model.upper()]()

    timer_start = perf_counter()
    model.train(train_sessions)
    timer_stop = perf_counter()
    train_time = timer_stop - timer_start

    val_df = evaluate(model_name, model, val_sessions, "val")

    test_df = evaluate(model_name, model, test_sessions, "test")
    test_df["train_time_s"] = train_time

    val_df.to_csv(result_dir / f"val_{model_name}.csv", index=False)
    test_df.to_csv(result_dir / f"test_{model_name}.csv", index=False)


if __name__ == "__main__":
    main()
