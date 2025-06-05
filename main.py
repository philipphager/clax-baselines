from pathlib import Path
from time import perf_counter
from typing import List, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf
from progress_table import ProgressTable
from pyclick.click_models import CM, PBM, UBM, DBN, SDBN, DCM, CCM
from pyclick.click_models.CTR import GCTR, DCTR, RCTR
from pyclick.click_models.Evaluation import LogLikelihood, Perplexity, PerplexityCond
from pyclick.click_models.task_centric.TaskCentricSearchSession import \
    TaskCentricSearchSession
from pyclick.utils.YandexRelPredChallengeParser import YandexRelPredChallengeParser

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


def train_val_test_split(
        sessions: List[TaskCentricSearchSession],
        splits: List[float]
) -> Tuple[List, List, List]:
    assert sum(splits) == 1.0, "train/val/test splits must sum to 1.0"

    total_sessions = len(sessions)
    n_train = int(total_sessions * splits[0])
    n_val = int(total_sessions * splits[1])

    return (
        sessions[:n_train],
        sessions[n_train:(n_train + n_val)],
        sessions[(n_train + n_val):]
    )


@hydra.main(config_name="config", config_path="config", version_base="1.2")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    parser = YandexRelPredChallengeParser()
    sessions = parser.parse(config.dataset, config.total_sessions)
    train_sessions, _, test_sessions = train_val_test_split(
        sessions, config.train_val_test_splits
    )

    model = name2model[config.model.upper()]()

    timer_start = perf_counter()
    model.train(train_sessions)
    timer_stop = perf_counter()
    train_time = timer_stop - timer_start

    loglikelihood = LogLikelihood()
    ppl = Perplexity()
    cond_ppl = PerplexityCond()

    logger = ProgressTable(
        columns=[
            "model",
            "test_ll",
            "test_ppl",
            "test_cond_ppl",
            "train_time_s",
        ],
    )
    logger.update_from_dict(
        {
            "model": config.model.upper(),
            "test_ll": loglikelihood.evaluate(model, test_sessions),
            "test_ppl": ppl.evaluate(model, test_sessions)[0],
            "test_cond_ppl": cond_ppl.evaluate(model, test_sessions)[0],
            "train_time_s": train_time,
        }
    )
    logger.close()

    result_dir = Path("results/")
    result_dir.mkdir(exist_ok=True)

    df = logger.to_df()
    df.to_csv(result_dir / f"{config.model.lower()}_test.csv", index=False)


if __name__ == '__main__':
    main()
