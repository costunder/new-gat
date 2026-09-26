"""User-approved production benchmark; historical data/model helpers stay readable."""

PRIMARY_DATASET = "ogbn-arxiv"
REQUIRED_BASELINES = ("gcn", "graphsage", "gatv2")


def require_benchmark_datasets(datasets):
    if list(datasets) != [PRIMARY_DATASET]:
        raise ValueError(
            "New comparison runs use ogbn-arxiv only. PPI and legacy citation datasets "
            "are excluded from this benchmark; old results remain preserved."
        )
