# Scopo: entrypoint CLI della pipeline.
from pipeline_modules.cli import build_arg_parser
from pipeline_modules.constants import DEFAULT_VLM_MODEL, DEFAULT_YOLO_MODEL
from pipeline_modules.runner import run_pipeline


def main() -> None:
    """
    Avvia la pipeline da CLI.
    Legge gli argomenti di default per modelli e soglie.
    Inoltra i parametri a run_pipeline.
    """
    args = build_arg_parser(DEFAULT_VLM_MODEL, DEFAULT_YOLO_MODEL).parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
