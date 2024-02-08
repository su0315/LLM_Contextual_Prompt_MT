FROM continuumio/miniconda3:latest

WORKDIR /app

COPY llm_mt.yml .
RUN conda info \
conda env create -f llm_mt.yml

ENTRYPOINT [ "conda", "run", "--no-capture-output", "-n", "llm_mt", "python" ]

