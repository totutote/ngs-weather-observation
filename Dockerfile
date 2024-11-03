# 1. ベースイメージの選択
FROM condaforge/miniforge3

COPY environment.yml /tmp/environment.yml

RUN conda update conda && conda env create -f /tmp/environment.yml && conda clean -afy

# 環境変数の設定
ENV LD_PRELOAD=/opt/conda/envs/ngswether/lib/python3.10/site-packages/cv2/python-3.10/../../../.././libgomp.so.1
RUN echo "export LD_PRELOAD=$LD_PRELOAD" >> ~/.bashrc

# activate myenv
ENV CONDA_DEFAULT_ENV ngswether

RUN echo "conda activate $CONDA_DEFAULT_ENV" >> ~/.bashrc
ENV PATH /opt/conda/envs/$CONDA_DEFAULT_ENV/bin:$PATH

# 作業ディレクトリを設定
WORKDIR /home/jovyan/work

# 環境をアクティベートする
SHELL ["conda", "run", "-n", "$CONDA_DEFAULT_ENV", "/bin/bash", "-c"]
