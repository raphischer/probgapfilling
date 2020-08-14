FROM continuumio/anaconda3

LABEL maintainer="Raphael Fischer"

RUN apt-get -y install nano zip unzip screen gcc htop texlive-latex-extra

RUN mkdir -p /home/exp/probgf
RUN mkdir -p /home/exp/scripts
RUN mkdir -p /home/exp/tests
COPY probgf/* /home/exp/probgf/
COPY tests/* /home/exp/tests/
COPY environment.yml /home/exp/
COPY MANIFEST.in /home/exp/
COPY setup.py /home/exp/
COPY README.md /home/exp/
COPY LICENSE.md /home/exp/
COPY scripts/* /home/exp/scripts/

RUN conda env create -f /home/exp/environment.yml
RUN conda run -n probgf python -m pip install /home/exp