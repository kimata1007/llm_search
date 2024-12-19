FROM python:3.11.2

ENV LOCAL_UID 501
ENV LOCAL_UNAME kimata.20038
ENV GIT_UNAME kimata1007
ENV GIT_EMAIL muquanchunda@gmail.com

RUN apt-get update && apt-get install -y vim zsh git ssh sudo tmux
RUN python -m pip install --upgrade pip
RUN python -m pip install numpy pandas tqdm ipykernel jsonlines==4.0.0 openai python-dotenv==1.0.1 langchain==0.2.0 langchain-openai langchain_community toml pipreqs bs4

RUN useradd -u $LOCAL_UID $LOCAL_UNAME
# shをzshに変更
RUN chsh -s /bin/zsh
RUN mkdir -p /home/$LOCAL_UNAME && chown $LOCAL_UNAME /home/$LOCAL_UNAME
USER $LOCAL_UNAME
WORKDIR /workspace

RUN git config --global user.name $GIT_UNAME
RUN git config --global user.email $GIT_EMAIL

CMD [ "zsh" ]

