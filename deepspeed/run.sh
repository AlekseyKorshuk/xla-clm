git clone https://github.com/AlekseyKorshuk/xla-clm && \
  cd xla-clm && cd "deepspeed " && \
  pip install datasets && \
  pip install git+https://github.com/AlekseyKorshuk/DeepSpeed && \
  python3 bert.py
