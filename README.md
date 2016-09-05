# LSTM-Retreival-System
This code implements a dual LSTM model as well as the TF-IDF method for the Ubuntu Dialogue Corpus.

To run:
=======
1. Download the [data](https://drive.google.com/file/d/0B_bZck-ksdkpVEtVc1R6Y01HMWM/view) and put it in ```scripts/data```.
2. Run ```python scripts/prepare.py``` to create dataset specific to Theano.
3. Run ```python ran.py``` to test random evaluation for recall@k.
4. Run ```python tfidf.py``` to test TFIDF for recall@k.
5. Run ```python train.py``` to train the dual encoder LSTM model for recall@k.
6. Modify variable ```MODEL_DIR``` in ```train.py``` to resume training of model (runs are saved in ```runs/...```.
7. Run ```python test.py --model_dir=runs/...``` to test model on training set.
