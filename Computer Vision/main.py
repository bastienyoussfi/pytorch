from fastai.vision.all import *

path = Path('cat_or_dog')

learn_inf = load_learner('model.pkl')
print(learn_inf.predict('cat.jpg'))
print(learn_inf.dls.vocab)