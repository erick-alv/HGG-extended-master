from .hgg import HGGLearner, NormalLearner

learner_collection = {
	'normal': NormalLearner,
	'hgg': HGGLearner,
}

def create_learner(args):
	return learner_collection[args.learn](args)